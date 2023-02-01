import argparse
import dask.dataframe as dd
import json
import math
import numpy as np
import os

# import pandas as pd
import pickle
import pycountry
import sys
import wandb


from dataclasses import dataclass
from names_dataset import NameDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dotenv import load_dotenv

# load env variables
load_dotenv()
# uncomment below to run Dask dashboard on http://127.0.0.1:8787
# from dask.distributed import Client
# client = Client(n_workers=2)
# client

# we use /app folder as base_path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# and we add it at the beginning of PYTHONPATH otherwise we
# cannot import from modules at the same level
sys.path.insert(0, base_path)

# local imports
# from utils.dev_utils import load_raw_data, drop_duplicates  # noqa: E402
from utils.prod_utils import remove_digits_punctuation_doublespaces  # noqa: E402


def data_preparation_pipeline(args):
    config_filepath = os.path.join(base_path, "config.json")
    global CONFIG
    CONFIG = json.load(open(config_filepath))
    wandb.login()
    global PROJECT_NAME
    PROJECT_NAME = f"{args.tag}-clf"
    if args.build_raw_data:
        print("*** Building raw data ***")
        with wandb.init(project=PROJECT_NAME, job_type="upload") as run:
            CONFIG[PROJECT_NAME] = {
                "artifacts": {"raw-data": {"filename": f"raw-data-top{args.top}"}}
            }
            raw_data_filename = CONFIG[PROJECT_NAME]["artifacts"]["raw-data"][
                "filename"
            ]
            raw_data_root_dir = os.path.join(base_path, "artifacts/data")
            CONFIG[PROJECT_NAME]["artifacts"]["raw-data"][
                "root_dir"
            ] = raw_data_root_dir
            raw_data_filepath = os.path.join(
                raw_data_root_dir, raw_data_filename + ".parquet.gzip"
            )
            CONFIG[PROJECT_NAME]["artifacts"]["raw-data"][
                "filepath"
            ] = raw_data_filepath
            print("↓↓↓ Saving raw data as artifact...")
            data = build_dataframe(args.top)
            data.to_parquet(raw_data_filepath, compression="gzip")
            raw_data_artifact = wandb.Artifact(
                raw_data_filename,
                type="dataset",
                description="raw data generated from name_dataset: https://pypi.org/project/names-dataset/",  # noqa: E501
                metadata={
                    "shape": data.shape,
                    "columns": list(data.columns),
                    "countries": list(data.Country.unique()),
                    "top_names_per_gender_per_country": args.top,
                },
            )
            raw_data_artifact.add_file(local_path=raw_data_filepath)
            run.log_artifact(raw_data_artifact)
    else:
        with wandb.init(project=PROJECT_NAME, job_type="load") as run:
            # load latest artifact from wandb
            raw_data_filename = CONFIG[PROJECT_NAME]["artifacts"]["raw-data"][
                "filename"
            ]
            raw_data_root_dir = CONFIG[PROJECT_NAME]["artifacts"]["raw-data"][
                "root_dir"
            ]
            raw_data_filepath = CONFIG[PROJECT_NAME]["artifacts"]["raw-data"][
                "filepath"
            ]
            run.use_artifact(raw_data_filename + ":latest").download(raw_data_root_dir)
            print("↑↑↑ Loading raw data from wandb...")
            data = dd.read_parquet(raw_data_filepath).compute()

    print("raw data shape = ", data.shape)
    # data cleaning / standardization (no need to log this to wandb)
    data = data.dropna()
    # lower
    data["Name"] = data["Name"].str.lower()
    # remove noise
    data["Name"] = data["Name"].apply(
        lambda x: remove_digits_punctuation_doublespaces(x)
    )

    max_name_lenght = data["Name"].str.len().max()
    assert max_name_lenght > 0
    # # input shape is the smallest multiple of power of 2 larger than max_name_lenght
    input_shape = 2 ** (math.ceil(math.log(max_name_lenght, 2)))
    print(f"max name lenght : {max_name_lenght} => input_shape : {input_shape}")
    CONFIG[PROJECT_NAME]["input_shape"] = input_shape
    # Prepare country datsets for training
    training_set = get_training_data(
        args.tag, data, args.train_val_test_size, input_shape
    )

    # save updates to CONFIG file
    with open(config_filepath, "w") as f:
        json.dump(CONFIG, f, indent=2)

    return training_set


def build_dataframe(top):
    nd = NameDataset()

    country_codes = nd.get_country_codes(alpha_2=True)
    country_mapping = {}
    for country_code in country_codes:
        country_name = pycountry.countries.get(alpha_2=country_code).name
        country_mapping[country_code] = country_name

    subset = ("ES", "FR", "DE")  # ("ES", "IT", "FR", "DE", "GB")
    europe = dict(filter(lambda i: i[0] in subset, country_mapping.items()))
    entries = {}
    names = []
    countries = []
    genders = []
    for key, value in europe.items():
        print(value)
        results = nd.get_top_names(n=top, country_alpha2=key)
        for name_male in results[key]["M"]:
            names.append(name_male)
            countries.append(value)
            genders.append("Male")

        for name_female in results[key]["F"]:
            names.append(name_female)
            countries.append(value)
            genders.append("Female")

    entries["Name"] = names
    entries["Country"] = countries
    entries["Gender"] = genders

    df = dd.from_dict(entries, npartitions=10).compute()
    print(f"Dataframe loaded: {df.shape}")

    return df


def get_training_data(tag, data, train_val_test_size, input_shape):
    target = tag
    # Split the data
    split_list = list(map(float, train_val_test_size.split(" ")))
    assert len(split_list) == 3
    CONFIG[PROJECT_NAME]["train_val_test_size"] = split_list
    split_list = np.divide(split_list, 100.0)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        data["Name"],
        data[target],
        shuffle=True,
        test_size=split_list[2],
        stratify=data[target],
    )
    assert y_trainval.nunique() == y_test.nunique()
    val_size = split_list[1] / (split_list[0] + split_list[1])
    print("val_size = ", val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        stratify=y_trainval,
    )
    assert y_train.nunique() == y_val.nunique()
    print(f"{target} training examples: ", len(X_train))
    print(f"{target} validation examples: ", len(X_val))
    print(f"{target} test examples: ", len(X_test))
    X_datasets = np.array([X_train, X_val, X_test], dtype=object)
    y_datasets = np.array([y_train, y_val, y_test], dtype=object)

    # tokenize on training data (new tokens found in val/test will be tokenized as UNK)
    tk = text_vectorization(target, X_train)
    # convert data to padded sequences
    X_datasets = padded_sequences(X_datasets, tk, input_shape)
    # one-hot encode targets
    y_datasets = ohe_classes(y_datasets, target)
    print(f"↓↓↓ Saving {target} datasets as artifact...")
    CONFIG[PROJECT_NAME]["artifacts"]["datasets"] = {"filename": f"{tag}-datasets"}
    datasets_filename = CONFIG[PROJECT_NAME]["artifacts"]["datasets"]["filename"]
    datasets_root_dir = os.path.join(base_path, "artifacts/data")
    CONFIG[PROJECT_NAME]["artifacts"]["datasets"]["root_dir"] = datasets_root_dir
    datasets_filepath = os.path.join(datasets_root_dir, datasets_filename + ".npz")
    CONFIG[PROJECT_NAME]["artifacts"]["datasets"]["filepath"] = datasets_filepath
    np.savez_compressed(
        datasets_filepath,
        X_train=X_datasets[0],
        y_train=y_datasets[0],
        X_val=X_datasets[1],
        y_val=y_datasets[1],
        X_test=X_datasets[2],
        y_test=y_datasets[2],
    )
    # return datasets as instance of @dataclass
    datasets = DataSets(
        X_datasets[0],
        y_datasets[0],
        X_datasets[1],
        y_datasets[1],
        X_datasets[2],
        y_datasets[2],
    )
    with wandb.init(project=PROJECT_NAME, job_type="upload") as run:
        datasets_artifact = wandb.Artifact(
            datasets_filename,
            type="dataset",
            description=f"Pre-processed train, validation and test datasets for {tag} classification",  # noqa: E501
            metadata={
                "examples": {
                    "training": len(X_train),
                    "validation": len(X_val),
                    "test": len(X_test),
                },
                "shape-after-tokenization-and-padding": {
                    "X_train": X_datasets[0].shape,
                    "y_train": y_datasets[0].shape,
                    "X_val": X_datasets[1].shape,
                    "y_val": y_datasets[1].shape,
                    "X_test": X_datasets[2].shape,
                    "y_test": y_datasets[2].shape,
                },
            },
        )
        datasets_artifact.add_file(local_path=datasets_filepath)
        run.log_artifact(datasets_artifact)

    return datasets


def text_vectorization(target, X_train):
    # Text Vectorization
    tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK")
    tk.fit_on_texts(X_train)
    print("vocabulary lenght : ", len(tk.word_index))

    # Save tokenizer
    print(f"↓↓↓ Saving {target} tokenizer as artifact...")
    CONFIG[PROJECT_NAME]["artifacts"]["tokenizer"] = {"filename": f"{target}-tokenizer"}
    tokenizer_filename = CONFIG[PROJECT_NAME]["artifacts"]["tokenizer"]["filename"]
    tokenizer_root_dir = os.path.join(base_path, "artifacts")
    CONFIG[PROJECT_NAME]["artifacts"]["tokenizer"]["root_dir"] = tokenizer_root_dir
    tokenizer_filepath = os.path.join(tokenizer_root_dir, tokenizer_filename + ".pkl")
    CONFIG[PROJECT_NAME]["artifacts"]["tokenizer"]["filepath"] = tokenizer_filepath
    with open(
        tokenizer_filepath,
        "wb",
    ) as f:
        pickle.dump(tk, f, protocol=pickle.HIGHEST_PROTOCOL)
    # upload tokenizer to wandb
    wandb.login()
    with wandb.init(project=PROJECT_NAME, job_type="upload") as run:
        tokenizer_artifact = wandb.Artifact(
            tokenizer_filename,
            type="tokenizer",
            description="tokenizer built on training set",  # noqa: E501
            metadata={"vocabulary-length": len(tk.word_index), "tokens": tk.word_index},
        )
        tokenizer_artifact.add_file(local_path=tokenizer_filepath)
        run.log_artifact(tokenizer_artifact)

    return tk


def padded_sequences(X_datasets, tk, INPUT_SHAPE):
    result = []
    for dataset in X_datasets:
        sequence = tk.texts_to_sequences(dataset)
        assert max(map(len, sequence)) <= INPUT_SHAPE
        # Padding with 0s
        padded = pad_sequences(sequence, maxlen=INPUT_SHAPE, padding="post")
        # Convert to numpy array
        nparray = np.array(padded, dtype="float32")
        result.append(nparray)

    return np.array(result, dtype=object)


def ohe_classes(y_datasets, target):
    ohe = OneHotEncoder(sparse=False)
    result = []
    for dataset in y_datasets:
        transformed = ohe.fit_transform(dataset.to_numpy().reshape(-1, 1))
        result.append(transformed)

    print("one hot encoded categories: ", list(ohe.categories_)[0])
    print(f"↓↓↓ Saving {target} one-hot-encoder as artifact...")
    CONFIG[PROJECT_NAME]["artifacts"]["one-hot-encoder"] = {"filename": f"{target}-ohe"}
    ohe_filename = CONFIG[PROJECT_NAME]["artifacts"]["one-hot-encoder"]["filename"]
    ohe_root_dir = os.path.join(base_path, "artifacts")
    CONFIG[PROJECT_NAME]["artifacts"]["one-hot-encoder"]["root_dir"] = ohe_root_dir
    ohe_filepath = os.path.join(ohe_root_dir, ohe_filename + ".pkl")
    CONFIG[PROJECT_NAME]["artifacts"]["one-hot-encoder"]["filepath"] = ohe_filepath
    with open(ohe_filepath, "wb") as f:
        pickle.dump(ohe, f)

    # upload ohe to wandb
    wandb.login()
    with wandb.init(project=PROJECT_NAME, job_type="upload") as run:
        ohe_artifact = wandb.Artifact(
            ohe_filename,
            type="one-hot-encoder",
            description=f"one hot encoder for {target}",  # noqa: E501
            metadata={"ohe-categories": list(ohe.categories_)[0]},
        )
        ohe_artifact.add_file(local_path=ohe_filepath)
        run.log_artifact(ohe_artifact)

    return np.array(result, dtype=object)


@dataclass
class DataSets:
    X_train: float
    y_train: float
    X_val: float
    y_val: float
    X_test: float
    y_test: float


if __name__ == "__main__":
    # python app/pipelines/data_preparation.py --tag="Country" --build_raw_data
    # Parse args
    docstring = """When running this script as main you can specify the filepath for the raw data to be prepared """  # noqa: E501
    parser = argparse.ArgumentParser(
        description=docstring,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        default="Country",
        type=str,
        choices=["Country", "Gender"],
        help="tag identifying the classifier to tune {'Country' | 'Gender'}",
    )
    # start data prep from the beginning
    parser.add_argument(
        "--build_raw_data", default=False, action=argparse.BooleanOptionalAction
    )
    # if build_raw_data==True, chose nb. of top male AND female names to use
    parser.add_argument("--top", default=100, type=int)
    # train / validation / test size (default: 80% 10% 10%)
    parser.add_argument("--train_val_test_size", nargs="+", default="80 10 10")
    args = parser.parse_args()
    # tag = args.tag
    # build_raw_data = args.build_raw_data
    # top = args.top
    # train_val_test_size = args.train_val_test_size

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key is None:
        print(
            "ERROR: Weights and Biases integration failed. You need a wandb account to run this script"  # noqa: E501
        )
    else:
        print("=== Data preparation pipeline ===")
        data_preparation_pipeline(args)
        print("=== Finished ===")
