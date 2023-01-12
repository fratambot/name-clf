import argparse
import math
import numpy as np
import os
import pandas as pd
import pickle
import pycountry
import sys

# import wandb

from dataclasses import dataclass
from names_dataset import NameDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dotenv import load_dotenv

load_dotenv()


# we use /app folder as base_path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# and we add it at the beginning of PYTHONPATH otherwise we
# cannot import from modules at the same level
sys.path.insert(0, base_path)

# local imports
# from utils.dev_utils import load_raw_data, drop_duplicates  # noqa: E402
# from utils.prod_utils import remove_digits_punctuation_doublespaces  # noqa: E402


def data_preparation_pipeline(tag, build_raw_data, train_val_test_size):
    # wandb.login()
    # PROJECT_NAME = f"{tag}-clf"
    if build_raw_data:
        # run = wandb.init(project=PROJECT_NAME, job_type="upload")
        data = build_dataframe()
        # TODO: save to file
        # raw_data_artifact = wandb.Artifact("raw_data_ES", type="raw_data")
    else:
        data = pd.DataFrame()
    # data cleaning / standardization
    data.dropna(inplace=True)
    # lower
    data["Name"] = data["Name"].str.lower()
    # remove noise
    # data["Name"] = data["Name"].apply(
    #     lambda x: remove_digits_punctuation_doublespaces(x)
    # )

    max_name_lenght = data["Name"].str.len().max()
    assert max_name_lenght > 0
    # # input shape is the smallest multiple of power of 2 larger than max_name_lenght
    INPUT_SHAPE = 2 ** (math.ceil(math.log(max_name_lenght, 2)))
    print(f"max name lenght : {max_name_lenght} => INPUT_SHAPE : {INPUT_SHAPE}")

    # Prepare nationality datsets for training
    target = "Country"
    country_training_set = get_training_data(
        target, data, train_val_test_size, INPUT_SHAPE
    )

    return country_training_set


def build_dataframe():
    nd = NameDataset()

    country_codes = nd.get_country_codes(alpha_2=True)
    country_mapping = {}
    for country_code in country_codes:
        country_name = pycountry.countries.get(alpha_2=country_code).name
        country_mapping[country_code] = country_name

    subset = "ES"  # , "IT", "FR", "DE", "GB")
    europe = dict(filter(lambda i: i[0] in subset, country_mapping.items()))
    entries = []
    for key, value in europe.items():
        print(key)
        result_male = nd.get_top_names(n=2000, gender="Male", country_alpha2=key)
        for name_male in result_male[key]["M"]:
            entries.append({"Name": name_male, "Country": value, "Gender": "Male"})
        result_female = nd.get_top_names(n=2000, gender="Female", country_alpha2=key)
        for name_female in result_female[key]["F"]:
            entries.append({"Name": name_female, "Country": value, "Gender": "Female"})

    df = pd.DataFrame(entries)
    print(f"Dataframe loaded: {df.shape}")

    return df


def get_training_data(target, data, train_val_test_size, INPUT_SHAPE):
    # Split the data

    split_list = list(map(float, train_val_test_size.split(" ")))
    assert len(split_list) == 3
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

    # print(f"{target} training examples: ", len(X_train))
    # print(f"{target} test examples: ", len(X_test))

    # # tokenize on gender training data
    # tk = text_vectorization(target, X_train)
    # # convert data to padded sequences
    # X_train, X_val = padded_sequences(X_train, X_val, tk, INPUT_SHAPE)
    # # one-hot encode targets
    # y_train, y_val = ohe_classes(y_train, y_val, target)
    # print(f"↓↓↓ Saving {target} training data as artifact...")
    # np.savez_compressed(
    #     os.path.join(base_path, "artifacts/training_data", f"{target}_data.npz"),
    #     x_train=X_train,
    #     y_train=y_train,
    #     x_val=X_val,
    #     y_val=y_val,
    # )
    # training_set = TrainingSet(X_train, y_train, X_val, y_val)

    return  # training_set


def text_vectorization(target, X_train):
    # Text Vectorization
    tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK")
    tk.fit_on_texts(X_train)
    print(f"{target} vocabulary lenght : ", len(tk.word_index))

    # Save tokenizer
    print(f"↓↓↓ Saving {target} tokenizer as artifact...")
    with open(
        os.path.join(base_path, "artifacts/training_data", f"{target}_tokenizer.pkl"),
        "wb",
    ) as f:
        pickle.dump(tk, f, protocol=pickle.HIGHEST_PROTOCOL)

    return tk


def padded_sequences(X_train, X_val, tk, INPUT_SHAPE):
    train_sequences = tk.texts_to_sequences(X_train)
    val_sequences = tk.texts_to_sequences(X_val)
    assert max(map(len, train_sequences)) <= INPUT_SHAPE
    assert max(map(len, val_sequences)) <= INPUT_SHAPE

    # Padding with 0s
    X_train = pad_sequences(train_sequences, maxlen=INPUT_SHAPE, padding="post")
    X_val = pad_sequences(val_sequences, maxlen=INPUT_SHAPE, padding="post")
    # Convert to numpy array
    X_train = np.array(X_train, dtype="float32")
    X_val = np.array(X_val, dtype="float32")

    return X_train, X_val


def ohe_classes(y_train, y_val, target):
    ohe = OneHotEncoder(sparse=False)
    y_train = ohe.fit_transform(y_train.to_numpy().reshape(-1, 1))
    y_val = ohe.fit_transform(y_val.to_numpy().reshape(-1, 1))
    print("one hot encoded categories: ", ohe.categories_)
    print(f"↓↓↓ Saving {target} one-hot-encoder as artifact...")
    with open(
        os.path.join(base_path, "artifacts/training_data", f"{target}_ohe.pkl"), "wb"
    ) as f:
        pickle.dump(ohe, f)

    return y_train, y_val


@dataclass
class TrainingSet:
    X_train: float
    y_train: float
    X_val: float
    y_val: float


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
        required=True,
        default="Country",
        type=str,
        choices=["Country", "Gender"],
        help="tag identifying the classifier to tune {'Country' | 'Gender'}",
    )
    parser.add_argument("--build_raw_data", action=argparse.BooleanOptionalAction)
    # optional train / validation / test size (default: 80% 10% 10%)
    parser.add_argument("--train_val_test_size", nargs="+", default="80 10 10")
    args = parser.parse_args()
    tag = args.tag
    build_raw_data = args.build_raw_data
    train_val_test_size = args.train_val_test_size
    wandb_key = os.environ.get("WANDB_KEY")
    if wandb_key is None:
        print(
            "ERROR: Weights and Biases integration failed. You need a wandb account to run this script"  # noqa: E501
        )
    else:
        print("*** Data preparation ***")
        data_preparation_pipeline(tag, build_raw_data, train_val_test_size)
        print("*** Finished ***")
