import argparse
import math
import numpy as np
import os
import pickle
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# local imports
import sys

# we use /app folder as base_path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# and we add it at the beginning of PYTHONPATH otherwise we
# cannot import from modules at the same level
sys.path.insert(0, base_path)

from utils.dev_utils import load_raw_data, drop_duplicates  # noqa: E402
from utils.prod_utils import remove_digits_punctuation_doublespaces  # noqa: E402


def data_preparation_pipeline(validation_size_gender, validation_size_nationality):

    filepath = os.path.join(base_path, "data", "names-by-nationality.csv")
    print("raw_data_filepath = ", filepath)
    try:
        data = load_raw_data(filepath)
        data.rename(columns={"sex": "gender"}, inplace=True)
    except OSError as error:
        print(error)
        return

    # data cleaning / standardization
    data.dropna(inplace=True)
    # lower
    data["name"] = data["name"].str.lower()
    # remove noise
    data["name"] = data["name"].apply(
        lambda x: remove_digits_punctuation_doublespaces(x)
    )
    # remove duplicates
    # TODO: some names can belong to different nationalities, though...
    data = drop_duplicates(data, ["name"])
    max_name_lenght = data["name"].str.len().max()
    assert max_name_lenght > 0
    # input shape is the smallest multiple of power of 2 larger than max_name_lenght
    INPUT_SHAPE = 2 ** (math.ceil(math.log(max_name_lenght, 2)))
    print(f"max name lenght : {max_name_lenght} => INPUT_SHAPE : {INPUT_SHAPE}")

    # Prepare gender datsets for training
    target = "gender"
    gender_training_set = get_training_data(
        target, data, validation_size_gender, INPUT_SHAPE
    )

    # Prepare nationality datsets for training
    target = "nationality"
    nationality_training_set = get_training_data(
        target, data, validation_size_gender, INPUT_SHAPE
    )

    return gender_training_set, nationality_training_set


def get_training_data(target, data, val_size, INPUT_SHAPE):
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        data["name"],
        data[target],
        test_size=val_size,
        stratify=data[target],
    )
    assert y_train.nunique() == y_val.nunique()
    print(f"{target} training examples: ", len(X_train))
    print(f"{target} validation examples: ", len(X_val))

    # tokenize on gender training data
    tk = text_vectorization(target, X_train)
    # convert data to padded sequences
    X_train, X_val = padded_sequences(X_train, X_val, tk, INPUT_SHAPE)
    # one-hot encode targets
    y_train, y_val = ohe_classes(y_train, y_val, target)
    print(f"↓↓↓ Saving {target} training data as artifact...")
    np.savez_compressed(
        os.path.join(base_path, "artifacts/training_data", f"{target}_data.npz"),
        x_train=X_train,
        y_train=y_train,
        x_val=X_val,
        y_val=y_val,
    )
    training_set = TrainingSet(X_train, y_train, X_val, y_val)

    return training_set


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
    # Parse args
    docstring = """When running this script as main you can specify the filepath for the raw data to be prepared """  # noqa: E501
    parser = argparse.ArgumentParser(
        description=docstring,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # optional validation size for gender (default: 15%)
    parser.add_argument("--val_size_gen", nargs="?", default=0.15)
    # optional validation size for nationality (default: 15%)
    parser.add_argument("--val_size_nat", nargs="?", default=0.15)
    args = parser.parse_args()
    validation_size_gender = args.val_size_gen
    validation_size_nationality = args.val_size_nat
    print("*** Data preparation")
    data_preparation_pipeline(validation_size_gender, validation_size_nationality)
    print("*** Finished ***")
