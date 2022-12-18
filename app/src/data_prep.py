import argparse
import numpy as np
import os
import pandas as pd
import pathlib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# local imports
from utils.dev_utils import load_raw_data, drop_duplicates
from utils.prod_utils import unidecode_string, remove_digits_punctuation_doublespaces

def data_prep():

    filepath = os.path.join("data", "names-by-nationality.csv")
    print("raw_data_filepath = ", filepath)
    try:
        data = load_raw_data(filepath)
        print(data.shape)
    except OSError as error:
        print(error)
        #print(f"Raw data not found at {filepath}")


if __name__ == "__main__":
    # Parse args
    # docstring = """When running this script as main you can specify the filepath for the raw data to be prepared """  # noqa: E501
    # parser = argparse.ArgumentParser(
    #     description=docstring,
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    # )
    # # raw data filepath
    # default = os.path.join("../data", "names-by-nationality.csv")
    # parser.add_argument(
    #     "raw_data_filepath", nargs="?", default=default, type=pathlib.Path
    # )
    # args = parser.parse_args()
    # raw_data_filepath = args.raw_data_filepath.resolve()
    print("*** Data preparation")
    data_prep()
    print("*** Finished ***")
