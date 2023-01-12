import argparse
import json
import numpy as np
import os
import pickle
import sys

# import tensorflow as tf
import wandb

from dotenv import load_dotenv

load_dotenv()

# we use /app folder as base_path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# and we add it at the beginning of PYTHONPATH otherwise we
# cannot import from modules at the same level
sys.path.insert(0, base_path)

# local imports
from models.model import CNN_model  # noqa: E402


def main():
    wandb_key = os.environ.get("WANDB_KEY")
    if wandb_key is None:
        print(
            "ERROR: Weights and Biases integration failed. You need a wandb account to run this script"  # noqa: E501
        )
        return

    print("== Load artifacts ==")
    with np.load(
        os.path.join(base_path, "artifacts/training_data", f"{clf_tag}_data.npz"),
        allow_pickle=True,
    ) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_val, y_val = f["x_val"], f["y_val"]
    print(x_val, y_val)

    # print("x_train shape = ", x_train.shape)
    # print("x_val shape = ", x_val.shape)
    # Batch data
    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
    # val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)
    print("Training data loaded")

    with open(
        os.path.join(base_path, "artifacts/training_data", f"{clf_tag}_tokenizer.pkl"),
        "rb",
    ) as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded")

    print("== Build the model ==")
    # Model params from data
    print("model parameters:")
    embedding_size = len(tokenizer.word_index)
    print("embedding_size = ", embedding_size)
    softmax_units = y_train.shape[1]
    print("softmax_units = ", softmax_units)
    input_shape = x_train.shape[1]
    print("input_shape = ", input_shape)
    tune(embedding_size, softmax_units, input_shape)

    # print(
    #     "some_arg_for_tuning_like_max_epochs = ",
    #     some_arg_for_tuning_like_max_epochs,
    # )
    # print("data_tag = ", data_tag)
    # # load training data from /data folder or cloud using data_tag
    # # get your tuner..
    # some_tuner()
    # # ..or pipe to tune
    # some_pipe()
    # print("=== Tuning === ")
    # # define search space / param_grid
    # # perform your favourite search of parameters specific to your model
    # # obtain :
    # best_model = None
    # print("=== Evaluation === ")
    # # evaluate the model on the validaiton set
    # # produce some plots for reporting
    # history_or_confusion_matrix = None
    # model_performance_plot(history_or_confusion_matrix)
    # # save best model in local /data folder and/or in the cloud
    best_model = None

    return best_model


def get_sweep_config(clf_tag):
    try:
        with open(
            os.path.join(base_path, "artifacts/models", f"{clf_tag}_sweep_config.json"),
            "r",
        ) as f:
            sweep_config = json.load(f)
            print(f"configuration for {clf_tag} classifier tuning loaded")
    except OSError as err:
        raise Exception(detail=repr(err))

    return sweep_config


def tune(embedding_size, softmax_units, input_shape):
    print("Hello tune")
    project_name = f"{clf_tag}-CNN"
    # run = wandb.init(project=project_name)
    model = CNN_model(embedding_size, softmax_units, input_shape, project_name)
    print(model.summary())
    sweep_config = get_sweep_config(clf_tag)
    print(sweep_config)
    wandb.login()
    wandb.init(project=project_name)  # , entity="fratambot")
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    print(sweep_id)


if __name__ == "__main__":
    # Parse args
    docstring = """Script for data tuning using wandb"""
    parser = argparse.ArgumentParser(
        description=docstring,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clf_tag",
        required=True,
        default="Country",
        type=str,
        choices=["Country", "Gender"],
        help="tag identifying the classifier to tune {'Country' | 'Gender'}",
    )

    args = parser.parse_args()
    clf_tag = args.clf_tag

    # python app/pipelines/tune.py --clf_tag="nationality"
    main()
    print("=== Finished ===")
