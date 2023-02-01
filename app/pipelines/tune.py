import argparse
import json
import numpy as np
import os
import pickle
import sys

# import tensorflow as tf
import wandb

from dotenv import load_dotenv

# from wandb.keras import WandbMetricsLogger

# load env variables
load_dotenv()

# we use /app folder as base_path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# and we add it at the beginning of PYTHONPATH otherwise we
# cannot import from modules at the same level
sys.path.insert(0, base_path)

# local imports
from models.model import CNN_LSTM_model  # noqa: E402
from pipelines.data_preparation import data_preparation_pipeline  # noqa: E402


def main(tag, launch_data_prep, model_selection):
    config_filepath = os.path.join(base_path, "config.json")
    global CONFIG
    CONFIG = json.load(open(config_filepath))
    wandb.login()
    global PROJECT_NAME
    PROJECT_NAME = f"{tag}-clf"
    if launch_data_prep:
        # TODO: allow to start from beginning (i.e. raw data) ?
        build_raw_data = False
        train_val_test_size = "80 10 10"
        datasets = data_preparation_pipeline(tag, build_raw_data, train_val_test_size)
        X_train, y_train = datasets.X_train, datasets.y_train
        X_val, y_val = datasets.X_val, datasets.y_val
        X_test, y_test = datasets.X_test, datasets.y_test

    else:
        # load latest datasets from wandb
        with wandb.init(project=PROJECT_NAME, job_type="load") as run:
            datasets_filename = CONFIG[PROJECT_NAME]["artifacts"]["datasets"][
                "filename"
            ]
            datasets_root_dir = CONFIG[PROJECT_NAME]["artifacts"]["datasets"][
                "root_dir"
            ]
            datasets_filepath = CONFIG[PROJECT_NAME]["artifacts"]["datasets"][
                "filepath"
            ]

            run.use_artifact(datasets_filename + ":latest").download(datasets_root_dir)
            print("↑↑↑ Loading datasets from wandb...")
            with np.load(
                datasets_filepath,
                allow_pickle=True,
            ) as f:
                X_train, y_train = f["X_train"], f["y_train"]
                X_val, y_val = f["X_val"], f["y_val"]
                X_test, y_test = f["X_test"], f["y_test"]

    training_set = (X_train, y_train)
    validation_set = (X_val, y_val)
    test_set = (X_test, y_test)

    print(training_set.shape)
    print(validation_set.shape)
    print(test_set.shape)
    # Batch data
    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64)
    # val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
    print("Training data loaded")

    # load tokenizer from wandb:
    with wandb.init(project=PROJECT_NAME, job_type="load") as run:
        tokenizer_filename = CONFIG[PROJECT_NAME]["artifacts"]["tokenizer"]["filename"]
        tokenizer_root_dir = CONFIG[PROJECT_NAME]["artifacts"]["tokenizer"]["root_dir"]
        tokenizer_filepath = CONFIG[PROJECT_NAME]["artifacts"]["tokenizer"]["filepath"]
        run.use_artifact(tokenizer_filename + ":latest").download(tokenizer_root_dir)
        with open(tokenizer_filepath, "rb") as f:
            tokenizer = pickle.load(f)
    print("Tokenizer loaded")
    # params from artifacts
    data_config = {
        "input_shape": CONFIG[PROJECT_NAME]["input_shape"],
        "embedding_size": len(tokenizer.word_index),
        "softmax_units": y_train.shape[1],
    }
    print(data_config)
    print("== Build the model ==")
    # initialize with default params:
    # model = build_model(model_selection, data_config)
    # simple train
    # history = train(model_selection, model, training_set, validation_set)

    # print("history = ", history)

    best_model = None

    # save updates to CONFIG file
    with open(config_filepath, "w") as f:
        json.dump(CONFIG, f, indent=2)

    return best_model


def build_model(model_selection, config):
    # run wandb agent to initialize the model of your choice and log it
    if model_selection == "CNN_LSTM":
        default_config = {"dense_1": 64}
        default_config.update(config)
        print(default_config)
        with wandb.init(
            project=PROJECT_NAME, job_type="initialize", config=default_config
        ) as run:
            def_config = wandb.config
            model = CNN_LSTM_model(**def_config)
            CONFIG[PROJECT_NAME]["artifacts"]["models"] = {
                f"{model_selection}": {"filename": f"{model_selection}_model"}
            }
            model_filename = CONFIG[PROJECT_NAME]["artifacts"]["models"][
                f"{model_selection}"
            ]["filename"]
            model_root_dir = os.path.join(base_path, "artifacts/models")
            CONFIG[PROJECT_NAME]["artifacts"]["models"][f"{model_selection}"][
                "root_dir"
            ] = model_root_dir
            model_filepath = os.path.join(model_root_dir, model_filename + ".h5")
            CONFIG[PROJECT_NAME]["artifacts"]["models"][f"{model_selection}"][
                "filepath"
            ] = model_filepath
            # saving default config (a.k.a. metadata in wandb)
            CONFIG[PROJECT_NAME]["artifacts"]["models"][f"{model_selection}"][
                "config"
            ] = dict(def_config)
            model.save(model_filepath)
            model_artifact = wandb.Artifact(
                model_filename,
                type="model",
                description="A simple CNN+LSTM classifier",
                metadata=dict(def_config),
            )
            model_artifact.add_file(local_path=model_filepath)
            wandb.save(base_path=model_filepath)
            run.log_artifact(model_artifact)
    # elif select_model=="":
    else:
        print("model not found")
    return model


# def train(model_selection, model, training_set, validation_set):
#     # Just for testing. Move to train pipeline
#     print("Hello train")
#     model_config = CONFIG[PROJECT_NAME]["artifacts"]["models"][f"{model_selection}"][
#         "config"
#     ]
#     train_config = {"epochs": 10}
#     model_config.update(train_config)
#     print(model_config)
#     with wandb.init(project=PROJECT_NAME, job_type="train", config=model_config) as run: # noqa: E501
#         mod_config = wandb.config
#         history = model.fit(
#             training_set[0],
#             training_set[1],
#             epochs=mod_config.epochs,
#             batch_size=64,
#             verbose=1,
#             validation_data=validation_set,
#             callbacks=[WandbMetricsLogger()],
#         )

#     return history


# def get_sweep_config(clf_tag):
#     try:
#         with open(
#             os.path.join(base_path, "artifacts/models", f"{clf_tag}_sweep_config.json"), # noqa: E501
#             "r",
#         ) as f:
#             sweep_config = json.load(f)
#             print(f"configuration for {clf_tag} classifier tuning loaded")
#     except OSError as err:
#         raise Exception(detail=repr(err))

#     return sweep_config


# def tune(model_selection, model, training_set, validation_set):
#     print("Hello tune")

#     # run = wandb.init(project=project_name)

#     print(sweep_config)
#     wandb.init(project=project_name)  # , entity="fratambot")
#     sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
#     print(sweep_id)


if __name__ == "__main__":
    # python app/pipelines/tune.py --tag="Country"
    # Parse args
    docstring = """Script for data tuning using wandb"""
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
    parser.add_argument(
        "--launch_data_prep", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--model_selection",
        default="CNN_LSTM",
        type=str,
        choices=["CNN_LSTM", "CNN"],
        help="tag identifying the model to use {'CNN_LSTM' | 'CNN'}",
    )
    args = parser.parse_args()
    tag = args.tag
    launch_data_prep = args.launch_data_prep
    model_selection = args.model_selection

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key is None:
        print(
            "ERROR: Weights and Biases integration failed. You need a wandb account to run this script"  # noqa: E501
        )
    else:
        print("=== Model tuning pipeline ===")
        main(tag, launch_data_prep, model_selection)
        print("=== Finished ===")
