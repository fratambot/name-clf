import keras_tuner
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl


class CNNLSTMHyperModel(keras_tuner.HyperModel):
    def __init__(self, input_shape, metrics, softmax_units, embedding_size):
        self.input_shape = input_shape
        self.metrics = metrics
        self.softmax_units = softmax_units
        self.embedding_size = embedding_size

    def build(self, hp):
        # Build matrix for embedding weights initialization
        embedding_weigths = np.concatenate(
            (
                [np.zeros((self.embedding_size), dtype=int)],
                np.identity(self.embedding_size, dtype="int"),
            ),
            axis=0,
        )

        # Build Sequential model
        model = tf.keras.Sequential()
        model.add(
            tfl.Input(shape=(self.input_shape,), name="input_layer", dtype="int64")
        )
        model.add(
            tfl.Embedding(
                self.embedding_size + 1,
                self.embedding_size,
                input_length=self.input_shape,
                weights=[embedding_weigths],
            )
        )
        model.add(tfl.Conv1D(filters=16, kernel_size=7, strides=1, activation="relu"))
        model.add(tfl.MaxPooling1D(pool_size=2))
        # Drop 1 (tune)
        model.add(
            tfl.Dropout(
                rate=hp.Float(
                    "dropout_1",
                    min_value=0.0,
                    max_value=0.5,
                    default=0.3,
                    step=0.1,
                )
            )
        )
        model.add(tfl.LSTM(8, return_sequences=True))

        model.add(tfl.Flatten())
        # Dense 1 (tune)
        model.add(
            tfl.Dense(
                units=hp.Choice("dense_units_1", values=[64, 128, 256]),
                activation="relu",
            )
        )
        # Drop 2 (tune)
        model.add(
            tfl.Dropout(
                rate=hp.Float(
                    "dropout_2",
                    min_value=0.0,
                    max_value=0.5,
                    default=0.5,
                    step=0.1,
                )
            )
        )
        # Dense 2 (tune)
        model.add(
            tfl.Dense(
                units=hp.Choice("dense_units_2", values=[64, 128, 256]),
                activation="relu",
            )
        )
        # Softmax
        model.add(tfl.Dense(units=self.softmax_units, activation="softmax"))
        # Learning rate (tune)
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=self.metrics,
        )

        return model


def hyperband_tuner(
    input_shape,
    metrics,
    softmax_units,
    embedding_size,
    project_name,
    # max_tuning_epochs,
    directory,
):
    hypermod = CNNLSTMHyperModel(input_shape, metrics, softmax_units, embedding_size)

    tuner = keras_tuner.Hyperband(
        hypermodel=hypermod,
        objective="val_loss",
        max_epochs=20,
        factor=3,
        hyperband_iterations=1,
        overwrite=True,
        directory=directory,
        project_name=project_name,
        allow_new_entries=True,
        tune_new_entries=True,
    )

    return tuner


def model(input_shape, metrics, softmax_units, embedding_size):
    # Build matrix for embedding weights initialization
    print(input_shape)
    embedding_weigths = np.concatenate(
        (
            [np.zeros((embedding_size), dtype=int)],
            np.identity(embedding_size, dtype="int"),
        ),
        axis=0,
    )

    # Build Sequential model
    model = tf.keras.Sequential()
    model.add(tfl.Input(shape=(input_shape,), name="input_layer", dtype="int64"))
    model.add(
        tfl.Embedding(
            embedding_size + 1,
            embedding_size,
            input_length=input_shape,
            weights=[embedding_weigths],
        )
    )
    model.add(
        tfl.Conv1D(filters=input_shape, kernel_size=7, strides=1, activation="relu")
    )
    model.add(tfl.MaxPooling1D(pool_size=2))
    model.add(tfl.Dropout(rate=0.3))
    model.add(tfl.LSTM(input_shape, return_sequences=True))
    model.add(tfl.Flatten())
    model.add(
        tfl.Dense(
            units=128,
            activation="relu",
        )
    )
    model.add(
        tfl.Dense(
            units=64,
            activation="relu",
        )
    )
    # Softmax
    model.add(tfl.Dense(units=softmax_units, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=metrics,
    )

    return model
