import os
import pandas as pd

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src.vector_space import train_vector_space
from src.word_embedding_with_context import train_bert


def main_bert():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    x_train = df_train["text_a"]
    y_train = df_train["label"]
    x_test = df_test["text_a"]
    y_test = df_test["label"]

    maps = {"no": 0, "yes": 1}
    y_train = y_train.replace(maps)
    y_test = y_test.replace(maps)

    optimizer = Adam(learning_rate=1e-4)
    metrics = ["accuracy"]
    train_bert(x_train, y_train, x_test, y_test, "binary_crossentropy", optimizer, metrics, batch_size=1, max_length=128)


def main_vector_space():
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    x_train = df_train["text_a"]
    y_train = df_train["label"]
    x_test = df_test["text_a"]
    y_test = df_test["label"]

    maps = {"no": 0, "yes": 1}
    y_train = y_train.replace(maps)
    y_test = y_test.replace(maps)

    train_vector_space(x_train, y_train, x_test, y_test, verbose=0)


if __name__ == "__main__":
    # main_bert()
    main_vector_space()
