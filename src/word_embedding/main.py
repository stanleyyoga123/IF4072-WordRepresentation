import os
import pandas as pd

from tqdm import tqdm

from gensim.utils import simple_preprocess

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.metrics import Accuracy

from src.word_embedding.train import pipeline
from src.word_embedding.word2vec import build_word2vec, build_fasttext

tqdm.pandas()


def main(
    config,
    types,
    metrics=["accuracy"],
    learning_rate=1e-3,
    batch_size=64,
    epochs=5,
    max_length=128,
    detail="",
):

    METRICS = ["accuracy"]

    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train["text_a"] = train.text_a.progress_apply(simple_preprocess)
    test["text_a"] = test.text_a.progress_apply(simple_preprocess)

    x_train = train["text_a"].values[:200]
    y_train = train["label"][:200]
    x_test = test["text_a"].values
    y_test = test["label"]

    maps = {"no": 0, "yes": 1}
    y_train = y_train.replace(maps).values
    y_test = y_test.replace(maps).values

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    optimizer = Adam(learning_rate=learning_rate)

    pipeline(
        x_train,
        y_train,
        x_test,
        y_test,
        types,
        config,
        BinaryCrossentropy(),
        optimizer,
        metrics,
        batch_size=batch_size,
        epochs=epochs,
        max_length=max_length,
        detail=detail,
    )
