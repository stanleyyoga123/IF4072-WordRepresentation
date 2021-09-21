import os
import pandas as pd

from tqdm import tqdm 

from gensim.utils import simple_preprocess

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.metrics import Accuracy

from src.word_embedding import train_word2vec, f1


tqdm.pandas()

def main_w2v():

    METRICS = [Accuracy(name="accuracy")]

    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train['text_a'] = train.text_a.progress_apply(simple_preprocess)
    test['text_a'] = test.text_a.progress_apply(simple_preprocess)

    x_train = train["text_a"].values[:200]
    y_train = train["label"][:200]
    x_test = test["text_a"].values[:200]
    y_test = test["label"][:200]

    maps = {"no": 0, "yes": 1}
    y_train = y_train.replace(maps).values
    y_test = y_test.replace(maps).values

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    optimizer = Adam(learning_rate=1e-3)

    train_word2vec(
        x_train,
        y_train,
        x_test,
        y_test,
        BinaryCrossentropy(),
        optimizer,
        METRICS,
        batch_size=64,
        max_length=128,
        detail="exp-2"
    )


if __name__ == "__main__":
    main_w2v()
