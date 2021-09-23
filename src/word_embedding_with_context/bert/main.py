import os
import pandas as pd
from tqdm import tqdm 

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from gensim.utils import simple_preprocess

from src.word_embedding_with_context import train_bert

tqdm.pandas()

def main_bert():
    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train['text_a'] = df_train.text_a.progress_apply(simple_preprocess)
    df_test['text_a'] = df_test.text_a.progress_apply(simple_preprocess)
    df_train['text_a'] = df_train['text_a'].progress_apply(lambda x: ' '.join(x))
    df_test['text_a'] = df_test['text_a'].progress_apply(lambda x: ' '.join(x))

    x_train = df_train["text_a"]
    y_train = df_train["label"]
    x_test = df_test["text_a"]
    y_test = df_test["label"]

    maps = {"no": 0, "yes": 1}
    y_train = y_train.replace(maps)
    y_test = y_test.replace(maps)

    optimizer = Adam(learning_rate=3e-6)
    metrics = ["accuracy"]
    train_bert(
        x_train,
        y_train,
        x_test,
        y_test,
        "binary_crossentropy",
        optimizer,
        metrics,
        batch_size=4,
        max_length=512,
    )