import os
import pandas as pd

from tensorflow.keras.optimizers import Adam

from src.word_embedding_with_context import train_bert


def main_bert():
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
    train_bert(x_train, y_train, x_test, y_test, "binary_crossentropy", optimizer, metrics)


if __name__ == "__main__":
    main_bert()
