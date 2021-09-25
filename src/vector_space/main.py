import os
import pandas as pd

from src.vector_space import train_vector_space

def main_vector_space(vectorizer='tfidf', model='lgbm'):
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")
    # train_path = os.path.join("data", "cleaned_train.csv")
    # test_path = os.path.join("data", "cleaned_test.csv")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    x_train = df_train["text_a"]
    y_train = df_train["label"]
    x_test = df_test["text_a"]
    y_test = df_test["label"]

    maps = {"no": 0, "yes": 1}
    y_train = y_train.replace(maps)
    y_test = y_test.replace(maps)

    train_vector_space(
      x_train, y_train, x_test, y_test,
      vectorizer=vectorizer, model=model, verbose=0
    )