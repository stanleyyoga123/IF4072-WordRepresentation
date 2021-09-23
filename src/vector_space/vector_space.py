import numpy as np
import pandas as pd
from time import time

from lightgbm import LGBMClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def vectorize_data(
    x_train,
    x_test,
    max_features=2000,
    ngram_range=(1,1)
):
    vectorizer = TfidfVectorizer(max_features=2000)
    x_train_vectorized = vectorizer.fit_transform(x_train).toarray()
    x_test_vectorized = vectorizer.transform(x_test).toarray()
    return x_train_vectorized, x_test_vectorized
