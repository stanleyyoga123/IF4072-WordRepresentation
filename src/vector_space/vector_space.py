from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def vectorize_data(
    x_train,
    x_test,
    vectorizer='tfidf',
    max_features=5000,
    ngram_range=(1,1)
):
    if vectorizer == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
    x_train_vectorized = vectorizer.fit_transform(x_train).toarray()
    x_test_vectorized = vectorizer.transform(x_test).toarray()
    return x_train_vectorized, x_test_vectorized
