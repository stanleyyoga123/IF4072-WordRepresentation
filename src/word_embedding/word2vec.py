from gensim.models import Word2Vec, FastText

configs = {
    "window": 5,
    "min_count": 1,
    "workers": -1,
}


def build_word2vec(texts, configs=configs, log=True):
    w2v_model = Word2Vec(sentences=texts, **configs)
    if log:
        print_vocab_details(w2v_model.wv)
    return w2v_model


def build_fasttext(texts, configs=configs, log=True):
    fasttext_model = FastText(sentences=texts, **configs)
    if log:
        print_vocab_details(fasttext_model.wv)
    return fasttext_model


def print_vocab_details(vocab):
    print("Number of word vectors: {}".format(len(vocab)))
