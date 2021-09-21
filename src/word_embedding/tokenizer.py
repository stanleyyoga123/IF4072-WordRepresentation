import numpy as np

# == Tokenize Input ==
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Global variable
vocabulary_size = 1000
oov_tok = "<OOV>"
max_length = 128

configs = {
    "lower": True,
    "split": " ",
    "num_words": vocabulary_size,
    "filters": '•ˆ²≥≤⅔¢×ĺ±…–ƪʃںᵤₒᵗʰᵃᶰᵏᵧₒᵤ!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    "char_level": False,
    "oov_token": oov_tok,
}


class W2VTokenizer:
    def __init__(self, configs=configs, max_length=max_length):
        self.configs = configs
        self.tokenizer = Tokenizer(**self.configs)
        self.max_length = configs.get("max_length")
        self.fitted = False

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)
        self.fitted = True

    def tokenize(self, texts):
        x = self.tokenizer.texts_to_sequences(texts)
        x = pad_sequences(x, maxlen=self.max_length, padding="pre", truncating="post")
        return x

    def get_embedding_matrix(self, model, embedding_dim):
        if not self.fitted:
            print("Model not fitted yet!")
        else:
            word_index = self.tokenizer.word_index
            num_words = self.configs.get("num_words")
            vocabulary_size = min(len(word_index) + 1, num_words)
            embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
            for word, i in word_index.items():
                if i >= num_words:
                    continue
                try:
                    embedding_vector = model.wv[word]
                    embedding_matrix[i] = embedding_vector
                except KeyError:
                    embedding_matrix[i] = np.random.normal(
                        0, np.sqrt(0.25), embedding_dim
                    )
            return embedding_matrix
