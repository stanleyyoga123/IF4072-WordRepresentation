# Tensorflow
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# === Model Creation ===
METRICS = [
    f1,
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]


# Architecture : Input -> Embedding -> LSTM -> Dropout (to prevent overfit) -> Flatten -> Dropout (again...) -> Dense (output layer)
def create_classifier(
    vocabulary_size=1000, max_length=128, embedding_configs=None, metrics=METRICS
):
    """
    Model Architecture :
          Input
            |
        Embedding (Word2Vec, no context)
            |
          LSTM
            |
         Dropout
            |
           Out
    """
    model = Sequential()
    if embedding_configs == None:
        model.add(
            Embedding(
                vocabulary_size, output_dim=256, input_length=max_length, trainable=True
            )
        )
    else:
        embedding_dim = embedding_configs.get("embedding_dim")
        embedding_matrix = embedding_configs.get("embedding_matrix")
        model.add(
            Embedding(
                vocabulary_size,
                output_dim=100,
                weights=[embedding_matrix],
                trainable=False,
            )
        )
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))

    return model
