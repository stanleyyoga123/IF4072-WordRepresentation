from transformers import TFAutoModel

from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model


def bert(model_name="bert-base-uncased", length=512):
    bert = TFAutoModel.from_pretrained(model_name)

    bert_layer = bert.bert
    input_ids = Input(shape=(length,), name="input_ids", dtype="int32")
    token_type_ids = Input(shape=(length,), name="token_type_ids", dtype="int32")
    attention_mask = Input(shape=(length,), name="attention_mask", dtype="int32")
    inputs = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
    }

    x = bert_layer(inputs)[0]
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    return model
