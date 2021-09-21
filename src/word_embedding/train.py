import os
import datetime
import pickle
import timeit

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.word_embedding.tokenizer import W2VTokenizer
from src.word_embedding.word2vec import build_word2vec
from src.word_embedding.classifier import create_classifier


def train_word2vec(
    x_train,
    y_train,
    x_test,
    y_test,
    loss,
    optimizer,
    metrics,
    validation_split=0.2,
    batch_size=4,
    epochs=5,
    verbose=1,
    max_length=512,
    save_config=True,
    save_model=True,
    model_path=os.path.join("bin", "w2v"),
    log=True,
):
    if log:
        print("Tokenizing...")

    start = timeit.default_timer()
    tokenizer = W2VTokenizer(max_length=max_length)
    tokenizer.fit(x_train)
    x = tokenizer.tokenize(x_train)
    y = y_train

    if log:
        print(f"Time Taken: {timeit.default_timer() - start:.4f}")

    print("Build Word2Vec")
    start = timeit.default_timer()
    w2v = build_word2vec(x, log=False)

    embedding_matrix = tokenizer.get_embedding_matrix(w2v, 100)

    if log:
        print(f"Time Taken: {timeit.default_timer() - start:.4f}")

    model = create_classifier(
        embedding_configs={"embedding_dim": 100, "embedding_matrix": embedding_matrix}
    )
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    print(model.summary())

    # === Defining Early Stopping ===
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1", verbose=verbose, patience=10, mode="max", restore_best_weights=True
    )

    # === Defining model checkpoint ===
    checkpoint_path = os.path.join("bin", "w2v", "lstm-w2v.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, verbose=verbose
    )

    history = model.fit(
        x,
        y,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=[
            early_stopping,
            cp_callback,
        ],  # Callbacks : Early stopping | Checkpoint Model
    )

    # == Predict ==
    # == Save Model ==
    # == Save config ==
    # == Save Result ==

