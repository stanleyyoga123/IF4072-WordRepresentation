import os
import datetime
import pickle
import timeit

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from src.word_embedding.tokenizer import W2VTokenizer
from src.word_embedding.classifier import create_classifier
from src.word_embedding.word2vec import build_word2vec, build_fasttext


def pipeline(
    x_train,
    y_train,
    x_test,
    y_test,
    types,
    config,
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
    save_pred=True,
    model_path=os.path.join("bin", "w2v"),
    log=True,
    detail="default-exp",
):
    if log:
        print("Tokenizing Training data...")

    start = timeit.default_timer()
    tokenizer = W2VTokenizer(max_length=max_length)
    tokenizer.fit(x_train)
    x = tokenizer.tokenize(x_train)
    y = y_train

    if log:
        print(f"Time Taken: {timeit.default_timer() - start:.4f}")

    start = timeit.default_timer()

    print(x[:100])
    if types == "ft":
        w2v = build_fasttext(x, config, log=True)
    else:
        w2v = build_word2vec(x, config, log=True)

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
        monitor="val_accuracy",
        verbose=verbose,
        patience=10,
        mode="max",
        restore_best_weights=True,
    )

    # === Defining model checkpoint ===
    checkpoint_path = os.path.join(model_path, "w2v-clf-checkpoint.ckpt")
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
        ],
    )

    # == Predict ==
    if log:
        print("Tokenizing test data...")
    start = timeit.default_timer()
    tokenizer.fit(x_train)
    x_test_tokenized = tokenizer.tokenize(x_test)

    if log:
        print(f"Time Taken: {timeit.default_timer() - start:.4f}")

    # loss, f1_score, precision_score, recall_score = model.evaluate(x_test_tokenized, y_test, verbose=2)
    # print("Metric score (F1): {:5.2f}%".format(100 * f1_score))
    y_pred_proba = model.predict(
        x_test_tokenized, batch_size=batch_size, verbose=verbose
    )
    y_pred = np.round(y_pred_proba)
    score = accuracy_score(y_test, y_pred)
    print(f"Validation Score (Acc): {score}")

    # == Save Model ==
    if save_model or save_config or save_pred:
        model_folder = os.path.join(
            model_path,
            datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S") + "-" + detail,
        )
        clf_name = f"clf-{detail}-weight"
        model_name = f"w2v-{detail}.model"
        config_name = "configs.txt"
        pred_name = "y_pred.pkl"

        if save_model:
            # Save classifier
            if log:
                print("Saving classifier weight...")
            weight_path = os.path.join(model_folder, clf_name)
            model.save_weights(weight_path)

            # Save word2vec
            if log:
                print("Saving word2vec model...")
            w2v_path = os.path.join(model_folder, model_name)
            w2v.save(w2v_path)

        if save_config:
            configs = {
                "max_length": max_length,
                "loss": loss.get_config(),
                "optimizer": optimizer.get_config(),
                "batch_size": batch_size,
                "epochs": epochs,
                "metrics": "acc",
                "validation_split": validation_split,
                "history": history.history,
                "evaluation": score,
                "detail": detail,
                "type": "skip-gram" if config.get("sg") == 1 else "cbow",
                "window": config.get("window"),
            }
            write_configs(os.path.join(model_folder, config_name), **configs)

        if save_pred:
            # Saving result
            pickle.dump(y_pred, open(os.path.join(model_folder, pred_name), "wb"))

            if log:
                print("Saving Error Snapshot...")

            test_result_df = pd.DataFrame(
                {
                    "text": x_test,
                    "label": np.argmax(y_test, axis=1),
                    "pred": np.argmax(y_pred, axis=1),
                }
            )
            # Type II error
            FP = test_result_df.loc[
                (test_result_df["label"] == 0) & (test_result_df["pred"] == 1)
            ]

            # Type I error
            FN = test_result_df.loc[
                (test_result_df["label"] == 1) & (test_result_df["pred"] == 0)
            ]

            if log:
                print(
                    f"False Positive (Type II Error) : {len(FP)} / {len(test_result_df)}"
                )
            FP.to_csv(os.path.join(model_folder, "FP.csv"), index=False)
            if log:
                print(
                    f"False Negative (Type I Error) : {len(FN)} / {len(test_result_df)}"
                )
            FN.to_csv(os.path.join(model_folder, "FN.csv"), index=False)


def write_configs(path, **kwargs):
    configs = kwargs.get("detail") + "\n"
    for key, val in kwargs.items():
        if key != "detail":
            configs += f"{key} : {val}\n"
    with open(path, "w") as f:
        f.write(configs)
