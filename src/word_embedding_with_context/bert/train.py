import os
import datetime
import pickle
import timeit

import numpy as np
from sklearn.metrics import accuracy_score

from src.word_embedding_with_context.bert import TokenizerBert, bert


def train_bert(
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
    save_pred=True,
    model_path=os.path.join("bin", "bert"),
    log=True,
):
    if log:
        print("Tokenizing")

    start = timeit.default_timer()
    tokenizer = TokenizerBert(max_length=max_length)
    x = tokenizer.tokenize(x_train)
    y = y_train

    if log:
        print(f"Time Taken: {timeit.default_timer() - start:.4f}")

    print("Build Model")
    start = timeit.default_timer()
    model = bert(length=len(x["input_ids"][0]))

    if log:
        print(f"Time Taken: {timeit.default_timer() - start:.4f}")

    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    summary = "\n".join(summary)
    print(summary)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    history = model.fit(
        x=x,
        y=y,
        validation_split=validation_split,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )

    if log:
        print("Tokenizing Test")

    start = timeit.default_timer()
    x_test = tokenizer.tokenize(x_test)

    if log:
        print(f"Time Taken: {timeit.default_timer() - start:.4f}")

    y_pred_proba = model.predict(x_test, batch_size=batch_size, verbose=verbose)
    y_pred = np.round(y_pred_proba)
    # Currently evaluation use Accuracy
    score = accuracy_score(y_test, y_pred)
    print(f"Validation Score: {score}")

    if save_model or save_config or save_pred:
        model_folder = os.path.join(model_path, datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S"))
        model_name = "bert.h5"
        config_name = "config.txt"
        pred_name = "y_pred.pkl"

        os.mkdir(model_folder)
        if save_model:
            model.save_weights(os.path.join(model_folder, model_name))

        if save_config:
            f = open(os.path.join(model_folder, config_name), "w+")
            train_config = ""
            train_config += f"summary mode: \n{summary}\n"
            train_config += f"max_length: {max_length}\n"
            train_config += f"loss: {loss}\n"
            train_config += f"optimizer: {optimizer.get_config()}\n"
            train_config += f"batch_size: {batch_size}\n"
            train_config += f"epochs: {epochs}\n"
            train_config += f"metrics: {metrics}\n"
            train_config += f"validation_split: {validation_split}\n"
            train_config += f"history\n{history.history}\n"
            train_config += f"evaluation: {score}\n"
            f.write(train_config)

        if save_pred:
            pickle.dump(y_pred, open(os.path.join(model_folder, pred_name), "wb"))
