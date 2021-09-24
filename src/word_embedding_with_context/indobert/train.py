import os
import pandas as pd
import datetime

from tqdm import tqdm
import torch
from torch import optim

from src.word_embedding_with_context.indobert import *


def train_indobert(
    train_path,
    dev_path,
    test_path,
    epochs=5,
    max_seq_len=512,
    learning_rate=3e-6,
    batch_size=4,
    folder=os.path.join("bin", "indobert"),
):
    set_seed(42)
    tokenizer = tokenizer_indobert()
    model = indobert()

    train_dataset = DocumentDataset(train_path, tokenizer, lowercase=True)
    valid_dataset = DocumentDataset(dev_path, tokenizer, lowercase=True)
    test_dataset = DocumentDataset(test_path, tokenizer, lowercase=True)

    train_loader = DocumentDataLoader(
        dataset=train_dataset,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
    )
    valid_loader = DocumentDataLoader(
        dataset=valid_dataset,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
    )
    test_loader = DocumentDataLoader(
        dataset=test_dataset,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
    )

    w2i, i2w = DocumentDataset.LABEL2INDEX, DocumentDataset.INDEX2LABEL

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.cuda()

    # Train
    n_epochs = epochs
    for epoch in range(n_epochs):
        model.train()
        torch.set_grad_enabled(True)

        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(train_loader, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # Forward model
            loss, batch_hyp, batch_label = forward_sequence_classification(
                model, batch_data[:-1], i2w=i2w, device="cuda"
            )

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label

            train_pbar.set_description(
                "(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format(
                    (epoch + 1), total_train_loss / (i + 1), get_lr(optimizer)
                )
            )

        # Calculate train metric
        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
        print(
            "(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format(
                (epoch + 1),
                total_train_loss / (i + 1),
                metrics_to_string(metrics),
                get_lr(optimizer),
            )
        )

        # Evaluate on validation
        model.eval()
        torch.set_grad_enabled(False)

        total_loss, total_correct, total_labels = 0, 0, 0
        list_hyp, list_label = [], []

        pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]
            loss, batch_hyp, batch_label = forward_sequence_classification(
                model, batch_data[:-1], i2w=i2w, device="cuda"
            )

            # Calculate total loss
            valid_loss = loss.item()
            total_loss = total_loss + valid_loss

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label
            metrics = document_sentiment_metrics_fn(list_hyp, list_label)

            pbar.set_description(
                "VALID LOSS:{:.4f} {}".format(
                    total_loss / (i + 1), metrics_to_string(metrics)
                )
            )

        metrics = document_sentiment_metrics_fn(list_hyp, list_label)
        print(
            "(Epoch {}) VALID LOSS:{:.4f} {}".format(
                (epoch + 1), total_loss / (i + 1), metrics_to_string(metrics)
            )
        )

    # Evaluate on test
    model.eval()
    torch.set_grad_enabled(False)

    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    pbar = tqdm(test_loader, leave=True, total=len(test_loader))
    for i, batch_data in enumerate(pbar):
        _, batch_hyp, _ = forward_sequence_classification(
            model, batch_data[:-1], i2w=i2w, device="cuda"
        )
        list_hyp += batch_hyp

    # Save prediction
    df = pd.DataFrame({"label": list_hyp}).reset_index()

    model_folder = os.path.join(folder, datetime.datetime.now().strftime("%Y-%m-%d %H.%M.%S"))
    os.mkdir(model_folder)

    df_path = os.path.join(model_folder, 'prediction.csv')
    model_path = os.path.join(model_folder, 'indobert.pth')
    config_path = os.path.join(model_folder, 'config.txt')

    df.to_csv(df_path, index=False)

    train_config = ''
    train_config += f'epochs: {epochs}\n'
    train_config += f'max_length: {max_seq_len}\n'
    train_config += f'learning_rate: {learning_rate}\n'
    train_config += f'batch_size: {batch_size}\n'
    f = open(config_path, 'w+')
    f.write(train_config)

    # torch.save(model_path)