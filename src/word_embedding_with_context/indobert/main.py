import os
from src.word_embedding_with_context.indobert import train_indobert


def main_indobert(epochs=5, batch_size=4, learning_rate=3e-6, max_seq_len=512):
    train_path = os.path.join("data", "train.csv")
    dev_path = os.path.join("data", "dev.csv")
    test_path = os.path.join("data", "test.csv")

    train_indobert(
        train_path,
        dev_path,
        test_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
    )
