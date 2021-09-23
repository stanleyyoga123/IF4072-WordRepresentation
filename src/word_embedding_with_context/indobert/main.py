import os
from src.word_embedding_with_context.indobert import train_indobert


def main_indobert():
    train_path = os.path.join('data', 'train.csv')
    dev_path = os.path.join('data', 'dev.csv')
    test_path = os.path.join('data', 'test.csv')

    train_indobert(train_path, dev_path, test_path, batch_size=2)
