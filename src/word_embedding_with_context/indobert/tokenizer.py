from transformers import BertTokenizer


def tokenizer_indobert(name='indobenchmark/indobert-base-p1'):
    tokenizer = BertTokenizer.from_pretrained(name)
    return tokenizer