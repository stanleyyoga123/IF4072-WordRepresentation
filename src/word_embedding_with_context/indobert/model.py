from transformers import BertForSequenceClassification, BertConfig

def indobert(name='indobenchmark/indobert-base-p1', num_labels=2):
    config = BertConfig.from_pretrained(name)
    config.num_labels = num_labels
    model = BertForSequenceClassification.from_pretrained(name, config=config)
    return model