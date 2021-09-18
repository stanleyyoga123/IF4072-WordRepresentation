from transformers import BertConfig, BertTokenizerFast


class Tokenizer:
    def __init__(self, model_name="bert-base-uncased", max_length=512):
        self.config = BertConfig.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=model_name, config=self.config
        )
        self.max_length = max_length

    def tokenize(self, texts):
        x = self.tokenizer(
            text=list(texts),
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="tf",
        )
        return {
            'input_ids': x['input_ids'],
            'attention_mask': x['attention_mask'],
            'token_type_ids': x['token_type_ids']
        }
