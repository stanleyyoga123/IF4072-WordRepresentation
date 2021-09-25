import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DocumentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'no': 0, 'yes': 1}
    INDEX2LABEL = {0: 'no', 1: 'yes'}
    NUM_LABELS = 2
    
    def load_dataset(self, path): 
        df = pd.read_csv(path)
        if len(df.columns) == 3:
            df.drop(columns='Unnamed: 0', inplace=True)
        df.columns = ['text','label']
        df['label'] = df['label'].replace(self.LABEL2INDEX)
        return df
    
    def __init__(self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, sentiment = data['text'], data['label']
        subwords = self.tokenizer.encode(text, add_special_tokens=not self.no_special_token)
        return np.array(subwords), np.array(sentiment), data['text']
    
    def __len__(self):
        return len(self.data)   