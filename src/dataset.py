import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from src.config import *


class DisasterTweetDataset(Dataset):
    def __init__(self, df, checkpoint=MODEL_CHECKPOINT, max_length=512):
        self.df = df
        if 'target_relabeled' in df:
            self.labels = df['target_relabeled'].tolist()
        elif 'target' in df:
            self.labels = df['target'].tolist()
        else:
            self.labels = None
        # self.labels = df['target_relabeled'].tolist() if 'target_relabeled' in df else df['target'].tolist()

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.max_length = max_length
        self.tokenized_data = self.tokenizer.batch_encode_plus(
            df['text'].tolist(),
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = {
            'input_ids': self.tokenized_data['input_ids'][idx],
            'attention_mask': self.tokenized_data['attention_mask'][idx],
            'token_type_ids': self.tokenized_data['token_type_ids'][idx]
        }
        if self.labels:
            item['label'] = torch.tensor(self.labels[idx])
        return item


if __name__ == "__main__":
    print(EPOCH)
