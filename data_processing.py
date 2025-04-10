import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import random


# 数据增强：随机替换词语
def random_replace(text, tokenizer, p=0.1):
    tokens = tokenizer.tokenize(text)
    new_tokens = []
    for token in tokens:
        if random.random() < p:
            new_token = random.choice(list(tokenizer.vocab.keys()))
            new_tokens.append(new_token)
        else:
            new_tokens.append(token)
    new_text = tokenizer.convert_tokens_to_string(new_tokens)
    return new_text


# 定义数据集类
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, augment=False):
        self.texts = texts.reset_index(drop=True)  # 重置索引
        self.labels = labels.reset_index(drop=True)  # 重置索引
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.augment:
            text = random_replace(text, self.tokenizer)
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 加载数据
def load_data():
    df = pd.read_csv('df_file.csv')
    labels = df['Label']
    label_set = set(labels)
    data = df['Text']
    return data, labels, len(label_set)