from src.model import BertClassifier
from src.dataset import DisasterTweetDataset
from src.config import *
from src.train import train, prediction
import torch
import pandas as pd
import numpy as np

df_train = pd.read_csv('./data/train.csv', dtype={'id': np.int16, 'target': np.int8})
df_test = pd.read_csv('./data/test.csv', dtype={'id': np.int16})

for df in [df_train, df_test]:
    df['keyword'] = df['keyword'].fillna('no_keyword')
    df['location'] = df['location'].fillna('no_location')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertClassifier(freeze_bert=False)
trained_model = train(model, df_train, device)

prediction(trained_model, df_test, device)
