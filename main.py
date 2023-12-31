from src.model import BertClassifier
from src.dataset import DisasterTweetDataset
from src.config import *
from src.train import train, prediction, train_validation
import torch
import pandas as pd
import numpy as np

df_train = pd.read_csv('./data/train_cleaned.csv', dtype={'id': np.int16, 'target': np.int8})
df_test = pd.read_csv('./data/test_cleaned.csv', dtype={'id': np.int16})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertClassifier(freeze_bert=False)
validation_models = train_validation(BertClassifier, df_train, device)
# trained_model = train(model, df_train, device)

# prediction(trained_model, df_test, device)
