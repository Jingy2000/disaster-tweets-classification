import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import numpy as np
import pandas as pd

from src.dataset import DisasterTweetDataset
from src.config import *


def compute_metrics(true_labels, predicted_labels, predicted_probs):
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_probs)
    return precision, recall, f1, auc


def train(model, dataframe, device, epochs=EPOCH, batch_size=BATCH_SIZE, learning_rate=LR):
    # Load model and move it to device
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    skf = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED, shuffle=True)
    folds = list(skf.split(dataframe, dataframe['keyword']))

    for epoch in range(epochs):
        for train_idx, val_idx in folds:
            train_df = dataframe.iloc[train_idx]
            val_df = dataframe.iloc[val_idx]

            train_dataset = DisasterTweetDataset(train_df)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            val_dataset = DisasterTweetDataset(val_df)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            model.train()
            total_loss = 0
            train_true_labels = []
            train_predictions = []
            train_probs = []

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
            for batch in pbar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                labels = batch['label'].to(device, dtype=torch.float)

                logits = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                train_true_labels.extend(labels.cpu().numpy())
                train_predictions.extend(torch.round(logits).cpu().detach().numpy())
                train_probs.extend(logits.cpu().detach().numpy())

                loss = criterion(logits.squeeze(), labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{total_loss / (pbar.n + 1):.4f}'})
            precision, recall, f1, auc = compute_metrics(train_true_labels, train_predictions, train_probs)

            print(f"Epoch {epoch + 1} - Training Loss: {total_loss / len(train_loader)}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

            # validation loop to evaluate model on validation data after each epoch
            model.eval()
            total_val_loss = 0
            val_true_labels = []
            val_predictions = []
            val_probs = []
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation"):
                with torch.no_grad():
                    # Unpack the batch
                    input_ids = batch['input_ids'].to(device, dtype=torch.long)
                    attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
                    token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
                    labels = batch['label'].to(device, dtype=torch.float)

                    # Forward pass
                    logits = model(input_ids, attention_mask, token_type_ids)
                    val_loss = criterion(logits.squeeze(), labels)

                    total_val_loss += val_loss.item()

                    val_true_labels.extend(labels.cpu().numpy())
                    val_predictions.extend(torch.round(logits).cpu().detach().numpy())
                    val_probs.extend(logits.cpu().detach().numpy())

            print(f"Validation loss after epoch {epoch + 1}: {total_val_loss / len(val_loader):.4f}")

            precision, recall, f1, auc = compute_metrics(val_true_labels, val_predictions, val_probs)
            print(
                f"Epoch {epoch + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    print("Training complete.")

    return model


def prediction(model, df_test, device):
    test_dataset = DisasterTweetDataset(df_test)
    data_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=32
    )
    model.eval()
    val_predictions = []
    val_probs = []
    for batch in tqdm(data_loader, desc="Test"):
        with torch.no_grad():
            # Unpack the batch
            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)

            # model prediction
            logits = model(input_ids, attention_mask, token_type_ids)
            val_predictions.extend(torch.round(logits).cpu().detach().numpy())
            val_probs.extend(logits.cpu().detach().numpy())

    model_submission = pd.read_csv("./data/sample_submission.csv")
    model_submission['target'] = np.array(val_predictions).astype('int')
    model_submission.to_csv('./results/model_submission.csv', index=False)


def evaluation(model, df_test, device):
    test_dataset = DisasterTweetDataset(df_test)
    data_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=16
    )
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_val_loss = 0
    val_true_labels = []
    val_predictions = []
    val_probs = []
    for batch in tqdm(data_loader, desc="Test"):
        with torch.no_grad():
            # Unpack the batch
            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            labels = batch['label'].to(device, dtype=torch.float)

            # model prediction
            logits = model(input_ids, attention_mask, token_type_ids)
            val_true_labels.extend(labels.cpu().numpy())
            val_predictions.extend(torch.round(logits).cpu().detach().numpy())
            val_probs.extend(logits.cpu().detach().numpy())

            # loss
            val_loss = criterion(logits.squeeze(), labels)
            total_val_loss += val_loss.item()

    print(f"Test loss: {total_val_loss / len(data_loader):.4f}")

    precision, recall, f1, auc = compute_metrics(val_true_labels, val_predictions, val_probs)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

    model_submission = pd.read_csv("../data/sample_submission.csv")
    model_submission['target'] = np.array(val_predictions).astype('int')
    model_submission.to_csv('../results/model_submission.csv', index=False)
