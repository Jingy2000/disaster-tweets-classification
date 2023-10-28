import torch
import torch.nn as nn
from transformers import BertModel

from src.config import MODEL_CHECKPOINT


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.1, checkpoint=MODEL_CHECKPOINT, freeze_bert=False):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(checkpoint)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs[0][:, 0, :]
        logits = torch.sigmoid(self.linear(cls_output))
        return logits
