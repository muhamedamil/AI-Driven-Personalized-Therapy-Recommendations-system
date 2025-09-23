#!/usr/bin/env python
# coding: utf-8

# In[79]:


import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer,AutoModel


# In[81]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[86]:



# In[84]:


MODEL_PATHS = {
    "sentiment": r"C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_2\sentiment_analysis\Model\fine_tuned_model",
    "mental_health": r"C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_3\Predictive_model\model\best_model\model.pth",
    "mental_health_tokenizer": r"C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_3\Predictive_model\model\best_model"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sentiment_model():
    """
    Loads the fine-tuned sentiment analysis Transformer model and tokenizer.
    """
    model_path = MODEL_PATHS["sentiment"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model

class CustomDebertaClassifier(nn.Module):
    def __init__(self, deberta_model, num_labels):
        super(CustomDebertaClassifier, self).__init__()
        self.deberta = deberta_model  
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}

def load_predective_model():
    """
    Loads the fine-tuned DeBERTa model for mental illness prediction.
    """
    checkpoint_path = MODEL_PATHS["mental_health"]
    tokenizer_path = MODEL_PATHS["mental_health_tokenizer"]

    MODEL_CHECKPOINT = "microsoft/deberta-v3-base"
    NUM_LABELS = 19  # Your model's number of output classes
    predictive_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    deberta_model = AutoModel.from_pretrained(MODEL_CHECKPOINT)

    predictive_model = CustomDebertaClassifier(deberta_model, NUM_LABELS)
    predictive_model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        predictive_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        predictive_model.load_state_dict(checkpoint)

    # Load optimizer (optional)
    optimizer = AdamW(predictive_model.parameters(), lr=2e-5, weight_decay=0.01)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    predictive_model.eval()
    return predictive_model, predictive_tokenizer


sentiment_tokenizer, sentiment_model = load_sentiment_model()
predective_model, predective_model_tokenizer = load_predective_model()
print("✅ All models loaded successfully!")


# In[ ]:





# In[ ]:




