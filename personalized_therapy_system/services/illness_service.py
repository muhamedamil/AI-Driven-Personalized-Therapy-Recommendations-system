import os 
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel




class CustomDebertaClassifier(nn.Module):
    """Custom classifier using DeBERTa for mental illness detection."""
    
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

        loss = self.criterion(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

class IllnessService:
    """Service for detecting mental health conditions using a fine-tuned DeBERTa model."""
    
    def __init__(self):
        """Initialize the illness detection model."""
        
        self.model_dir = os.getenv("MODEL_DIR", "/app/ml_model/illness_predection")
        self.device = torch.device("cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # Load base DeBERTa model
        model_checkpoint = "microsoft/deberta-v3-base"
        self.deberta_model = AutoModel.from_pretrained(model_checkpoint)

        # Load fine-tuned classifier model
        self.model = CustomDebertaClassifier(self.deberta_model, num_labels=19)
        model_bin_path = f"{self.model_dir}/pytorch_model.bin"
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_bin_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # Mapping for illness labels
        self.mental_health_mapping = {
            0: "ADHD", 1: "Anxiety", 2: "BDD", 3: "Bipolar", 4: "BPD",
            5: "Depression", 6: "Eating Disorder", 7: "Hoarding Disorder",
            8: "Mental Illness", 9: "Normal", 10: "OCD", 11: "Off My Chest",
            12: "Panic Disorder", 13: "Personality Disorder", 14: "PTSD",
            15: "Schizophrenia", 16: "Social Anxiety", 17: "Stress", 18: "Suicidal"
        }

    def predict_illness(self, text):
        """
        Predict the mental health condition for the given text.
        Args:
            text (str): User input text.
        Returns:
            dict: Prediction with label and confidence score.
        """
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            logits = outputs["logits"]

        # Apply softmax to get confidence scores
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

        # Get illness label
        label = self.mental_health_mapping.get(pred, "Unknown")

        return label


