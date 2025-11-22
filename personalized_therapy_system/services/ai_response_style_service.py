import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ResponseStyleService:
    def __init__(self, model_name="muhamedamil/AI_response_style_model"):
        """Initialize response style classification model."""
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        self.ai_response_mapping = {
            0: "De-escalation & Validation",
            1: "Reframing & Encouragement",
            2: "Reassurance & Coping Strategies",
            3: "Encouragement & Positive Reinforcement",
            4: "Active Listening & Encouragement",
            5: "Compassion & Support",
            6: "Clarification & Stability"
        }
    def classify_response_style(self, text: str) -> dict:
        """Classifies response style based on user input."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)  
        prediction = torch.argmax(probs, dim=-1).item()
        response_style = self.ai_response_mapping.get(prediction, "Unknown Style")
        return response_style
