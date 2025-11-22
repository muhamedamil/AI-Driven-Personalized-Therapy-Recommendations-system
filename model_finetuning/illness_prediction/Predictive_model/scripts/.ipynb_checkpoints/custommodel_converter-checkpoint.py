import torch
from transformers import AutoTokenizer, AutoModel
import json
import os
MODEL_PATHS = {
    "mental_health":r"C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_3\Predictive_model\model\best_model\model.pth",
    "mental_health_tokenizer": r"C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_3\Predictive_model\model\best_model"
}

class CustomDebertaClassifier(torch.nn.Module):
    def __init__(self, deberta_model, num_labels):
        super(CustomDebertaClassifier, self).__init__()
        self.deberta = deberta_model  
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_labels)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return {"loss": loss, "logits": logits}

output_dir = r"C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_3\ouptput"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHECKPOINT = "microsoft/deberta-v3-base"
NUM_LABELS = 19

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS['mental_health_tokenizer'])
deberta_model = AutoModel.from_pretrained(MODEL_CHECKPOINT)

model = CustomDebertaClassifier(deberta_model, NUM_LABELS)
model.to(device)

checkpoint_path = r"C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_3\Predictive_model\model\best_model\model.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")

config = {
    "model_type": "CustomDebertaClassifier",
    "num_labels": NUM_LABELS,
    "base_model": MODEL_CHECKPOINT
}
with open(f"{output_dir}/config.json", "w") as f:
    json.dump(config, f)

# ✅ Save the tokenizer
tokenizer.save_pretrained(output_dir)

print(f"✅ Model saved successfully in {output_dir}")
