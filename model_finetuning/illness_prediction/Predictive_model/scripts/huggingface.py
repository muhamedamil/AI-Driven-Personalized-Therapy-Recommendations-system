from huggingface_hub import HfApi, HfFolder
from transformers import AutoTokenizer, AutoModel
import torch
import os

api = HfApi()
token = HfFolder.get_token()
repo_id = 'muhamedamil/mental_illness_predection_model'


model_path = r"C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_3\Predictive_model\model\converted_model\pytorch_model.bin"
folder_path = r"C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_3\Predictive_model\model\converted_model"

# Push the model file
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="pytorch_model.bin",
    repo_id=repo_id,
    token=token
)

# Push the tokenizer folder
api.upload_folder(
    folder_path=folder_path,
    path_in_repo=".",
    repo_id=repo_id,
    token=token
)

print("✅ Model and tokenizer pushed successfully!")