from huggingface_hub import HfApi, HfFolder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
api = HfApi()
token = HfFolder.get_token()
repo_id = 'muhamedamil/AI_response_style_model'
api.create_repo(repo_id=repo_id, token= token)

model = AutoModelForSequenceClassification.from_pretrained(r'C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_2\sentiment_analysis\Model\fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained(r'C:\Users\amil\OneDrive\Documents\AI-Driven Personalized Therapy Recommendations system\Module_2\sentiment_analysis\Model\fine_tuned_model')

model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)