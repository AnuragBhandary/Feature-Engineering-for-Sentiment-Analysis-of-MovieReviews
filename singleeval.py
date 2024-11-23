import torch
from transformers import BertForSequenceClassification, BertTokenizer

model_path = "./bert-imdb-model" 
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

def preprocess_text(text):
    """Tokenizes and encodes the text for the BERT model."""
    return tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

def predict_single_review(review_text):
    inputs = preprocess_text(review_text)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()  # Get the predicted class (0 or 1)
    
    return "Positive" if predicted_class == 1 else "Negative"

# Test
review = "This movie was not worth my time"
prediction = predict_single_review(review)
print(f"{review}\nThe sentiment of the review is: {prediction}")
