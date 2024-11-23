import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the trained model and tokenizer
model_path = "./bert-imdb-model"  
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval() 

# Load the IMDb dataset for evaluation
from datasets import load_dataset

test_dataset = load_dataset("imdb", split="test")

# Preprocess the test dataset
def preprocess_text(text):
    """Tokenizes and encodes the text for the BERT model."""
    return tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

def predict(model, inputs):
    """Runs inference on the model and returns predictions."""
    with torch.no_grad():
        outputs = model(**inputs)
        return torch.argmax(outputs.logits, dim=1).cpu().numpy()

# Prepare the test data for evaluation
test_texts = test_dataset['text']
test_labels = test_dataset['label']
predicted_labels = []
true_labels = []

# Process in batches for better performance
batch_size = 16
for i in range(0, len(test_texts), batch_size):
    batch_texts = test_texts[i:i+batch_size]
    batch_labels = test_labels[i:i+batch_size]
    
    # Preprocess batch texts
    batch_inputs = preprocess_text(batch_texts)
    batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
    
    # Predict labels for the batch
    predictions = predict(model, batch_inputs)
    predicted_labels.extend(predictions)
    true_labels.extend(batch_labels)

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
classification_rep = classification_report(true_labels, predicted_labels, target_names=["Negative", "Positive"])
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Display results
print("Evaluation Results")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_rep)

# Visualizations
# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# 2. Classification Metrics Visualization
metrics = ['Precision', 'Recall', 'F1-Score']
values = [
    classification_rep.split()[-4],  # Precision for Positive
    classification_rep.split()[-3],  # Recall for Positive
    classification_rep.split()[-2]   # F1-Score for Positive
]
values = [float(v) for v in values]

plt.figure(figsize=(6, 4))
sns.barplot(x=metrics, y=values, palette="muted")
plt.ylim(0, 1)
plt.title("Classification Metrics for Positive Sentiment")
plt.ylabel("Score")
plt.show()
