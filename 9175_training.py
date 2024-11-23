# Import required libraries
import torch
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Preprocess text
def preprocess_text(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=512)

# Tokenize dataset
train_data = train_data.map(lambda x: preprocess_text(x["text"]), batched=True)
test_data = test_data.map(lambda x: preprocess_text(x["text"]), batched=True)

# Set format for PyTorch tensors
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Implement class weights for handling class imbalance
class_weights = torch.tensor([0.6, 0.4]).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Compute_metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,                   
    per_device_train_batch_size=8,        
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,                  
    gradient_accumulation_steps=4,    
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# Train model
trainer.train()

# Evaluation on test data
eval_results = trainer.evaluate()

# Display evaluation results
print("Evaluation Results:", eval_results)
