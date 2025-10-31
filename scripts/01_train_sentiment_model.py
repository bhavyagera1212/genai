# scripts/01_train_sentiment_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers.training_args import TrainingArguments
import numpy as np
import torch

# --- Configuration ---
DATASET_PATH = 'data/punjabiData.csv'
MODEL_NAME = 'xlm-roberta-base' # A good multilingual model for fine-tuning
OUTPUT_DIR = 'models/sentiment_classifier'
# NUM_LABELS = 3 # 'positive', 'negative'. Adjust if you have more (e.g., 'neutral')

# --- 1. Load Data ---
print(f"Loading data from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)
print("Data loaded successfully.")
print("Sample data:")
print(df.head())

# --- 2. Map Labels to Integers ---
# Assuming 'positive' and 'negative' sentiments.
# If you have 'neutral', adjust this mapping.
NUM_LABELS = 3
label_mapping = {'positive': 0, 'negative': 1, 'neutral': 2}
# label_mapping = {'positive': 0, 'negative': 1} # Assign integers to your sentiment labels
df['labels'] = df['sentiment'].map(label_mapping)

if df['labels'].isnull().any():
    print("Warning: Some sentiment labels were not mapped. Check your label_mapping.")
    print("Unmapped sentiments:", df[df['labels'].isnull()]['sentiment'].unique())
    df.dropna(subset=['labels'], inplace=True) # Remove rows with unmapped sentiments
    df['labels'] = df['labels'].astype(int)

id_to_label = {v: k for k, v in label_mapping.items()}
print(f"Label mapping: {label_mapping}")

# --- 3. Split Data ---
train_texts, val_texts, train_labels, val_labels = train_test_split(
    list(df['sentence']),
    list(df['labels']),
    test_size=0.2,
    random_state=42,
    stratify=list(df['labels']) # Ensure balanced splits if possible
)
print(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}")

# --- 4. Initialize Tokenizer and Model ---
print(f"Initializing tokenizer and model '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# --- 5. Prepare Dataset for Trainer ---
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# --- 6. Define Metrics ---
def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, predictions, average='weighted')
    acc = accuracy_score(p.label_ids, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="artifacts/sentiment-xlmrb",
    # evaluation_strategy="epoch", 
    eval_strategy="epoch",    # or "steps"
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    warmup_steps=500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=42,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# --- 8. Train Model ---
print("Starting model training...")
trainer.train()
print("Training complete.")

# --- 9. Save Model and Tokenizer ---
print(f"Saving trained model and tokenizer to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Model and tokenizer saved successfully.")

# Save label mapping for inference
import json
with open(f"{OUTPUT_DIR}/label_mapping.json", 'w', encoding='utf-8') as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=4)
with open(f"{OUTPUT_DIR}/id_to_label.json", 'w', encoding='utf-8') as f:
    json.dump(id_to_label, f, ensure_ascii=False, indent=4)

print("Sentiment model training script finished.")