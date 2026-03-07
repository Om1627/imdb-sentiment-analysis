import kagglehub
import pandas as pd
import numpy as np
import torch
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import confusion_matrix


# 1. DOWNLOAD DATASET FROM KAGGLE
# Downloads the 50k labeled reviews dataset
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
csv_path = f"{path}/IMDB Dataset.csv"

# 2. PREPROCESS DATA
df = pd.read_csv(csv_path)
# Map string labels to integers: positive -> 1, negative -> 0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df.rename(columns={'review': 'text'})[['text', 'label']]

# Convert to Hugging Face format and split (80% Train, 20% Test)
full_ds = Dataset.from_pandas(df)
ds_split = full_ds.train_test_split(test_size=0.2)
dataset = DatasetDict({'train': ds_split['train'], 'test': ds_split['test']})

# 3. TOKENIZATION
# DistilBERT uses WordPiece tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    # Truncate to 512 tokens (BERT's max limit)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. LOAD MODEL
# Load pre-distilled BERT with a 2-label classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)

# 5. DEFINE METRICS & CONFUSION MATRIX
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate accuracy
    acc = metric.compute(predictions=predictions, references=labels)
    
    # Generate Confusion Matrix to analyze errors (sarcasm/mixed reviews)
    cm = confusion_matrix(labels, predictions)
    print("\n" + "="*30)
    print("CONFUSION MATRIX")
    print("="*30)
    print(pd.DataFrame(cm, index=['Actual Neg', 'Actual Pos'], 
                          columns=['Pred Neg', 'Pred Pos']))
    print("="*30 + "\n")
    
    return acc

# 6. TRAINING ARGUMENTS
# Optimized hyperparameters for fine-tuning DistilBERT
training_args = TrainingArguments(
    output_dir="./distilbert_imdb_results",
    learning_rate=2e-5,              # Small LR prevents 'forgetting' pre-trained info
    per_device_train_batch_size=16,  # Adjust based on your VRAM
    num_train_epochs=3,              # Transformers usually converge in 3 epochs
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=100,
)

# 7. INITIALIZE TRAINER & START
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training on 50,000 reviews...")
trainer.train()

# 8. SAVE FOR DEPLOYMENT
model.save_pretrained("./my_imdb_model")
tokenizer.save_pretrained("./my_imdb_model")
print("Fine-tuning complete. Model saved to './my_imdb_model'.")