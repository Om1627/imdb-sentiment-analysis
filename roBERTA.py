import kagglehub
import pandas as pd
import numpy as np
import torch
import evaluate
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, Features, Value, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim import AdamW

# ─────────────────────────────────────────────
# 1. DOWNLOAD IMDB DATASET FROM KAGGLE
# ─────────────────────────────────────────────
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
csv_path = f"{path}/IMDB Dataset.csv"

# ─────────────────────────────────────────────
# 2. PREPROCESS IMDB DATA
# ─────────────────────────────────────────────
df = pd.read_csv(csv_path)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df.rename(columns={'review': 'text'})[['text', 'label']]

imdb_ds = Dataset.from_pandas(df)
imdb_split = imdb_ds.train_test_split(test_size=0.2, seed=42)

# ─────────────────────────────────────────────
# 3. ADD SARCASM / IRONY DATASET (tweet_eval)
#    Maps irony labels -> sentiment labels:
#      irony=1 (ironic)     -> 0 (treat as negative / misleading positive)
#      irony=0 (non-ironic) -> 1 (treat as positive / straightforward)
#    This gives the model exposure to ironic language patterns.
# ─────────────────────────────────────────────
print("Loading irony dataset from tweet_eval...")
irony_ds = load_dataset("tweet_eval", "irony")

def remap_irony_labels(example):
    # Ironic text (label=1) is the hard case — map to negative class
    # so the model learns to distrust surface-level positive phrasing
    example['label'] = 0 if example['label'] == 1 else 1
    example['text'] = example['text']
    return example


# Define a unified schema — forces both datasets to identical feature types
# so concatenate_datasets doesn't raise a ValueError on mismatched types
unified_features = Features({
    'text':  Value('string'),
    'label': ClassLabel(num_classes=2, names=['negative', 'positive'])
})

irony_train = irony_ds['train'].map(remap_irony_labels)\
                               .select_columns(['text', 'label'])\
                               .cast(unified_features)
irony_test  = irony_ds['test'].map(remap_irony_labels)\
                              .select_columns(['text', 'label'])\
                              .cast(unified_features)

imdb_train  = imdb_split['train'].cast(unified_features)
imdb_test   = imdb_split['test'].cast(unified_features)

# Merge IMDB + irony datasets (schema-aligned)
combined_train = concatenate_datasets([imdb_train, irony_train])
combined_test  = concatenate_datasets([imdb_test,  irony_test])

dataset = DatasetDict({'train': combined_train, 'test': combined_test})
print(f"Train size: {len(combined_train)} | Test size: {len(combined_test)}")

# ─────────────────────────────────────────────
# 4. TOKENIZATION
#    Using the CASED tokenizer so punctuation like !!!, ALL CAPS,
#    and quoted "words" (sarcasm cues) are preserved.
# ─────────────────────────────────────────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
print(f"Loading tokenizer from: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    """
    Tail-truncation strategy: keeps the LAST 512 tokens of each review.
    Sarcastic punchlines and sentiment reversals are often at the end of
    long reviews, so we prefer the ending over the beginning.
    """
    # Tokenize without truncation first
    batch = tokenizer(
        examples["text"],
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )

    MAX_LEN = 512
    result = {"input_ids": [], "attention_mask": []}

    for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
        if len(ids) > MAX_LEN:
            # Keep the last MAX_LEN tokens (tail truncation)
            ids  = ids[-MAX_LEN:]
            mask = mask[-MAX_LEN:]
        result["input_ids"].append(ids)
        result["attention_mask"].append(mask)

    # Pad the batch to uniform length
    padded = tokenizer.pad(result, padding="max_length", max_length=MAX_LEN)
    return padded

print("Tokenizing datasets...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ─────────────────────────────────────────────
# 5. LOAD MODEL
#    Starting from cardiffnlp/twitter-roberta-base-irony gives the model
#    a strong prior on ironic / sarcastic language patterns before
#    fine-tuning on IMDB sentiment.
# ─────────────────────────────────────────────
print(f"Loading model from: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,   # classification head size may differ
)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last THREE transformer blocks (10, 11 for roberta-base has 12 layers 0-11)
# Plus the pooler and classifier head
for name, param in model.named_parameters():
    if any(f"layer.{i}" in name for i in [9, 10, 11]):
        param.requires_grad = True
    if any(k in name for k in ["pooler", "classifier"]):
        param.requires_grad = True

print("\nTrainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  ✓ {name}")

# ─────────────────────────────────────────────
# 6. METRICS — including per-class F1 to catch
#    sarcasm-related classification bias
# ─────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    print("\n" + "="*40)
    print("CONFUSION MATRIX")
    print("="*40)
    print(pd.DataFrame(
        cm,
        index=['Actual Neg', 'Actual Pos'],
        columns=['Pred Neg', 'Pred Pos']
    ))

    # Per-class F1 — low F1 on one class often signals sarcasm confusion
    print("\nPER-CLASS REPORT (low F1 = sarcasm/irony leakage)")
    print(classification_report(
        labels, predictions,
        target_names=["Negative", "Positive"]
    ))
    print("="*40 + "\n")

    return acc

# ─────────────────────────────────────────────
# 7. OPTIMIZER WITH DIFFERENTIAL LEARNING RATES
#    Transformer layers get a smaller LR to preserve irony knowledge.
#    Classifier head gets a larger LR to adapt to IMDB labels.
# ─────────────────────────────────────────────
optimizer_grouped_parameters = [
    {
        "params": [
            p for n, p in model.named_parameters()
            if "layer" in n and p.requires_grad
        ],
        "lr": 2e-6,   # Tiny LR: preserve pre-trained irony representations
    },
    {
        "params": [
            p for n, p in model.named_parameters()
            if any(k in n for k in ["pooler", "classifier"]) and p.requires_grad
        ],
        "lr": 2e-5,   # Larger LR: classification head adapts faster
    },
]
optimizer = AdamW(optimizer_grouped_parameters)

# ─────────────────────────────────────────────
# 8. TRAINING ARGUMENTS
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./roberta_irony_imdb_results",
    per_device_train_batch_size=16,
    num_train_epochs=5,             # More epochs — sarcasm needs more iterations
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=100,
    warmup_steps=1000,              # Longer warmup to stabilise irony-pretrained weights
    fp16=True,                      # Speed up on RTX 4060
    label_smoothing_factor=0.1,     # Soft labels — sarcastic reviews are ambiguous
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    optimizers=(optimizer, None),
    compute_metrics=compute_metrics,
)

# ─────────────────────────────────────────────
# 9. TRAIN
# ─────────────────────────────────────────────
print("Starting training...")
trainer.train()

# ─────────────────────────────────────────────
# 10. SAVE
# ─────────────────────────────────────────────
model.save_pretrained("./my_imdb_sarcasm_model")
tokenizer.save_pretrained("./my_imdb_sarcasm_model")
print("Fine-tuning complete. Model saved to './my_imdb_sarcasm_model'.")