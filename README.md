#  RoBERTa Sentiment Classifier — IMDB Fine-Tuning

Fine-tuning `roberta-base` on the IMDB 50K Movie Reviews dataset for binary sentiment classification (positive / negative). Achieves strong accuracy with a clean, reproducible pipeline built on 🤗 Transformers.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model & Architecture](#model--architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Project Structure](#project-structure)

---

## Overview

This project fine-tunes `roberta-base` on the IMDB movie review dataset for binary sentiment classification. Key design decisions:

- **Full fine-tune** (all layers trainable) — 40K training samples is sufficient to fine-tune RoBERTa without overfitting
- **Differential learning rates** — lower LR for pre-trained transformer layers, higher LR for the randomly initialized classification head
- **Head truncation** — IMDB reviews typically state their sentiment early, so keeping the first 512 tokens captures the most signal
- **No label smoothing** — IMDB labels are clean and unambiguous

---

## Dataset

| Split | Size |
|-------|------|
| Train | 40,000 |
| Test  | 10,000 |

Source: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) via KaggleHub.

Labels: `0` = Negative, `1` = Positive (balanced 50/50 split in the original dataset).

---

## Model & Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Base model | `roberta-base` | Strong general language understanding; no domain-specific biases that fight sentiment learning |
| Classification head | Linear (768 → 2) | Standard Transformers sequence classification head |
| Tokenization | WordPiece, max 512 tokens, head truncation | Sentiment signal is front-loaded in IMDB reviews |
| Parameters trained | All (full fine-tune) | 40K samples is large enough to avoid overfitting |

---

## Training Strategy

### Differential Learning Rates

Two parameter groups with separate learning rates:

```python
optimizer_grouped_parameters = [
    {"params": transformer_layers,         "lr": 2e-5},   # Pre-trained weights — small updates
    {"params": pooler_and_classifier_head, "lr": 1e-4},   # Random init — larger updates OK
]
```

### Hyperparameters

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Transformer layer LR | `2e-5` | Standard fine-tuning LR |
| Classifier head LR | `1e-4` | 5× higher — randomly initialized |
| Optimizer | AdamW | Weight decay regularization |
| Weight decay | `0.01` | Light regularization |
| Batch size | `16` per device | Fits comfortably on a 16GB GPU |
| Epochs | `3` | Sufficient for convergence on 40K samples |
| Warmup ratio | `6%` | Scales warmup with dataset size |
| Mixed precision | `fp16` | Faster training with minimal accuracy loss |
| Label smoothing | `0.0` | Labels are clean — smoothing would hurt |
| Best model selection | Enabled | Loads best checkpoint by eval loss |

---

## Results

The model is evaluated after each epoch with accuracy, a confusion matrix, and a full per-class classification report printed to stdout. Expected final test accuracy on IMDB with this setup is in the range of **~94–95%**, consistent with published benchmarks for `roberta-base` on this dataset.

---

## Project Structure

```
.
├── train.py                        # Main fine-tuning script
├── my_imdb_sentiment_model/        # Saved model + tokenizer (generated after training)
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── model.safetensors
└── roberta_imdb_results/           # Training checkpoints (generated during training)
    └── checkpoint-*/
```

---

## References

- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [IMDB Dataset — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
