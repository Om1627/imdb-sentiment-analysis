import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Movie Review Sentiment AI", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("Type a review below to see if the AI thinks it is **Positive** or **Negative**.")

model_path = "zenitsu1607/imdb-bert"

# ─────────────────────────────────────────────
# 2. LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ─────────────────────────────────────────────
# 3. COMPUTE REAL CONFUSION MATRIX
#    Runs inference on 1000 IMDB test samples.
#    Cached so it only runs once per session.
# ─────────────────────────────────────────────
@st.cache_data
def compute_real_cm(num_samples=1000, batch_size=32):
    """
    Loads IMDB test split, runs your model on num_samples reviews,
    returns the real confusion matrix and accuracy.
    """
    # Load IMDB test set from HuggingFace
    dataset   = load_dataset("imdb", split=f"test[:{num_samples}]")
    texts     = dataset["text"]
    # IMDB labels: 0=negative, 1=positive
    true_labels = dataset["label"]

    all_preds = []

    # Run inference in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors  = "pt",
            truncation      = True,
            max_length      = 512,
            padding         = True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=-1).tolist()
        all_preds.extend(preds)

    cm       = confusion_matrix(true_labels, all_preds)
    accuracy = np.trace(cm) / np.sum(cm)   # (TP + TN) / total
    return cm, accuracy, num_samples

# ─────────────────────────────────────────────
# 5. USER INPUT & INFERENCE
# ─────────────────────────────────────────────
user_input = st.text_area(
    "Enter your movie review here:",
    placeholder="The acting was great, but the plot..."
)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 512,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs      = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()

        label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        color = "green"    if label == "POSITIVE" else "red"

        st.markdown(f"### Result: :{color}[{label}]")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence:.2%}**")

        # Show both class probabilities
        neg_prob = probs[0][0].item()
        pos_prob = probs[0][1].item()

        col1, col2 = st.columns(2)
        col1.metric("Negative probability", f"{neg_prob:.2%}")
        col2.metric("Positive probability", f"{pos_prob:.2%}")

        if confidence < 0.70:
            st.info("💡 The AI is unsure — this might be a mixed or sarcastic review!")
