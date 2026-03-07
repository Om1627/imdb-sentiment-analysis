import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# 1. SET UP THE PAGE
st.set_page_config(page_title="Movie Review Sentiment AI", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("Type a review below to see if the AI thinks it is **Positive** or **Negative**.")

# 2. LOAD THE MODEL (Cached for speed)
# Change this in your app.py
# Old: model_path = "./my_imdb_model"
# New: 
model_path = "zenitsu1607/my-imdb-bert" 

@st.cache_resource
def load_model():
    # Transformers automatically downloads and caches the model from HF
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# 3. USER INPUT
user_input = st.text_area("Enter your movie review here:", placeholder="The acting was great, but the plot...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # 4. INFERENCE LOGIC
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculate probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()
        
        # 5. DISPLAY RESULTS
        label = "POSITIVE" if prediction == 1 else "NEGATIVE"
        color = "green" if label == "POSITIVE" else "red"
        
        st.markdown(f"### Result: :{color}[{label}]")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence:.2%}**")

        # Fun feedback based on sarcasm/nuance
        if confidence < 0.70:
            st.info("💡 The AI is a bit unsure. This might be a mixed or sarcastic review!")
# 2. ADD SIDEBAR FOR MODEL STATS
st.sidebar.title("📊 Model Performance")
st.sidebar.info("These stats are from the 10,000 test reviews in the IMDb dataset.")

# Mock data for demonstration (Replace with your actual trainer results)
# In a real app, you can save your CM to a CSV after training and load it here
cm_data = [[4410, 590], [380, 4620]] 

def plot_cm(data):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'], ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    return fig

st.sidebar.pyplot(plot_cm(cm_data))

st.sidebar.write("""
**What this means:**
* **Top Left**: Correctly caught 'Hate' 📉
* **Bottom Right**: Correctly caught 'Love' 📈
* **Top Right**: Model was 'fooled' by Sarcasm 🎭
""")