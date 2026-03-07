import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your fine-tuned model
model_path = "./my_imdb_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2. Define various test scenarios
test_cases = [
    "This movie was an absolute masterpiece with brilliant acting.",
    "I wasted two hours of my life on this garbage plot.",
    "Oh great, another three-hour movie where nothing actually happens.",
    "The cinematography was stunning, but the script was incredibly weak.",
    "It wasn't as bad as people said, but I wouldn't call it a good film.",
    "I've never seen anything quite like it; truly a unique experience.",
    "I fell asleep halfway through; don't bother watching this."
]

print(f"{'SENTENCE':<70} | {'PREDICTION':<10} | {'CONFIDENCE'}")
print("-" * 100)

model.eval() # Set to evaluation mode
for text in test_cases:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][prediction].item()
    
    label = "POSITIVE" if prediction == 1 else "NEGATIVE"
    
    # Truncate text for display purposes
    display_text = (text[:67] + '..') if len(text) > 67 else text
    print(f"{display_text:<70} | {label:<10} | {confidence:.2%}")