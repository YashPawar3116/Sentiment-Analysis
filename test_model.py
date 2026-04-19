import pickle
import numpy as np
import json

# Load model
with open("data/trained_models/Logistic_Regression_TF-IDF.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]

# Load vectorizer
with open("data/models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load label map
with open("data/models/id2emotion.json") as f:
    id2emotion = {int(k): v for k, v in json.load(f).items()}

# =========================
# TEST SENTENCES
# =========================
test_sentences = [
    # Positive
    "I am so happy today!",
    "This made my day, thank you so much!",
    "I love this so much",
    
    # Negative
    "I am really angry about this",
    "This is so frustrating and annoying",
    "I feel so sad and empty today",
    
    # Fear
    "I am scared of what will happen next",
    "I feel anxious about my exam",
    
    # Surprise
    "I did not expect that at all!",
    "This is so confusing",
    
    # Caring
    "Thank you for being there for me",
    "I care about you a lot",
    
    # Neutral / tricky
    "I am going to the market",
    "Will it rain tomorrow?",
    "The meeting is at 5 PM",
    
    # Mixed
    "I am happy but also a bit nervous",
    "This is exciting but scary at the same time"
]

# Transform all inputs
X = vectorizer.transform(test_sentences)

# Predict probabilities
probs = model.predict_proba(X)

print("\n==================== RESULTS ====================\n")

for text, prob in zip(test_sentences, probs):
    top3 = np.argsort(prob)[-3:][::-1]
    confidence = prob[top3[0]]

    print(f"TEXT: {text}")

    # Adjusted threshold (important fix)
    if confidence < 0.2:
        print(f"→ Weak / Neutral emotion (confidence: {confidence:.2f})")
    else:
        print(f"→ Top 3 Predictions (confidence: {confidence:.2f}):")
        for i in top3:
            print(f"   {id2emotion[i]} : {prob[i]:.4f}")

    print("-" * 50)