from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

# =========================
# PATHS
# =========================
MODEL_PATH = "data/trained_models/Logistic_Regression_TF-IDF.pkl"
VEC_PATH = "data/models/tfidf_vectorizer.pkl"
LABEL_PATH = "data/models/id2emotion.json"

# =========================
# LOAD FILES
# =========================
for path in [MODEL_PATH, VEC_PATH, LABEL_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]

with open(VEC_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(LABEL_PATH) as f:
    id2emotion = {int(k): v for k, v in json.load(f).items()}

print("✅ Model loaded successfully")

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "🚀 Emotion API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        X = vectorizer.transform([text])
        probs = model.predict_proba(X)[0]

        # 🔥 Convert to UI format
        result = {
            id2emotion[i]: float(probs[i])
            for i in range(len(probs))
        }

        return jsonify({
            "probs": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)