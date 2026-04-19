# 🧠 Emotion Analysis on Social Media

## 🚀 Overview

This project implements a complete **end-to-end machine learning pipeline** for emotion detection from social media text using the **GoEmotions dataset (28 emotion classes)**.

The system performs data processing, feature extraction, model training, evaluation, and visualization — and supports real-time inference with confidence-based predictions.

---

## 🎯 Features

* 🔹 Multi-class emotion classification (28 emotions)
* 🔹 End-to-end ML pipeline automation
* 🔹 Multiple models comparison:

  * Logistic Regression
  * Support Vector Machine (SVM)
  * Naive Bayes
  * Random Forest
  * XGBoost
  * BiLSTM (Deep Learning)
* 🔹 Feature extraction:

  * TF-IDF
  * Bag of Words
  * Word2Vec
  * FastText
* 🔹 Evaluation metrics:

  * Accuracy, F1-score, MCC, Kappa, ROC-AUC
* 🔹 Visualization:

  * Confusion matrices
  * ROC curves
  * Emotion distribution
  * Radar charts
* 🔹 Batch testing with:

  * Top-3 predictions
  * Confidence scoring
  * Neutral/weak emotion detection

---

## 🧠 Best Model Performance

| Model                           | Accuracy | F1 Score |
| ------------------------------- | -------- | -------- |
| 🏆 Logistic Regression (TF-IDF) | ~0.47    | ~0.47    |

---

## 📊 Example Predictions

```
TEXT: I am so happy today!
→ Top 3 Predictions:
   joy : 0.32
   approval : 0.12
   excitement : 0.07

TEXT: I am going to the market
→ Weak / Neutral emotion (confidence: 0.18)
```

---

## ⚙️ Project Structure

```
Emotion-Analysis/
│
├── run_all.py                  # Full pipeline runner
├── test_model.py               # Batch testing script
│
├── 1_data_collection.py
├── 2_data_preprocessing.py
├── 3_feature_extraction.py
├── 4_model_training.py
├── 5_evaluation_metrics.py
├── 6_output_visualization.py
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
git clone https://github.com/YashPawar3116/Sentiment-Analysis.git
cd Sentiment-Analysis

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Run Full Pipeline

```bash
python run_all.py
```

### Resume from a specific part:

```bash
python run_all.py --from 4
```

---

## 🧪 Test the Model

```bash
python test_model.py
```

---

## ⚡ Key Learnings

* Classical ML (TF-IDF + Logistic Regression) can outperform deep learning on moderate datasets
* Handling **multi-class probability distribution** is critical
* Confidence-based filtering improves real-world usability
* End-to-end pipeline design is essential for production systems

---

## 🚀 Future Improvements

* 🌐 Streamlit web interface
* ⚡ FastAPI deployment
* 🧠 Transformer-based optimization (DistilBERT with GPU)
* 📱 Real-time social media integration

---

## 💼 Use Cases

* Social media sentiment monitoring
* Brand reputation analysis
* Customer feedback analysis
* Emotion-aware chat systems

---

## 👨‍💻 Author

**Yash Pawar**
Computer Engineering Student

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
