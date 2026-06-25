# 🛡️ Hybrid GA-Optimized Stacking Ensemble for Multiclass SMS Spam Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official Repository for the JOSCEX Paper:** *Enhancing Threat-Level Granularity in SMS Spam Detection Using Genetic Algorithm and Stacking Ensemble Classifier*

## 📌 Overview
Traditional SMS spam classification models often simplify the problem into a binary task (Spam vs. Ham). However, real-world telecommunication gateways require finer threat-level granularity. This project addresses the original **3-class taxonomic challenge**:
1. 🟢 **Normal**: Legitimate personal communications.
2. 🔴 **Fraud**: Malicious attempts (e.g., phishing, credential theft).
3. 🟠 **Promo**: Commercial broadcasts and marketing.

By utilizing a **Hybrid Genetic Algorithm (GA)** combined with a **Stacking Ensemble Classifier**, this model achieves **93.01% accuracy** on the highly intersected 3-class Indonesian SMS dataset (Wibisono, 2018).

## 🧠 Model Architecture
- **Text Preprocessing**: Sastrawi Stemmer + Custom Indonesian Stopwords.
- **Feature Engineering**: Dual-Orthographic `FeatureUnion` combining Word-level TF-IDF (1,2-gram) and Character-level TF-IDF (3-5 char_wb) to capture both formal words and slang/abbreviations.
- **Feature Selection**: Chi-Square (`SelectKBest`).
- **Base Learners (Level-0)**: Calibrated `LinearSVC`, `RandomForestClassifier`, and `MultinomialNB`.
- **Meta-Learner (Level-1)**: `LogisticRegression`.
- **Hyperparameter Optimization**: `GASearchCV` (Genetic Algorithm).

## 🚀 Live Demo (Web App)
You can try the live model prediction via our Streamlit Community Cloud deployment:
👉 **[Click Here to Access the Web App](https://share.streamlit.io/)** *(Update this link once your app is fully deployed!)*

## 💻 Local Installation
To run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rik-Maghi/sms-spam-hybrid.git
   cd sms-spam-hybrid
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## 📊 Evaluation Metrics
Tested on 20% unseen data (229 messages):
- **Accuracy**: 93.01%
- **Weighted Precision**: 93.05%
- **Weighted Recall**: 93.01%
- **Weighted F1-Score**: 93.02%

## 📝 Citation
If you find this repository useful for your research, please consider citing our paper:
```bibtex
@article{rikon2026hybrid,
  title={Enhancing Threat-Level Granularity in SMS Spam Detection Using Genetic Algorithm and Stacking Ensemble Classifier},
  author={Rikon, et al.},
  journal={Journal of Computer Science and Excellence (JOSCEX)},
  year={2026}
}
```
