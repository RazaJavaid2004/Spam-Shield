# 🛡️ SpamShield: AI-Based SMS Spam and Scam Message Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

**SpamShield** is an end-to-end machine learning application designed to classify Short Message Service (SMS) texts as either legitimate conversational messages (**Ham**) or unsolicited commercial/fraudulent threats (**Spam / Scam**). Developed as a Complex Computing Activity (CCA) for **CT-264: Programming for Artificial Intelligence** at **NED University of Engineering and Technology**, this project combines regex-driven text preprocessing, unigram/bigram TF-IDF feature extraction, comparative supervised model evaluation, and a real-time **Streamlit** web interface featuring a sigmoid-calibrated **Risk Score**.

📄 **[View Complete Technical Report (PDF)](docs/SpamShield_Report.pdf)** | 📝 **[Download Report (DOCX)](docs/SpamShield_Report.docx)** | 🚀 **[Jump to Quick File Navigator](#-quick-file-navigator)**

---

## 📑 Table of Contents

1. [Project Team (BSCS - Cyber Security)](#-project-team-bscs---cyber-security)
2. [Key Features](#-key-features)
3. [Repository Structure & Quick File Navigator](#-repository-structure--quick-file-navigator)
4. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
5. [Model Performance & Evaluation](#-model-performance--evaluation)
6. [Installation & Usage Guide](#-installation--usage-guide)
7. [Sample Test Cases for Verification](#-sample-test-cases-for-verification)
8. [Dataset Citation & Acknowledgments](#-dataset-citation--acknowledgments)

---

## 👥 Project Team (BSCS - Cyber Security)

| Roll Number | Student Name | Key Responsibilities |
| :--- | :--- | :--- |
| **CT-24278** | **Muhammad Saad** | Exploratory Data Analysis (EDA), Statistical Summaries & Data Visualization |
| **CT-24125** | **Haris Zamir** | Model Training (Naive Bayes, Logistic Regression, Linear SVM) & Metric Evaluation |
| **CT-24136** | **Ali Hassan** | Streamlit Web Interface, Sigmoid Risk Score Calibration & Technical Report |
| **CT-24138** | **Muhammad Raza** | Core Python Pipeline Architecture, Regex Text Preprocessing & TF-IDF Vectorization |

---

## ✨ Key Features

- **Automated Data Hygiene & Deduplication:** Cleanses the raw UCI SMS Spam Collection dataset ($5,572$ rows) in [`src/preprocess.py`](src/preprocess.py) by stripping malformed CSV overflow columns and removing $403$ duplicate campaigns, resulting in $5,169$ unique records saved to [`data/processed/cleaned_spam.csv`](data/processed/cleaned_spam.csv).
- **Comprehensive Exploratory Data Analysis (EDA):** Generates publication-ready statistical distributions comparing character length, word count, class imbalance ($87.37\%$ Ham vs. $12.63\%$ Spam), and top lexical triggers in [`outputs/figures/`](outputs/figures/).
- **N-Gram TF-IDF Vectorization:** Extracts the top $3,000$ unigram and bigram features ($\text{ngram\_range}=(1, 2)$) while filtering English stop words and preserving numeric shortcodes.
- **Multi-Model Benchmarking:** Trains and evaluates **Multinomial Naive Bayes**, **Logistic Regression**, and **Linear Support Vector Machine (`LinearSVC`)** in [`src/train_evaluate.py`](src/train_evaluate.py) on a stratified $80/20$ train-test split.
- **Interactive Streamlit Threat Analyzer:** Provides an intuitive web UI in [`app.py`](app.py) where users can paste any SMS message to receive an instant classification verdict alongside a continuous **Calculated Risk Score ($0\%$–$100\%$)**.

---

## 📂 Repository Structure & Quick File Navigator

```text
Spam-Shield/
├── data/
│   ├── processed/
│   │   └── cleaned_spam.csv              # Deduplicated & regex-cleaned dataset (5,169 records)
│   └── raw/
│       └── spam.csv                      # Original UCI SMS Spam Collection dataset (5,572 records)
├── docs/
│   ├── SpamShield_Report.docx            # Editable CCA Technical Report (Microsoft Word)
│   └── SpamShield_Report.pdf             # Final 20-page CCA Technical Report (PDF)
├── models/
│   ├── best_spam_model.pkl               # Serialized best model (LinearSVC)
│   └── tfidf_vectorizer.pkl              # Fitted 3,000-feature TF-IDF vectorizer
├── outputs/
│   ├── figures/
│   │   ├── class_distribution.png        # Ham vs. Spam class distribution bar chart
│   │   ├── confusion_matrices.png        # Side-by-side confusion matrix heatmaps
│   │   ├── message_length_dist.png       # Character length histogram & KDE by class
│   │   └── top_words_comparison.png      # Top 15 lexical tokens in Spam vs. Ham
│   └── metrics_report.txt                # Exported classification performance summary
├── src/
│   ├── __init__.py                       # Python package initializer
│   ├── preprocess.py                     # Phase 1: Cleaning, EDA & TF-IDF pipeline
│   └── train_evaluate.py                 # Phase 2: Model training, evaluation & serialization
├── .gitattributes                        # Git line-ending & repository attributes
├── .gitignore                            # Git ignore rules
├── app.py                                # Phase 3: Streamlit web interface & risk engine
├── README.md                             # Project documentation & file navigator
└── requirements.txt                      # Python package dependencies
```

### 🔗 Quick File Navigator

Click any link below to view or download the corresponding file directly on GitHub:

| Category | File / Directory | Description |
| :--- | :--- | :--- |
| **💻 Source Code** | [`app.py`](app.py) | Phase 3: Interactive Streamlit web application & sigmoid risk score engine |
| **💻 Source Code** | [`src/preprocess.py`](src/preprocess.py) | Phase 1: Data cleaning, deduplication, statistical EDA, and figure generation |
| **💻 Source Code** | [`src/train_evaluate.py`](src/train_evaluate.py) | Phase 2: TF-IDF transformation, model training, evaluation, and `.pkl` export |
| **💻 Source Code** | [`src/__init__.py`](src/__init__.py) | Source module package marker |
| **📄 Documentation** | [`docs/SpamShield_Report.pdf`](docs/SpamShield_Report.pdf) | Phase 4: Complete 20-page Complex Computing Activity (CCA) Technical Report (PDF) |
| **📄 Documentation** | [`docs/SpamShield_Report.docx`](docs/SpamShield_Report.docx) | Phase 4: Editable Microsoft Word version of the CCA Technical Report |
| **📊 Datasets** | [`data/raw/spam.csv`](data/raw/spam.csv) | Raw Kaggle / UCI SMS Spam Collection dataset ($5,572$ rows) |
| **📊 Datasets** | [`data/processed/cleaned_spam.csv`](data/processed/cleaned_spam.csv) | Cleaned, deduplicated dataset with engineered length & target columns ($5,169$ rows) |
| **🧠 Saved Models** | [`models/best_spam_model.pkl`](models/best_spam_model.pkl) | Trained `LinearSVC` classifier serialized via `joblib` |
| **🧠 Saved Models** | [`models/tfidf_vectorizer.pkl`](models/tfidf_vectorizer.pkl) | Fitted `TfidfVectorizer` ($3,000$ unigram/bigram features) |
| **📈 Outputs & Plots** | [`outputs/metrics_report.txt`](outputs/metrics_report.txt) | Plain-text evaluation table of Accuracy, Precision, Recall, and F1-Score |
| **📈 Outputs & Plots** | [`outputs/figures/class_distribution.png`](outputs/figures/class_distribution.png) | Bar chart of Ham ($4,516$) vs. Spam ($653$) class distribution |
| **📈 Outputs & Plots** | [`outputs/figures/message_length_dist.png`](outputs/figures/message_length_dist.png) | Histogram & KDE comparing character length distributions by class |
| **📈 Outputs & Plots** | [`outputs/figures/top_words_comparison.png`](outputs/figures/top_words_comparison.png) | Horizontal bar plots of the top $15$ words in Spam vs. Ham messages |
| **📈 Outputs & Plots** | [`outputs/figures/confusion_matrices.png`](outputs/figures/confusion_matrices.png) | Confusion matrix heatmaps for Naive Bayes, Logistic Regression, and Linear SVM |
| **⚙️ Configuration** | [`requirements.txt`](requirements.txt) | Required Python libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `joblib`) |
| **⚙️ Configuration** | [`.gitignore`](.gitignore) | Excludes `__pycache__/`, virtual environments, and IDE metadata |
| **⚙️ Configuration** | [`.gitattributes`](.gitattributes) | Repository text and binary file handling rules |

---

## 📊 Exploratory Data Analysis (EDA)

### 1. Structural Statistical Summary

Generated via [`src/preprocess.py`](src/preprocess.py), our analysis reveals that spam messages are nearly twice as long on average ($137.89$ characters) as legitimate messages ($70.46$ characters), clustering tightly near the standard $160$-character SMS billing limit.

| Class Label | Metric | Count ($N$) | Mean ($\mu$) | Median | Min | Max | Std. Dev. ($\sigma$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ham (`0`)** | Character Length | $4,516$ | $70.46$ | $52.0$ | $2$ | $910$ | $56.36$ |
| **Spam (`1`)** | Character Length | $653$ | $137.89$ | $149.0$ | $13$ | $224$ | $30.14$ |
| **Ham (`0`)** | Word Count | $4,516$ | $14.13$ | $11.0$ | $1$ | $171$ | $11.12$ |
| **Spam (`1`)** | Word Count | $653$ | $23.68$ | $25.0$ | $2$ | $35$ | $5.97$ |

### 2. Visual Insights

| [Class Distribution](outputs/figures/class_distribution.png) | [Character Length Distribution](outputs/figures/message_length_dist.png) |
| :---: | :---: |
| [![Class Distribution](outputs/figures/class_distribution.png)](outputs/figures/class_distribution.png) | [![Message Length Distribution](outputs/figures/message_length_dist.png)](outputs/figures/message_length_dist.png) |

#### [Top 15 Lexical Triggers (Spam vs. Ham)](outputs/figures/top_words_comparison.png)

[![Top Words Comparison](outputs/figures/top_words_comparison.png)](outputs/figures/top_words_comparison.png)

---

## 🤖 Model Performance & Evaluation

All three models in [`src/train_evaluate.py`](src/train_evaluate.py) were evaluated on an unseen $20\%$ stratified test set ($1,034$ messages: $903$ Ham, $131$ Spam), with full metrics exported to [`outputs/metrics_report.txt`](outputs/metrics_report.txt). **Linear SVM (`LinearSVC`)** achieved the highest overall **F1-Score ($91.94\%$)**, **Recall ($87.02\%$)**, and **Accuracy ($98.07\%$)**, making it the optimal model for deployment in [`models/best_spam_model.pkl`](models/best_spam_model.pkl).

| Model | Accuracy | Precision | Recall | F1-Score | Confusion Matrix ($TN$ / $FP$ / $FN$ / $TP$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes (`MultinomialNB`)** | $96.91\%$ | **$98.06\%$** | $77.10\%$ | $86.32\%$ | $901$ / $2$ / $30$ / $101$ | Runner-up |
| **Logistic Regression** | $95.65\%$ | $97.78\%$ | $67.18\%$ | $79.64\%$ | $901$ / $2$ / $43$ / $88$ | Baseline |
| **Linear SVM (`LinearSVC`)** | **$98.07\%$** | $97.44\%$ | **$87.02\%$** | **$91.94\%$** | $900$ / $3$ / $17$ / $114$ | 🏆 **Deployed** |

### [Confusion Matrices](outputs/figures/confusion_matrices.png)

[![Confusion Matrices](outputs/figures/confusion_matrices.png)](outputs/figures/confusion_matrices.png)

---

## 🚀 Installation & Usage Guide

### 1. Clone the Repository

```bash
git clone https://github.com/RazaJavaid2004/Spam-Shield.git
cd Spam-Shield
```

### 2. Install Dependencies

Install all required packages listed in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline (Optional – Models & Plots Included)

If you want to regenerate [`data/processed/cleaned_spam.csv`](data/processed/cleaned_spam.csv), the EDA charts in [`outputs/figures/`](outputs/figures/), and the serialized `.pkl` artifacts in [`models/`](models/) from scratch:

```bash
# Phase 1: Data Preprocessing & EDA
python src/preprocess.py

# Phase 2: Model Training & Evaluation
python src/train_evaluate.py
```

### 4. Launch the Streamlit Web App

Run [`app.py`](app.py) using Python's module runner:

```bash
python -m streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

---

## 🧪 Sample Test Cases for Verification

Try pasting these test messages into the Streamlit interface ([`app.py`](app.py)) to observe how **SpamShield** computes threat verdicts and sigmoid-calibrated risk scores (documented with full screenshots in Section 5 of [`docs/SpamShield_Report.pdf`](docs/SpamShield_Report.pdf)):

| ID | Scenario | Sample SMS Input | Verdict | Risk Score |
| :--- | :--- | :--- | :---: | :---: |
| **TC-01** | **Lottery / Prize Scam** | `come there as soon as possibleURGENT! You have won a FREE £1000 cash prize and a new Nokia mobile! Txt CLAIM to 87121 now!` | 🚨 **SPAM / SCAM** | **$91.26\%$** |
| **TC-02** | **Subscription Spam** | `FREE entry into our £250 weekly comp! Text WIN to 80086 now to claim your guaranteed gift voucher. T&Cs apply.` | 🚨 **SPAM / SCAM** | **$84.36\%$** |
| **TC-03** | **University / Academic** | `Hey bro, are we still meeting at the CSIT lab at 11 AM tomorrow to finalize the PAI project report?` | ✅ **LEGITIMATE (HAM)** | **$23.21\%$** |
| **TC-04** | **Everyday Conversational** | `I'm running about 10 minutes late, traffic near the main gate is packed. Don't wait up for me.` | ✅ **LEGITIMATE (HAM)** | **$28.75\%$** |
| **TC-05** | **Borderline Trigger** | `Call me back when you are free, I have some urgent news to tell you about the cash we collected.` | 🚨 **SPAM / SCAM** *(Edge Case)* | **$61.00\%$** |

---

## 📜 Dataset Citation & Acknowledgments

- **Dataset:** Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng'11), Mountain View, CA, USA, 2011. ([Kaggle Dataset Link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) | Local Copy: [`data/raw/spam.csv`](data/raw/spam.csv))
