import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('models', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

def main():
    # 1. Load processed data
    processed_path = 'data/processed/cleaned_spam.csv'
    if not os.path.exists(processed_path):
        raise FileNotFoundError("Processed dataset not found! Please run 'python src/preprocess.py' first.")
    
    df = pd.read_csv(processed_path)
    df = pd.read_csv(processed_path)
    X_text = df['cleaned_message'].fillna('').astype(str)
    y = df['target']
    

    # 2. Stratified Train/Test Split (80% train, 20% test)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Fit TF-IDF on training text and transform both
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)
    
    # Save vectorizer for inference app
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

    # 4. Define Models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(random_state=42, dual='auto')
    }

    results = []
    trained_models = {}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, model) in enumerate(models.items()):
        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        trained_models[name] = model
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx])
        axes[idx].set_title(f'{name}\nConfusion Matrix')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('Actual Label')

    plt.tight_layout()
    plt.savefig('outputs/figures/confusion_matrices.png', dpi=300)
    plt.close()

    results_df = pd.DataFrame(results)
    
    # Save metrics report
    report_text = f"=== MODEL EVALUATION REPORT ===\n\n{results_df.to_string(index=False)}\n"
    with open('outputs/metrics_report.txt', 'w') as f:
        f.write(report_text)

    # Save best model based on F1-Score
    best_model_name = results_df.sort_values(by='F1-Score', ascending=False).iloc[0]['Model']
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, 'models/best_spam_model.pkl')

    print("=== PHASE 2 COMPLETE ===")
    print(results_df.to_string(index=False))
    print(f"\nBest Model: {best_model_name} saved to models/best_spam_model.pkl")
    print("Saved confusion matrices to outputs/figures/confusion_matrices.png")
    print("Saved evaluation metrics to outputs/metrics_report.txt")

if __name__ == '__main__':
    main()