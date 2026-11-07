import os
import re
from collections import Counter
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure output directories exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

def clean_text(text):
    """Lowercase and remove punctuation/special characters."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # 1. Load raw data from the new structure
    raw_path = 'data/raw/spam.csv'
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Could not find {raw_path}. Make sure spam.csv is in the data/raw/ folder.")
    
    df = pd.read_csv(raw_path, encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})

    # 2. Deduplicate and handle missing values
    initial_count = len(df)
    df = df.dropna().drop_duplicates(keep='first').reset_index(drop=True)
    duplicates_removed = initial_count - len(df)

    # 3. Feature Engineering for EDA
    df['char_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    df['cleaned_message'] = df['message'].apply(clean_text)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})

    # 4. Statistical Summary
    stats = df.groupby('label')[['char_length', 'word_count']].agg(
        ['count', 'mean', 'median', 'min', 'max', 'std']
    )

    # 5. Plot 1: Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label', hue='label', palette={'ham': '#3498db', 'spam': '#e74c3c'}, legend=False)
    plt.title('Class Distribution: Ham vs. Spam')
    plt.xlabel('Message Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('outputs/figures/class_distribution.png', dpi=300)
    plt.close()

    # 6. Plot 2: Character Length Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='char_length', hue='label', bins=50, kde=True, palette={'ham': '#3498db', 'spam': '#e74c3c'})
    plt.title('Distribution of SMS Character Length by Class')
    plt.xlabel('Character Length')
    plt.ylabel('Frequency')
    plt.xlim(0, 400)
    plt.tight_layout()
    plt.savefig('outputs/figures/message_length_dist.png', dpi=300)
    plt.close()

    # 7. Plot 3: Top 15 Words Comparison
    tfidf_temp = TfidfVectorizer(stop_words='english')
    stop_words = tfidf_temp.get_stop_words()

    def get_top_words(series, n=15):
        words = ' '.join(series).split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        return Counter(words).most_common(n)

    top_spam = pd.DataFrame(get_top_words(df[df['label'] == 'spam']['cleaned_message']), columns=['word', 'count'])
    top_ham = pd.DataFrame(get_top_words(df[df['label'] == 'ham']['cleaned_message']), columns=['word', 'count'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=top_spam, x='count', y='word', ax=axes[0], color='#e74c3c')
    axes[0].set_title('Top 15 Words in Spam Messages')
    axes[0].set_xlabel('Frequency')
    axes[0].set_ylabel('Word')

    sns.barplot(data=top_ham, x='count', y='word', ax=axes[1], color='#3498db')
    axes[1].set_title('Top 15 Words in Ham Messages')
    axes[1].set_xlabel('Frequency')
    axes[1].set_ylabel('Word')

    plt.tight_layout()
    plt.savefig('outputs/figures/top_words_comparison.png', dpi=300)
    plt.close()

    # 8. TF-IDF Vectorization & Saving Artifacts
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['cleaned_message'])

    # Save the files for Phase 2 and Phase 3
    df.to_csv('data/processed/cleaned_spam.csv', index=False)
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

    print("=== PHASE 1 COMPLETE ===")
    print(f"Initial records: {initial_count} | Duplicates removed: {duplicates_removed} | Final records: {len(df)}")
    print("\n--- Statistical Summary ---")
    print(stats.to_string())
    print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape}")
    print("Saved cleaned dataset to: data/processed/cleaned_spam.csv")
    print("Saved TF-IDF vectorizer to: models/tfidf_vectorizer.pkl")
    print("Saved 3 EDA figures to: outputs/figures/")

if __name__ == '__main__':
    main()
