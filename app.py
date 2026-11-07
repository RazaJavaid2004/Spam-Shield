import streamlit as st
import joblib
import re
import numpy as np
import os

# Page configuration
st.set_page_config(page_title="SpamShield", page_icon="🛡️", layout="centered")

def clean_text(text):
    """Mirror the exact preprocessing steps from training."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_models():
    """Load and cache the vectorizer and best model."""
    try:
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        model = joblib.load('models/best_spam_model.pkl')
        return tfidf, model
    except FileNotFoundError:
        st.error("Model artifacts not found. Please ensure 'models/tfidf_vectorizer.pkl' and 'models/best_spam_model.pkl' exist.")
        st.stop()

# Initialize models
tfidf, model = load_models()

# App Header
st.title("🛡️ SpamShield")
st.subheader("AI-Based SMS Spam and Scam Detection System")
st.markdown("Paste an SMS message below to analyze its contents and predict whether it is spam or legitimate.")

# User Input
user_input = st.text_area("Message Content:", height=150, placeholder="Enter text message here...")

if st.button("Analyze Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        # Preprocess and Vectorize
        cleaned_input = clean_text(user_input)
        vectorized_input = tfidf.transform([cleaned_input])
        
        # Predict Class
        prediction = model.predict(vectorized_input)[0]
        
        # Calculate Risk Score
        # If model has predict_proba (like Logistic Regression), use it.
        if hasattr(model, "predict_proba"):
            risk_score = model.predict_proba(vectorized_input)[0][1]
        # Otherwise, for LinearSVC, apply a sigmoid function to the decision boundary distance
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(vectorized_input)[0]
            risk_score = 1 / (1 + np.exp(-decision))
        else:
            risk_score = float(prediction)
            
        risk_percentage = risk_score * 100

        st.divider()
        
        # Display Results
        if prediction == 1:
            st.error("🚨 **Threat Detected: SPAM / SCAM**")
            st.metric(label="Calculated Risk Score", value=f"{risk_percentage:.2f}%")
            st.markdown("*Recommendation: Do not interact with links or reply to this sender.*")
        else:
            st.success("✅ **Safe: LEGITIMATE (HAM)**")
            st.metric(label="Calculated Risk Score", value=f"{risk_percentage:.2f}%")
            st.markdown("*This message exhibits standard conversational patterns.*")
