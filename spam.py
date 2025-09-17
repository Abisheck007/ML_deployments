import streamlit as st
import joblib
import re

# Load the trained model, vectorizer, and label encoder
try:
    model = joblib.load('logistic_regression_sms_spam_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model, vectorizer, or label encoder files not found. Please ensure they are saved in the correct location.")
    st.stop()

def preprocess_sms(sms_text):
    """
    Applies preprocessing steps to a single SMS message:
    - Removes special characters and numbers
    - Lowercases the text
    """
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', sms_text)  # Keep only letters and spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Remove extra spaces
    return cleaned_text.lower()

def predict_sms(sms_text, model, vectorizer, label_encoder):
    """
    Predicts whether a new SMS message is spam or ham.
    """
    processed_sms = preprocess_sms(sms_text)
    if not processed_sms: 
        return "Cannot classify empty or highly filtered message."

    vectorized_sms = vectorizer.transform([processed_sms])
    prediction = model.predict(vectorized_sms)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

# Streamlit App
st.title("📩 SMS Spam Detection")

st.write("Enter an SMS message below to check if it is Spam or Ham.")

sms_input = st.text_area("Enter SMS Message:", height=150)

if st.button("Predict"):
    if sms_input:
        prediction = predict_sms(sms_input, model, tfidf_vectorizer, le)
        if prediction == 'spam':
            st.error(f"Prediction: {prediction.upper()}")
        else:
            st.success(f"Prediction: {prediction.upper()}")
    else:
        st.warning("Please enter an SMS message to predict.")
