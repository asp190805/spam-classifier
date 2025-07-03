
import streamlit as st
import pandas as pd
import joblib
import os

# === SET PAGE CONFIGURATION ===
st.set_page_config(page_title="📨 Spam Classifier", layout="centered")
st.title("📨 Email Spam Classifier")

# === LOAD TRAINED MODEL ===
MODEL_FILE = "spam_regression.pkl"


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    else:
        return None

model = load_model()

if model is None:
    st.error("❌ Model file not found. Please ensure 'spam_regression.pkl' exists.")
    st.stop()

# === USER INPUT ===
subject_input = st.text_input("✉️ Enter Email Subject", "")

if subject_input:
    # Create input DataFrame
    input_df = pd.DataFrame([{"Subject": subject_input}])

    # Predict
    prediction = model.predict(input_df)[0]
    label = "📬 HAM (Not Spam)" if prediction == 0 else "📛 SPAM"

    # Display Result
    st.subheader("🔎 Prediction")
    st.success(label if prediction == 0 else label)
    st.balloons()
