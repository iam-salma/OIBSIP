import streamlit as st
import numpy as np
import joblib
import os
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

bow_model = joblib.load(os.path.join(MODELS_DIR, 'bow_model.pkl'))
tfidf_model = joblib.load(os.path.join(MODELS_DIR, 'tfidf_model.pkl'))
wnl = joblib.load(os.path.join(MODELS_DIR, 'wnl.pkl'))
cv = joblib.load(os.path.join(MODELS_DIR, 'cv.pkl'))
tv = joblib.load(os.path.join(MODELS_DIR, 'tv.pkl'))

# Function to preprocess input text
def clean_text(text):
    text=re.sub('[^a-zA-z]',' ',text).lower().split()
    text=[wnl.lemmatize(word, pos='v') for word in text if not word in stopwords.words('english')]
    text=' '.join(text)
    return text

# Streamlit UI
st.set_page_config(page_title="📩 Spam Classifier", layout="centered")
st.title("📩 SMS Spam Classifier")
st.markdown("Enter a message to check whether it's **Spam** or **Not Spam**.")

# Input box
message = st.text_area("Type your message here...")

if st.button("🔍 Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess and transform
        cleaned = clean_text(message)
        vect = cv.transform([cleaned])
        pred = bow_model.predict(vect)[0]

        # Output
        if pred == 1:
            st.error("🚨 This message is **SPAM**.")
        else:
            st.success("✅ This message is **NOT SPAM**.")
