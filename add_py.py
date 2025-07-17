import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import os

# Constants
VOCAB_SIZE = 10000
MAXLEN = 200

MODEL_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/model.h5"  # ⬅️ Replace this

# Download model from GitHub if not present
if not os.path.exists("model.h5"):
    with st.spinner("📦 Downloading model..."):
        r = requests.get(MODEL_URL)
        with open("model.h5", "wb") as f:
            f.write(r.content)
        st.success("✅ Model downloaded!")

# Load model
model = tf.keras.models.load_model("model.h5")

# Load IMDB word index
word_index = imdb.get_word_index()

# Preprocess text
def review_to_sequence(review):
    tokens = review.lower().split()
    sequence = [word_index.get(word, 2) for word in tokens]  # 2 is for unknown
    return pad_sequences([sequence], maxlen=MAXLEN)

def predict_sentiment(review):
    sequence = review_to_sequence(review)
    prediction = model.predict(sequence)[0][0]
    sentiment = "🟢 Positive 😀" if prediction > 0.5 else "🔴 Negative 😞"
    return f"{sentiment} ({prediction:.2f})"

# ───────────────────────────────────────
# Streamlit UI
st.set_page_config(page_title="Sentiment Classifier", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Classifier")
st.markdown("🔍 Enter a movie review and check if it's Positive or Negative.")

review = st.text_area("✏️ Your Movie Review")

if st.button("🔎 Predict Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter a valid review.")
    else:
        output = predict_sentiment(review)
        st.success(output)

st.markdown("---")
st.caption("📊 Powered by LSTM • IMDB dataset • Deployed with Streamlit")
