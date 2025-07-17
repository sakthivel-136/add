import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
VOCAB_SIZE = 10000
MAXLEN = 200

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load trained model
model = tf.keras.models.load_model('model.h5')

# Preprocess function
def review_to_sequence(review):
    tokens = review.lower().split()
    sequence = [word_index.get(word, 2) for word in tokens]  # 2 = unknown
    return pad_sequences([sequence], maxlen=MAXLEN)

def predict_sentiment(review):
    sequence = review_to_sequence(review)
    prediction = model.predict(sequence)[0][0]
    sentiment = "🟢 Positive 😀" if prediction > 0.5 else "🔴 Negative 😞"
    return f"{sentiment} ({prediction:.2f})"

# ───────────────────────────────────────────────
# Streamlit UI
st.set_page_config(page_title="Movie Sentiment Classifier", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Classifier")
st.subheader("🔍 Enter a movie review below:")

input_review = st.text_area("✏️ Your Review", height=150)

if st.button("Analyze Sentiment"):
    if input_review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        result = predict_sentiment(input_review)
        st.success(result)

st.markdown("---")
st.markdown("✅ Built using LSTM | Trained on IMDB Dataset")
