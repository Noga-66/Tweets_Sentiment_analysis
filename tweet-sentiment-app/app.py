import json
import re
import string

import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="centered",
)

EMOJI_MAP = {
    "positive": "😄",
    "negative": "😡",
    "neutral": "😐",
}

MODEL_FILES = {
    "SimpleRNN": "models/sentiment_rnn_tf_model.h5",
    "LSTM": "models/sentiment_lstm_tf_model.h5",
    "GRU": "models/sentiment_gru_tf_model.h5",
}


# ---------------------------------------------------------------------------
# NLTK setup (downloads the small resources needed for cleaning/lemmatizing)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting up text tools...")
def load_nltk():
    import nltk

    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    return word_tokenize, set(stopwords.words("english")), WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Vocabulary / config (built once from the training data — see vocab.json)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading vocabulary...")
def load_vocab():
    with open("vocab.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["word2idx"], data["seq_len"], data["classes"]


# ---------------------------------------------------------------------------
# Models (loaded once and cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return {name: tf.keras.models.load_model(path) for name, path in MODEL_FILES.items()}


# ---------------------------------------------------------------------------
# Preprocessing — mirrors the notebook's cleaning/tokenizing/lemmatizing steps
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pad_sequence(seq, max_len, pad_value=0):
    seq = seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))


def preprocess(text, word_tokenize, stop_words, lemmatizer, word2idx, seq_len):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    encoded = [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokens]
    padded = pad_sequence(encoded, seq_len, pad_value=word2idx["<PAD>"])
    return np.array([padded])


def predict(text, model, word_tokenize, stop_words, lemmatizer, word2idx, seq_len, classes):
    input_arr = preprocess(text, word_tokenize, stop_words, lemmatizer, word2idx, seq_len)
    prob_vec = model.predict(input_arr, verbose=0)[0]
    label_id = int(np.argmax(prob_vec))
    sentiment = classes[label_id]
    confidence = float(prob_vec[label_id])
    all_probs = {classes[i]: float(prob_vec[i]) for i in range(len(prob_vec))}
    return sentiment, confidence, all_probs


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🐦 Tweet Sentiment Analyzer")
st.caption("RNN / LSTM / GRU models trained on the Tweets sentiment dataset")

word_tokenize, stop_words, lemmatizer = load_nltk()
word2idx, seq_len, classes = load_vocab()
models = load_models()

model_choice = st.selectbox("Choose a model", list(models.keys()), index=1)

text_input = st.text_area(
    "Enter a tweet",
    placeholder="e.g. I really love this, it made my day!",
    height=120,
)

if st.button("Predict sentiment", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        model = models[model_choice]
        sentiment, confidence, all_probs = predict(
            text_input, model, word_tokenize, stop_words, lemmatizer, word2idx, seq_len, classes
        )
        emoji = EMOJI_MAP.get(sentiment, "")

        st.markdown(f"### Prediction: **{sentiment.upper()}** {emoji}")
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        st.write("**All class probabilities:**")
        for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            st.write(f"- {label}: {prob:.1%}")

st.divider()
with st.expander("Compare all three models on this tweet"):
    if text_input.strip():
        for name, model in models.items():
            sentiment, confidence, _ = predict(
                text_input, model, word_tokenize, stop_words, lemmatizer, word2idx, seq_len, classes
            )
            emoji = EMOJI_MAP.get(sentiment, "")
            st.write(f"**{name}**: {sentiment.upper()} {emoji} ({confidence:.1%})")
    else:
        st.write("Enter a tweet above to compare the models.")
