"""
Tweet Sentiment Analyzer - Streamlit Web App
Loads a trained RNN/LSTM/GRU model (from the notebook) and predicts
the sentiment (negative / neutral / positive) of a tweet.
"""

import re
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🕊️",
    layout="centered",
)

CUSTOM_CSS = """
<style>
:root {
    --ink: #1c2b2d;
    --paper: #f5f2ea;
    --moss: #33564f;
    --coral: #c9694f;
    --line: #d8d2c2;
}
html, body, [class*="css"]  {
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}
.stApp {
    background-color: var(--paper);
    color: var(--ink);
}
h1, h2, h3 {
    color: var(--moss);
    font-family: "IBM Plex Serif", Georgia, serif;
    letter-spacing: -0.01em;
}
.tweet-box textarea {
    border: 1px solid var(--line) !important;
    border-radius: 4px !important;
    font-size: 1.05rem !important;
}
.result-card {
    border-left: 6px solid var(--moss);
    background: #ffffffb0;
    padding: 1.1rem 1.4rem;
    border-radius: 4px;
    margin-top: 1rem;
}
.result-card.negative { border-left-color: #a4432f; }
.result-card.neutral  { border-left-color: #8a8163; }
.result-card.positive { border-left-color: #33564f; }
.bar-track {
    background: var(--line);
    border-radius: 3px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-top: 4px;
}
.bar-fill {
    height: 100%;
    border-radius: 3px;
}
footer, #MainMenu {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Load model + tokenizer (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("sentiment_model.keras")
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pickle", "rb") as f:
        label_encoder = pickle.load(f)
    with open("config.pickle", "rb") as f:
        config = pickle.load(f)
    return model, tokenizer, label_encoder, config


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


COLORS = {"negative": "#a4432f", "neutral": "#8a8163", "positive": "#33564f"}
EMOJI = {"negative": "🌧️", "neutral": "🌤️", "positive": "☀️"}


def main():
    st.title("Tweet Sentiment Analyzer")
    st.write(
        "Type a tweet in English and see what sentiment the model predicts — "
        "negative, neutral, or positive."
    )

    try:
        model, tokenizer, label_encoder, config = load_artifacts()
    except Exception:
        st.error(
            "Model files were not found in this folder "
            "(sentiment_model.keras, tokenizer.pickle, label_encoder.pickle, config.pickle). "
            "Run the notebook first to generate these files."
        )
        st.stop()

    st.caption(f"Model currently in use: **{config.get('best_model', 'N/A')}**")

    text = st.text_area(
        "Tweet",
        placeholder="e.g. I can't believe how great today turned out!",
        height=120,
        key="tweet_input",
    )

    analyze = st.button("Analyze Sentiment", type="primary")

    if analyze:
        if not text.strip():
            st.warning("Please type a tweet first 🙂")
        else:
            cleaned = clean_text(text)
            maxlen = config.get("maxlen", 40)
            seq = tokenizer.texts_to_sequences([cleaned])
            pad = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
            probs = model.predict(pad, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

            st.markdown(
                f"""
                <div class="result-card {pred_label}">
                    <h3 style="margin:0;">{EMOJI.get(pred_label, '')} {pred_label.capitalize()}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**Probability distribution:**")
            for label in label_encoder.classes_:
                idx = list(label_encoder.classes_).index(label)
                pct = float(probs[idx]) * 100
                st.markdown(
                    f"""
                    <div style="margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.9rem;">
                            <span>{label}</span><span>{pct:.1f}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:{pct}%; background:{COLORS[label]};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("About this project"):
        st.write(
            "This model was trained on the Tweet Sentiment Extraction dataset (Kaggle) "
            "using TensorFlow/Keras. SimpleRNN, LSTM, and GRU were compared, and the best "
            "one was selected based on test set performance."
        )


if __name__ == "__main__":
    main()
