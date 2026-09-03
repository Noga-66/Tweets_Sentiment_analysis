"""
Tweet Sentiment Analyzer - Streamlit Web App
Loads a trained RNN/LSTM/GRU model (from the notebook) and predicts
the sentiment (negative / neutral / positive) of a tweet.
"""

import re
import pickle
import time

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
    initial_sidebar_state="expanded"
)

# أضفنا Animations باستخدام CSS
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
    border-radius: 8px !important;
    font-size: 1.05rem !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes fillBar {
    from { width: 0%; }
}

.result-card {
    border-left: 6px solid var(--moss);
    background: #ffffffb0;
    padding: 1.5rem;
    border-radius: 8px;
    margin-top: 1rem;
    animation: fadeInUp 0.5s ease-out forwards;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.result-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}
.result-card.negative { border-left-color: #a4432f; }
.result-card.neutral  { border-left-color: #8a8163; }
.result-card.positive { border-left-color: #33564f; }

.bar-track {
    background: #e0e0e0;
    border-radius: 5px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    margin-top: 6px;
}
.bar-fill {
    height: 100%;
    border-radius: 5px;
    animation: fillBar 1.2s ease-out forwards;
}
footer, #MainMenu {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Load model + tokenizer
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
    # --- Sidebar ---
    with st.sidebar:
        st.title("🕊️ App Info")
        try:
            _, _, _, config = load_artifacts()
            st.success(f"**Model:** {config.get('best_model', 'N/A')}")
        except:
            st.warning("Model not loaded yet.")
            
        st.markdown("---")
        st.subheader("About this project")
        st.write(
            "This model was trained on the Tweet Sentiment Extraction dataset (Kaggle) "
            "using TensorFlow/Keras. SimpleRNN, LSTM, and GRU were compared, and the best "
            "one was selected based on test set performance."
        )
        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit")

    # --- Main Page ---
    st.title("Tweet Sentiment Analyzer")
    st.write("Type a tweet in English and see what sentiment the model predicts!")

    try:
        model, tokenizer, label_encoder, config = load_artifacts()
    except Exception:
        st.error(
            "Model files were not found in this folder. "
            "Run the notebook first to generate these files."
        )
        st.stop()

    # Feature: Examples Dropdown
    example_options = [
        "✏️ (Type your own tweet)",
        "I had an absolutely wonderful day today! The weather was perfect. ☀️",
        "I am so disappointed with the service. My order is completely ruined! 😡",
        "I am going to the grocery store to pick up some coffee and a few books. ☕"
    ]
    selected_example = st.selectbox("💡 Try a ready example:", example_options)

    # Set text area value based on selection
    if selected_example != example_options[0]:
        text = st.text_area("Tweet", value=selected_example[:-2], height=120) # removing emoji from end for cleaner input
    else:
        text = st.text_area("Tweet", placeholder="e.g. I can't believe how great today turned out!", height=120)

    # Layout for button
    col1, col2 = st.columns([1, 3])
    with col1:
        analyze = st.button("✨ Analyze Sentiment", type="primary", use_container_width=True)

    if analyze:
        if not text.strip():
            st.warning("Please type a tweet first 🙂")
        else:
            # Animation feature: Spinner
            with st.spinner('Analyzing the tweet... 🧠'):
                time.sleep(0.6) # Fake slight delay to show the nice loading animation
                
                cleaned = clean_text(text)
                maxlen = config.get("maxlen", 40)
                seq = tokenizer.texts_to_sequences([cleaned])
                pad = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
                probs = model.predict(pad, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                confidence_score = float(probs[pred_idx]) * 100

            # Toast Notification
            st.toast(f'Analysis complete! Sentiment is {pred_label.capitalize()}', icon='✅')
            
            # Easter Egg Feature: Balloons for high positive
            if pred_label == 'positive' and confidence_score > 85.0:
                st.balloons()

            # Confidence Badge logic
            if confidence_score >= 80:
                badge = f"<span style='background-color:#d4edda; color:#155724; padding:3px 8px; border-radius:12px; font-size:0.8rem;'>High Confidence ({confidence_score:.1f}%)</span>"
            elif confidence_score >= 50:
                badge = f"<span style='background-color:#fff3cd; color:#856404; padding:3px 8px; border-radius:12px; font-size:0.8rem;'>Moderate Confidence ({confidence_score:.1f}%)</span>"
            else:
                badge = f"<span style='background-color:#f8d7da; color:#721c24; padding:3px 8px; border-radius:12px; font-size:0.8rem;'>Low Confidence ({confidence_score:.1f}%)</span>"

            # Display Result Card
            st.markdown(
                f"""
                <div class="result-card {pred_label}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h2 style="margin:0;">{EMOJI.get(pred_label, '')} {pred_label.capitalize()}</h2>
                        {badge}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("---")
            st.markdown("### 📊 Probability Distribution")
            
            # Display Progress Bars
            for label in label_encoder.classes_:
                idx = list(label_encoder.classes_).index(label)
                pct = float(probs[idx]) * 100
                is_bold = "font-weight:bold;" if label == pred_label else ""
                
                st.markdown(
                    f"""
                    <div style="margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; font-size:1rem; {is_bold}">
                            <span style="color:var(--ink);">{label.capitalize()}</span>
                            <span style="color:var(--ink);">{pct:.1f}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:{pct}%; background:{COLORS[label]};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

if __name__ == "__main__":
    main()
