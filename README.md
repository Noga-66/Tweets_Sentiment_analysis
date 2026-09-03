# Tweet Sentiment Analyzer — RNN vs LSTM vs GRU

A tweet sentiment analysis project (negative / neutral / positive) built with TensorFlow/Keras,
comparing three model types (SimpleRNN, LSTM, GRU), plus a simple Streamlit web app for live
predictions.

## Project Contents

| File | Description |
|---|---|
| `Tweet_Sentiment_RNN_LSTM_GRU.ipynb` | Full notebook: data cleaning, preprocessing, model building, training, and comparison |
| `Tweets.csv` | Training data |
| `app.py` | Web app (Streamlit) |
| `requirements.txt` | Required libraries |
| `sentiment_model.keras`, `tokenizer.pickle`, `label_encoder.pickle`, `config.pickle` | Trained model artifacts (generated after running the notebook) |

## ⚠️ A note on model accuracy

The "Tweet Sentiment Extraction" dataset (Kaggle) has 3 classes and very short, informal tweets.
Published results and known benchmarks on this exact dataset with RNN/LSTM/GRU trained from
scratch typically land around **65%–78%** test accuracy. Reaching 90%+ with plain RNN/LSTM/GRU
(no Transformers) is not realistic on this dataset. The notebook reports the real number in the
comparison step and explains ways to push accuracy higher (pretrained embeddings, reducing to
2 classes, or using a Transformer model).

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebook (Jupyter) until it generates the model files
jupyter notebook Tweet_Sentiment_RNN_LSTM_GRU.ipynb
# Run all cells (Run All) — this will generate:
#   sentiment_model.keras, tokenizer.pickle, label_encoder.pickle, config.pickle

# 3. Run the web app locally
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Pushing the Project to GitHub

```bash
git init
git add .
git commit -m "Tweet sentiment analyzer: RNN/LSTM/GRU + Streamlit app"
git branch -M main
git remote add origin https://github.com/<username>/<repo-name>.git
git push -u origin main
```

> **Important:** The trained model (`sentiment_model.keras`) and the pickle files must also be
> pushed to GitHub for the deployed app to work (don't forget to run the notebook first so they
> get generated). If the model file is very large (over 100MB), use [Git LFS](https://git-lfs.com/).

## Deploying Online (Free) — Streamlit Community Cloud

The easiest way to deploy the web app directly from GitHub:

1. Push the project to GitHub (steps above) — the repo can be public or private.
2. Go to **https://share.streamlit.io** and sign in with your GitHub account.
3. Click **"New app"** and select your repository and branch (main).
4. In **"Main file path"**, enter: `app.py`
5. Click **Deploy** — wait a minute or two for everything to install.
6. You'll get a live link like: `https://your-app-name.streamlit.app`

Every new `git push` to the repo will trigger an automatic redeploy.

### Alternative: Hugging Face Spaces
You can also connect the same GitHub repo to Hugging Face Spaces using the Streamlit SDK, which
gives you a stable public link with no extra server configuration.

## Model Architecture

- **Embedding layer** (trainable, 128 dimensions)
- **Bidirectional recurrent layers** (SimpleRNN / LSTM / GRU depending on the model), 2 stacked layers
- **Dropout** to reduce overfitting
- **Dense layers** + **Softmax** for 3 classes (negative, neutral, positive)
- **Class weights** to handle class imbalance
- **EarlyStopping + ReduceLROnPlateau** during training
