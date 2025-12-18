import streamlit as st
import torch
import pickle
from model import LSTMModel

# Page config
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Next Word Predictor")
st.write("Enter a sentence and predict the **next word** using an LSTM model.")

# -------------------------
# Load vocab and model
# -------------------------
@st.cache_resource
def load_model():
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    idx_to_word = {i: w for w, i in vocab.items()}

    model = LSTMModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    return model, vocab, idx_to_word


model, vocab, idx_to_word = load_model()

# -------------------------
# User Input
# -------------------------
text = st.text_input(
    "✍️ Enter your text:",
    placeholder="e.g. deep learning is very"
)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Next Word 🚀"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        tokens = text.split()
        indices = [vocab.get(token, 0) for token in tokens]

        x = torch.tensor(indices).unsqueeze(0)

        with torch.no_grad():
            output = model(x)
            predicted_idx = torch.argmax(output, dim=1).item()

        next_word = idx_to_word[predicted_idx]

        st.success(f"**Predicted Next Word:** `{next_word}`")
