from fastapi import FastAPI
from pydantic import BaseModel
import torch
import pickle
from model import LSTMModel

app = FastAPI(title="Next Word Predictor API")

# Load vocab
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

idx_to_word = {i: w for w, i in vocab.items()}

# Load model
model = LSTMModel(vocab_size=len(vocab))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

class InputText(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Next Word Predictor API is Live 🚀"}

@app.post("/predict")
def predict(data: InputText):
    tokens = data.text.split()
    indices = [vocab.get(token, 0) for token in tokens]

    x = torch.tensor(indices).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        predicted_idx = torch.argmax(output, dim=1).item()

    return {
        "input_text": data.text,
        "next_word": idx_to_word[predicted_idx]
    }
