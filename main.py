from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI app
app = FastAPI()

# Load model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Input format
class InputData(BaseModel):
    text: str

# Home route
@app.get("/")
def home():
    return {"message": "🚀 API running"}

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    text = data.text

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        return {"result": "⚠️ Attack Detected"}
    else:
        return {"result": "✅ Safe Input"}