from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib

app = FastAPI()

# Load the trained pipeline once when the server starts
model_path = Path(__file__).parent / "spam_model.pkl"
model = joblib.load(model_path)


# Define the expected request data
class MessageRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "Spam Detector API is running!"}


@app.post("/predict")
def predict_message(request: MessageRequest):
    # Get the message and remove surrounding whitespace
    text = request.message.strip()

    if text == "":
        raise HTTPException(
            status_code=400,
            detail="Please provide a non-empty message."
        )

    # Predict using the saved pipeline
    prediction = model.predict([text])[0]

    return {"prediction": str(prediction)}