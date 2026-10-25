import streamlit as st
import joblib
from pathlib import Path

# Load the saved model from the same folder as app.py
model_path = Path(__file__).parent / "spam_model.pkl"
model = joblib.load(model_path)

# Display the interface
st.title("SMS Spam Detector")
st.write("Enter an English message to check whether it looks like spam.")

message = st.text_area("Enter your message:")

# Run a prediction when the button is clicked
if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]

        if prediction == "spam":
            st.error("Prediction: Spam")
        else:
            st.success("Prediction: Not Spam")

st.caption("Predictions may be incorrect. This is a learning project.")