import streamlit as st
import requests

st.title("🌧️ Rainfall Prediction App")

inputs = {
    "maxtemp": st.slider("Max Temp", 10, 50, 30),
    "mintemp": st.slider("Min Temp", 0, 30, 15),
    "humidity": st.slider("Humidity (%)", 0, 100, 60),
    "cloud": st.slider("Cloud Cover", 0, 10, 5),
    "sunshine": st.slider("Sunshine Hours", 0, 12, 6),
}

if st.button("Predict Rainfall"):
    response = requests.post("http://localhost:8000/predict/", json=inputs)
    st.write(f"🌧️ Predicted Rainfall Probability: {response.json()['predicted_rain_probability']:.2%}")
