import streamlit as st
import pandas as pd
import pickle

# Load model
with open("student_wellness_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧠 Student Mental Health Risk Predictor")
st.write("Enter the student details below:")

sleep = st.slider("Sleep Hours per Day", 0, 12, 7)
stress = st.slider("Stress Level (1–10)", 1, 10, 5)
study = st.slider("Study Hours per Day", 0, 12, 3)
social = st.slider("Social Hours per Day", 0, 10, 4)
mood = st.slider("Mood Score (1–10)", 1, 10, 6)

if st.button("Predict"):
    df_input = pd.DataFrame([{
        "sleep_hours": sleep,
        "stress_level": stress,
        "study_hours": study,
        "social_hours": social,
        "mood_score": mood
    }])

    prediction = model.predict(df_input)[0]

    if prediction == 1:
        st.error("⚠ HIGH RISK of mental stress")
    else:
        st.success("✔ LOW RISK")