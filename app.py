import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠", layout="centered")

# ============================
# Load Model
# ============================
with open("student_wellness_model.pkl", "rb") as f:
    model = pickle.load(f)

# ============================
# Title & Description
# ============================
st.markdown("""
# 🧠 Student Mental Health Risk Predictor  
This tool uses machine learning to estimate a student's **mental stress risk** based on  
sleep, study habits, mood, stress level, and social activity.
---
""")

# ============================
# Input Section
# ============================
st.subheader("📋 Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("😴 Sleep Hours (per day)", 0, 12, 7)
    study = st.slider("📘 Study Hours (per day)", 0, 12, 3)
    social = st.slider("🧑‍🤝‍🧑 Social Hours (per day)", 0, 10, 4)

with col2:
    stress = st.slider("🔥 Stress Level", 1, 10, 5)
    mood = st.slider("🙂 Mood Score", 1, 10, 6)

# ============================
# Predict Button
# ============================
if st.button("🔍 Predict Mental Health Risk", use_container_width=True):

    # Prepare DataFrame
    df_input = pd.DataFrame([{
        "sleep_hours": sleep,
        "stress_level": stress,
        "study_hours": study,
        "social_hours": social,
        "mood_score": mood
    }])

    # Predict
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]  # Probability of high risk

    st.markdown("---")
    st.subheader("🧾 Prediction Result")

    # ============================
    # Output with styling
    # ============================

    if prediction == 1:
        st.error("⚠ **HIGH RISK of Mental Stress Detected**")
    else:
        st.success("✔ **LOW RISK — Student Appears Mentally Stable**")

    # Probability bar
    st.write("### 📊 Risk Probability")
    st.progress(float(proba))

    st.write(f"**High Risk Probability:** `{proba*100:.2f}%`")

    st.markdown("---")

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        st.subheader("📌 Feature Importance")
        importance = model.feature_importances_
        features = ["Sleep", "Stress", "Study", "Social", "Mood"]

        fig, ax = plt.subplots()
        ax.barh(features, importance)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Impact on Prediction")
        st.pyplot(fig)

# ============================
# Footer
# ============================
st.markdown("""
---
### 📘 Disclaimer  
This tool is for educational purposes only.  
It does **not** replace professional psychological evaluation.
""")