import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠", layout="centered")

# Load Model
with open("student_wellness_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------
# Title & Description
# ---------------------------------------------
st.markdown("""
# 🧠 Student Mental Health Predictor  
""")

# ---------------------------------------------
# Input Section
# ---------------------------------------------
st.subheader("📋 Student Information")

col1, col2 = st.columns(2)

with col1:
    sleep = st.slider("😴 Sleep Hours (per day)", 0, 12, 7)
    study = st.slider("📘 Study Hours (per day)", 0, 12, 3)
    social = st.slider("🧑‍🤝‍🧑 Social Hours (per day)", 0, 10, 4)

with col2:
    stress = st.slider("🔥 Stress Level (1–10)", 1, 10, 5)
    mood = st.slider("🙂 Mood Score (1–10)", 1, 10, 6)

# ---------------------------------------------
# Recommendation Logic
# ---------------------------------------------
def generate_recommendations(sleep, stress, study, social, mood):

    tips = []

    # Sleep
    if sleep < 6:
        tips.append(("⚠ Low Sleep",
                     "Try to get at least **7–8 hours** of sleep. Poor sleep increases stress and reduces focus.",
                     "red"))
    elif sleep > 10:
        tips.append(("😴 Oversleeping",
                     "Sleeping too much can signal low mood. Try to maintain a balance of **7–9 hours**.",
                     "yellow"))
    else:
        tips.append(("✔ Healthy Sleep",
                     "Your sleep pattern looks good! Keep maintaining consistent sleep hours.",
                     "green"))

    # Stress
    if stress >= 8:
        tips.append(("🔥 High Stress",
                     "Practice breathing exercises, short breaks, or talk with a friend/mentor.",
                     "red"))
    elif stress >= 5:
        tips.append(("⚠ Moderate Stress",
                     "Try light physical activity, meditation, or time management techniques.",
                     "yellow"))
    else:
        tips.append(("✔ Low Stress",
                     "Great! Keep balancing your academics and life well.",
                     "green"))

    # Mood
    if mood <= 4:
        tips.append(("😕 Low Mood",
                     "Consider talking to someone you trust or engaging in activities that lift your mood.",
                     "red"))
    elif mood <= 6:
        tips.append(("🙂 Neutral Mood",
                     "Try hobbies, music, or short outdoor walks to uplift your mood.",
                     "yellow"))
    else:
        tips.append(("😄 Good Mood",
                     "Your mood seems stable! Keep doing what makes you feel good.",
                     "green"))

    # Study hours
    if study >= 8:
        tips.append(("📘 Overstudying",
                     "Avoid burnout. Take breaks using techniques such as **Pomodoro (25–5 rule)**.",
                     "yellow"))
    elif study < 2:
        tips.append(("📖 Very Low Study Hours",
                     "Try setting small daily goals to stay consistent.",
                     "yellow"))
    else:
        tips.append(("✔ Balanced Study",
                     "Good! Maintain your study routines consistently.",
                     "green"))

    # Social Life
    if social <= 1:
        tips.append(("🧑‍🤝‍🧑 Low Social Interaction",
                     "Try talking with friends/family or joining campus clubs.",
                     "red"))
    elif social >= 7:
        tips.append(("🎉 Very High Social Activity",
                     "Good social health! But ensure it doesn't affect academics.",
                     "yellow"))
    else:
        tips.append(("✔ Balanced Social Life",
                     "Your social activity seems healthy and well-balanced.",
                     "green"))

    return tips

# ---------------------------------------------
# Predict Button
# ---------------------------------------------
if st.button("🔍 Predict Mental Health", use_container_width=True):

    df_input = pd.DataFrame([{
        "sleep_hours": sleep,
        "stress_level": stress,
        "study_hours": study,
        "social_hours": social,
        "mood_score": mood
    }])

    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    st.markdown("---")
    st.subheader("🧾 Prediction Result")

    if prediction == 1:
        st.error("⚠ **HIGH RISK of Mental Stress**")
    else:
        st.success("✔ **LOW RISK — Student Appears Mentally Stable**")

    # Probability Bar
    st.write("### 📊 Risk Probability")
    st.progress(float(proba))
    st.write(f"**Estimated Risk:** `{proba*100:.2f}%`")

    st.markdown("---")
    st.subheader("💡 Personalized Recommendations")

    # Generate Recommendations
    tips = generate_recommendations(sleep, stress, study, social, mood)

    for title, text, color in tips:
        if color == "red":
            st.error(f"**{title}**\n\n{text}")
        elif color == "yellow":
            st.warning(f"**{title}**\n\n{text}")
        else:
            st.success(f"**{title}**\n\n{text}")

    st.markdown("---")

    # Show feature importance if model supports it
    if hasattr(model, "feature_importances_"):
        st.subheader("📌 Feature Importance")
        importance = model.feature_importances_
        features = ["Sleep", "Stress", "Study", "Social", "Mood"]

        fig, ax = plt.subplots()
        ax.barh(features, importance)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Impact on Prediction")
        st.pyplot(fig)

# Footer
st.markdown("""
 
This ML tool analyzes lifestyle factors such as **sleep**, **stress**, **study hours**, **mood**,  
and **social life** to estimate your mental well-being and offer personalized recommendations.
---
""")
st.markdown("""
---
### 📘 Disclaimer  
This tool is for **educational purposes only** and should not replace professional mental health advice.
""")