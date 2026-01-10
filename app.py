import streamlit as st
import pickle
import pandas as pd

# Model aur training columns load karo
model = pickle.load(open('mental_health_model.pkl', 'rb'))
training_columns = pickle.load(open('training_columns.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Mental Health Risk Predictor", page_icon="🧠")

st.title("🧠 Mental Health Risk Predictor")
st.markdown("Enter your details and get an instant risk assessment (Low/Medium/High). This is for awareness only – consult a professional if needed.")

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 35)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
    employment_status = st.selectbox("Employment Status", ["Employed", "Student", "Self-employed", "Unemployed"])
    work_environment = st.selectbox("Work Environment", ["On-site", "Remote", "Hybrid"])
    mental_health_history = st.selectbox("Mental Health History", ["Yes", "No"])
    seeks_treatment = st.selectbox("Seeks Treatment", ["Yes", "No"])

with col2:
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    sleep_hours = st.slider("Average Sleep Hours", 3.0, 10.0, 7.0)
    physical_activity_days = st.slider("Physical Activity Days/Week", 0, 7, 3)
    depression_score = st.slider("Depression Score (0-30)", 0, 30, 10)
    anxiety_score = st.slider("Anxiety Score (0-21)", 0, 21, 8)
    social_support_score = st.slider("Social Support Score (0-100)", 0, 100, 60)
    productivity_score = st.slider("Productivity Score (50-100)", 50, 100, 80)

# Predict button
if st.button("🔍 Predict My Risk Level", type="primary"):
    input_data = pd.DataFrame([{
        'age': age,
        'stress_level': stress_level,
        'sleep_hours': sleep_hours,
        'physical_activity_days': physical_activity_days,
        'depression_score': depression_score,
        'anxiety_score': anxiety_score,
        'social_support_score': social_support_score,
        'productivity_score': productivity_score,
        'gender': gender,
        'employment_status': employment_status,
        'work_environment': work_environment,
        'mental_health_history': mental_health_history,
        'seeks_treatment': seeks_treatment
    }])

    input_encoded = pd.get_dummies(input_data, columns=[
        'gender', 'employment_status', 'work_environment',
        'mental_health_history', 'seeks_treatment'
    ], drop_first=True)

    # Training columns match karo
    for col in training_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[training_columns]

    prediction = model.predict(input_encoded)[0]
    risk_map = {0: "Low 🟢", 1: "Medium 🟡", 2: "High 🔴"}
    risk = risk_map[prediction]

    st.markdown(f"### Predicted Mental Health Risk: **{risk}**")

    if prediction == 2:
        st.error("High Risk – Please consider talking to a professional soon.")
    elif prediction == 1:
        st.warning("Medium Risk – Some signs are present. Self-care might help.")
    else:
        st.success("Low Risk – You're doing great! Keep it up.")
