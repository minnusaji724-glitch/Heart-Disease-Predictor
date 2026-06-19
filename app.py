import streamlit as st
import numpy as np
import pickle
# Load your trained Decision Tree model and scaler
dt = pickle.load(open("decision_tree_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
st.title("Heart Disease Prediction App")
# Collect patient input
age = st.number_input("Age", min_value=20, max_value=100, value=63)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=145)
chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=233)
fbs = st.selectbox("Fasting Blood Sugar > 120 (0 = No, 1 = Yes)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=2.3)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (3 = Normal, 6 = Fixed defect, 7 = Reversible defect)", [3, 6, 7])
# Prepare input
new_patient = [[age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal]]
new_patient_scaled = scaler.transform(new_patient)
# Prediction
prediction = dt.predict(new_patient_scaled)
if prediction[0] == 1:
    st.error("⚠️ Heart Disease Detected")
else:
    st.success("✅ No Heart Disease Detected")
