import streamlit as st
import numpy as np
import pickle

# Load the saved model and scaler
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Diabetes Prediction using Machine Learning")

st.write("Enter patient details to predict diabetes:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Prediction button
if st.button("Predict"):
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age)

    input_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_array)
    prediction = model.predict(std_data)

    if prediction[0] == 0:
        st.success("✅ The person is NOT diabetic")
    else:
        st.error("⚠ The person IS diabetic")
