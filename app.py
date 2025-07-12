import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("failure_predictor_model.pkl")
equipment_encoder = joblib.load("equipment_encoder.pkl")
failure_encoder = joblib.load("failure_encoder.pkl")

# Title
st.title("🧠 Dairy Equipment Failure Prediction")
st.write("Enter sensor values to predict failure mode of equipment.")

# Equipment selection
equipment_list = equipment_encoder.classes_.tolist()
equipment = st.selectbox("Select Equipment", equipment_list)
encoded_equipment = equipment_encoder.transform([equipment])[0]

# SKU Changeover
sku_change = st.radio("SKU Changeover", ["Yes", "No"]) == "Yes"

# Sensor inputs
vib_x = st.slider("Vibration X", 0.0, 0.5, 0.1)
vib_y = st.slider("Vibration Y", 0.0, 0.3, 0.08)
vib_z = st.slider("Vibration Z", 0.0, 2.5, 0.06)
temp = st.slider("Temperature (°F)", 50.0, 110.0, 72.0)
pressure = st.slider("Pressure (psi)", 20.0, 60.0, 45.0)
current = st.slider("Motor Current (A)", 10.0, 20.0, 13.0)
wear = st.slider("Wear Level (%)", 0.0, 100.0, 10.0)

# Prediction
if st.button("🔍 Predict Failure Mode"):
    input_data = pd.DataFrame([[
        encoded_equipment, sku_change, vib_x, vib_y, vib_z,
        temp, pressure, current, wear
    ]], columns=[
        "Equipment_ID", "SKU_Changeover", "Vibration_X", "Vibration_Y",
        "Vibration_Z", "Temperature", "Pressure", "Motor_Current", "Wear_Level"
    ])
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    predicted_class = failure_encoder.inverse_transform([prediction])[0]

    st.success(f"🔧 Predicted Failure Mode: **{predicted_class}**")

    # Show probability chart
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        "Failure Mode": failure_encoder.inverse_transform(range(len(prediction_proba))),
        "Probability": prediction_proba
    }).sort_values("Probability", ascending=False)
    st.bar_chart(prob_df.set_index("Failure Mode"))

