import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/model.pkl")

st.set_page_config(page_title="Vehicle Class Predictor", layout="centered")

st.title(" Vehicle Class Prediction System ")
st.write("Predict Vehicle Class using Machine Learning")

brand = st.selectbox("Select Brand of the vehicle ", ['Toyota', 'Tata', 'Hyundai', 'Ashok Leyland', 'Land Rover', 'Ferrari', 'Jeep', 'Volvo', 'McLaren', 'Lamborghini', 'Honda', 'Porsche', 'Skoda', 'Maruti', 'Mahindra', 'Ford', 'Kia'])
hp = st.number_input("Enter Engine power in hp", min_value=0, max_value= 1000, value= 300)
torque = st.number_input("Enter Torque in nm", min_value=0, max_value=20000, value=300)
weight = st.number_input("Enter Weight in kg ", min_value=0, max_value= 20000, value= 2000)
gc = st.number_input("Enter ground clearance in mm", min_value=0, max_value=1000, value=200)
speed = st.number_input("Enter top speed in kmph ", min_value= 0, max_value= 500, value= 180)
acc = st.number_input("Enter time taken to reach 100 from 0 in sec ", min_value=0.0, max_value= 100.0, value= 10.0)
seat = st.number_input("Enter seating capacity ", min_value= 0, max_value= 12, value= 5)
drive = st.selectbox("Select drive train ", ["FWD", "RWD", "AWD"])

if st.button("Predict Vehicle Class"):
    input_data = pd.DataFrame([[brand, 
                           hp, 
                           torque,
                           weight, 
                           gc, 
                           speed, 
                           acc, 
                           seat, 
                           drive]], 
                           columns=['brand', 
                                    'engine_power_hp',
                                    'torque_nm',
                                    'weight_kg', 
                                    'ground_clearance_mm', 
                                    'top_speed_kmph', 
                                    'zero_to_100_sec', 
                                    'seating_capacity', 
                                    'drivetrain'])

    prediction = model.predict(input_data)

    st.success(f"Predicted Vehicle Class based on provided data: {prediction[0][0]}")
