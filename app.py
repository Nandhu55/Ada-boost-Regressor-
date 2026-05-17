import streamlit as st
import numpy as np
import pickle

with open("adaboost_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="AdaBoost Regressor Car Price Prediction",
    layout="centered"
)

st.title("🚗 Car Price Prediction using AdaBoost Regressor")

st.write("Enter car details to predict the estimated car price.")

car_age = st.number_input(
    "Car Age (Years)",
    min_value=0,
    max_value=30,
    value=5
)

mileage = st.number_input(
    "Mileage (km)",
    min_value=0,
    max_value=500000,
    value=50000
)

engine_size = st.number_input(
    "Engine Size (Liters)",
    min_value=0.5,
    max_value=10.0,
    value=2.0
)

horse_power = st.number_input(
    "Horse Power",
    min_value=50,
    max_value=1000,
    value=150
)

fuel_efficiency = st.number_input(
    "Fuel Efficiency (km/l)",
    min_value=1.0,
    max_value=50.0,
    value=15.0
)

brand_value = st.slider(
    "Brand Value Rating",
    min_value=1,
    max_value=10,
    value=5
)

owner_count = st.number_input(
    "Previous Owners",
    min_value=1,
    max_value=10,
    value=1
)

if st.button("Predict Car Price"):

    input_data = np.array([[
        car_age,
        mileage,
        engine_size,
        horse_power,
        fuel_efficiency,
        brand_value,
        owner_count
    ]])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    st.success(f"Estimated Car Price: ₹ {prediction:,.2f}")

st.markdown("---")
st.write("Built using Streamlit and AdaBoost Regressor")