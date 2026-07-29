import streamlit as st
import pandas as pd
import joblib
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("🏠 House Price Prediction")
st.write("Enter the house details below and click Predict.")

overall_qual = st.number_input("Overall Quality", min_value=1, max_value=10, value=5)

gr_liv_area = st.number_input("Ground Living Area (sq ft)", value=1500)

garage_cars = st.number_input("Garage Cars", min_value=0, max_value=5, value=2)

total_bsmt_sf = st.number_input("Total Basement Area", value=900)

year_built = st.number_input("Year Built", value=2005)

full_bath = st.number_input("Full Bathrooms", min_value=0, value=2)

bedroom = st.number_input("Bedrooms", min_value=1, value=3)

lot_area = st.number_input("Lot Area", value=7000)

if st.button("Predict Price"):

    input_data = pd.DataFrame([[
        overall_qual,
        gr_liv_area,
        garage_cars,
        total_bsmt_sf,
        year_built,
        full_bath,
        bedroom,
        lot_area
    ]], columns=columns)

    input_data[columns] = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"🏠 Predicted House Price: ${prediction[0]:,.2f}")