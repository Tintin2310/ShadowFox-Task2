import pickle
import streamlit as st
import pandas as pd

# Load the trained model (ensure the model file is in the same directory)
with open('car_price_model_gbr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit UI
st.title('Car Price Estimator')

st.sidebar.header('Enter Car Details')

# Input fields for user to enter data
fuel_type = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
year = st.sidebar.number_input('Year of Service', min_value=2000, max_value=2025, step=1)
# Present Price with smaller decimal numbers
present_price = st.sidebar.number_input('Present Price (in INR)', min_value=0.5, max_value=10000000.0, step=0.1)
kms_driven = st.sidebar.number_input('Kilometers Driven', min_value=0, max_value=500000, step=1000)
seller_type = st.sidebar.selectbox('Seller Type', ['Dealer', 'Individual'])
transmission = st.sidebar.selectbox('Transmission Type', ['Manual', 'Automatic'])
owner = st.sidebar.number_input('Number of Previous Owners', min_value=0, max_value=10, step=1)

# Prepare data for prediction
input_data = pd.DataFrame({
    'Fuel_Type': [fuel_type],
    'Year': [year],
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Seller_Type': [seller_type],
    'Transmission': [transmission],
    'Owner': [owner]
})

# Make prediction when button is clicked
if st.sidebar.button('Estimate Price'):
    prediction = model.predict(input_data)
    st.write(f'Estimated Car Price: ₹{prediction[0]:,.2f}')
