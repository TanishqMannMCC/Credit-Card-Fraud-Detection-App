import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the trained pipeline
pipeline = load_model('creditcard_pipeline')

# Set up the app title and description
st.title('💳 Credit Card Fraud Detection App')
st.write(
    'This app uses a machine learning pipeline to predict if a transaction is fraudulent. '
    'Enter the transaction details below.'
)

# Create input fields for the user
st.header("Enter Transaction Details")

# Based on feature importance, V14, V12, V10, and Amount are often significant
v14 = st.slider('V14', min_value=-20.0, max_value=11.0, value=-4.0)
v12 = st.slider('V12', min_value=-19.0, max_value=8.0, value=-6.0)
v10 = st.slider('V10', min_value=-25.0, max_value=24.0, value=-5.0)
amount = st.number_input('Transaction Amount ($)', min_value=0.0, value=150.0)

# Create a button to make predictions
if st.button('Check Transaction'):
    
    # Create a DataFrame from the inputs
    # Column names MUST match the original dataset
    # We create placeholder columns for all V features and then update the ones we have
    data_dict = {f'V{i}': [0.0] for i in range(1, 29)}
    data_dict['Amount'] = [amount]
    data_dict['Time'] = [0] # Placeholder for time
    
    # Update with user input
    data_dict['V14'] = [v14]
    data_dict['V12'] = [v12]
    data_dict['V10'] = [v10]
    
    input_data = pd.DataFrame(data_dict)

    # Make predictions
    prediction = predict_model(pipeline, data=input_data)
    
    # Extract the prediction label and score
    is_fraud = prediction['prediction_label'].iloc[0]
    confidence_score = prediction['prediction_score'].iloc[0]

    # Display the result
    st.subheader('Prediction Result')
    if is_fraud == 1:
        st.error(f'🚨 High Risk of Fraud! (Confidence: {confidence_score:.2%})')
    else:
        st.success(f'✅ Transaction Appears Safe. (Fraud Risk: {confidence_score:.2%})')