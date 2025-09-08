# Credit-Card-Fraud-Detection-App

This project showcases a complete machine learning pipeline for detecting fraudulent credit card transactions. It includes a Jupyter notebook for training the model and a Streamlit web application for real-time inference.

Files Included
Pipeline.ipynb: This notebook documents the full data science process, from initial data exploration and preprocessing to training a predictive model. It demonstrates how the model and its associated transformations are saved as a single pipeline.

creditcard_pipeline.pkl: This is the pre-trained machine learning model pipeline. The .pkl format allows the web application to load the model directly for making predictions without needing to re-run the training process.

app.py: The core of the web application. It uses the Streamlit framework to create an interactive user interface, loads the pre-trained model, and takes user inputs to predict whether a transaction is likely to be fraudulent.

requirements.txt: A list of all the Python libraries required to run this project. You can use this file to set up the necessary environment.
