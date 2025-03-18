# Import necessary libraries
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import joblib

# Function to load a model
def load_saved_model(model_path):
    return load_model(model_path)

# Function to load a scaler
def load_saved_scaler(scaler_path):
    return joblib.load(scaler_path)

# Prepare a sequence of requests for prediction
def prepare_sequence_request(sequence):
    # 'sequence' is a list of lists, where each sublist represents a time step
    # Since the data is already scaled, we just need to format it correctly
    return np.array([sequence])  # Adding the batch dimension

# Directories
model_data_directory = 'model_data'
scaler_data_directory = 'scaler_data'

# Model and scaler file information
models_and_scalers = {
    'y1': ('model_y1.h5', 'scaler_y1.joblib'),
    'y12': ('model_y12.h5', 'scaler_y12.joblib'),
    'y14': ('model_y14.h5', 'scaler_y14.joblib')
}

# Loop through each model and scaler, load them, and make a prediction
for target_column, (model_file, scaler_file) in models_and_scalers.items():
    model_path = os.path.join(model_data_directory, model_file)
    scaler_path = os.path.join(scaler_data_directory, scaler_file)

    # Load model and scaler
    model = load_saved_model(model_path)
    scaler = load_saved_scaler(scaler_path)

    sequence_data = [
        [0.0, 2.0, 13.05],
        [0.0, 2.0, 13.06666667],
        [0.0, 2.0, 13.08333333],
        [0.0, 2.0, 13.1],
        [0.0, 2.0, 13.11666667]
    ]

    # Prepare the input data for prediction
    request = prepare_sequence_request(sequence_data)

    # Make a prediction
    prediction = model.predict(request)
    print(f'Prediction for model {target_column}: {prediction}')
