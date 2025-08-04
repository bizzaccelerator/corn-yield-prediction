import os
import json
import logging
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# The model used is refered
model_file = 'model_Grid_GBT_learnig=0.1_depth=3.bin'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and metadata at startup
try:
    with open('model.pkl', 'rb') as f_in:
        dv, model = pickle.load(f_in)
                
    with open('model_metadata.json', 'r') as f_in:
        _, metadata = json.load(f_in)
                
    logger.info(f"Model loaded: {metadata['model_name']} v{metadata['model_version']}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Extracting the vectorizer and the model:
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Instanciating the app
app = Flask('yield')

@app.route('/predict', methods=['POST'])
# Function that calculates the target variable:
def predict():
    farmer = request.get_json()
    X = dv.transform([farmer])
    y_pred = model.predict(X)[0]
    result = {
        'Yield prediction': y_pred,
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)