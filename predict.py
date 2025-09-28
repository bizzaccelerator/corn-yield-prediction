import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# The model used is refered
model_file = "model.bin"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and metadata at startup
try:
    with open(model_file, "rb") as f_in:
        dv, model = pickle.load(f_in)

    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)

    logger.info(f"Model loaded: {metadata['model_name']} v{metadata['model_version']}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Instanciating the app
app = Flask("yield")


@app.route("/predict", methods=["POST"])
# Function that calculates the target variable:
def predict():
    farmer = request.get_json()
    X = dv.transform([farmer])
    y_pred = model.predict(X)[0]
    result = {
        "Yield prediction": y_pred,
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
