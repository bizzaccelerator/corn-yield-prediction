import pickle
from flask import Flask
from flask import request, jsonify

# The model used is refered
model_file = 'model_Grid_GBT_learnig=0.1_depth=3.bin'

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