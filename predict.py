import pickle

# The model used is refered
model_file = 'model_Grid_GBT_learnig=0.1_depth=3.bin'

# A farmer of test is defined as:
farmer = {'education': 'Certificate',
        'age_bracket': '36-45',
        'household_size': '7',
        'laborers': '2',
        'main_advisory_source': 'Radio',
        'acreage': '2.0',
        'fertilizer_amount': 50}

# Extracting the vectorizer and the model:
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Calculating the target variable as:
X = dv.transform([farmer])
y_pred = model.predict(X)[0]

print('Input: ', farmer)
print('Yield prediction: ', y_pred)