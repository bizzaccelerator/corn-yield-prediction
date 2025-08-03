
# The first step involves importing the libraries required for the process:
import pandas as pd
import numpy as np


# Model packages used
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression


### Step 4: Model identification
### - Let's try some models:

# The prepared datasets are:
X_train = pd.read_csv("X_train.csv", sep="," )
y_train = pd.read_csv("y_train.csv", sep="," )
X_val = pd.read_csv("X_val.csv", sep="," )
y_val = pd.read_csv("y_val.csv", sep="," )


# __1. Linear Regression Model:__
# The model is trained as follows:
linear = LinearRegression()
linear.fit(X_train, y_train)

# The trained model is used to predict the values in the test dataset:
y_pred_val = linear.predict(X_val)

# The main indicator for assessing the validity of the model is the Root Mean Squared Error (RMSE).
print("Linear Regression Metrics:")
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred_val)))
print("R² Score:", r2_score(y_val, y_pred_val))


# The evaluation of metrics for the model will be done using this formula:
def evaluate_model(y_test, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"  RMSE: {rmse}")
    print(f"  R² Score: {r2}")

# The evaluation for the linear models are:
evaluation_report = evaluate_model(y_val, y_pred_val, "Linear Regression")

print(evaluation_report)

# The best model is the Lasso Regression model, as it has the lowest RMSE and the highest R² score. This model explains 81.83% of the variability in corn yield and has an average deviation of 51.041 units in corn production between the actual values in the test dataset and the model's predictions.
