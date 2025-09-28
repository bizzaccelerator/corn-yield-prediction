#!/usr/bin/env python
# coding: utf-8

# To save the model
import pickle

# To check the time between steps:
import time

# The first step involves importing the libraries required for the process:
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer

# Model packages used
from sklearn.model_selection import GridSearchCV, train_test_split

# The first time stamp was taken at:
s0 = time.time()
fs0 = time.gmtime(s0)
strf0 = time.strftime("%D %T", fs0)
print("The process started at: ", strf0)

# Then the dataset is loaded as:
corn = pd.read_csv(
    "C://Users/jober/Data_Projects/corn-yield-prediction/Dataset/corn_data.csv",
    sep=";",
)

## Step 2: Data preparation

# Then, our subset selected for analysis is:
corn_subset = corn[
    [
        "Education",
        "Gender",
        "Age bracket",
        "Household size",
        "Acreage",
        "Fertilizer amount",
        "Laborers",
        "Yield",
        "Main credit source",
        "Farm records",
        "Main advisory source",
        "Extension provider",
        "Advisory format",
        "Advisory language",
    ]
]

# Column names in our refined dataframe are converted to lowercase, and spaces are removed for consistency and usability:
corn_subset.columns = [name.lower() for name in corn_subset.columns]
corn_subset.columns = [name.replace(" ", "_") for name in corn_subset.columns]

# The resulting dataframe is:
filter = corn_subset["acreage"].isna()
corn_subset = corn_subset[~filter]

# It makes sense that farmers in a developing country might have little to no formal education. Therefore, we can reasonably infer that many of them have not achieved any formal academic qualifications.
# We populate the missing values in the education column with "No educated":
corn_subset.loc[corn_subset["education"].isna()] = corn_subset.loc[
    corn_subset["education"].isna()
].fillna("No educated")

## Step 3: Feature understanding


# I identifyed atypical data in the '46-55' age_bracket category. Then,
# I'll use the function described as:
def atypical_data(df, target, variable, label):
    segment_data = df[df[variable] == label][target]
    Q1 = segment_data.quantile(0.25)
    Q3 = segment_data.quantile(0.75)
    IQR = Q3 - Q1
    Inf_Limit = Q1 - (1.5 * IQR)
    Sup_Limit = Q3 + (1.5 * IQR)
    outliers = df[
        (df[variable] == label) & ((df[target] < Inf_Limit) | (df[target] > Sup_Limit))
    ]
    return outliers


# While there is one row in the yield column with 600 units, this individual appears to have greater access to resources. He cultivate four times more land than others and use fertilizer more intensively. He also uses his own financial resources for farming. It is reasonable to infer that this farmer is wealthier than their peers. I concluded that there are no illogical entries in the dataset for this variable.

# Transformation of the data type in the variable 'household_size':
corn_subset["household_size"] = corn_subset["household_size"].apply(str)

# As evaluated in the age_bracket variable, the are some possible outliers at second and nineth categories. Then,
aty_hs_2 = atypical_data(corn_subset, "yield", "household_size", "2")
aty_hs_9 = atypical_data(corn_subset, "yield", "household_size", "9")
# These are records of farmers that break the consistency of the dataframe, particularly due to the small number of laborers they use. Therefore, it is reasonable to classify them as outliers that should be removed.
# Merging the 2 subsets of outliers:
aty_hs_merged = pd.concat([aty_hs_2, aty_hs_9], axis=0, ignore_index=False)
# Dropping the outliers
corn_subset = corn_subset.drop(aty_hs_merged.index)

# Transformation of the data type in the variable 'laborers':
corn_subset["laborers"] = corn_subset["laborers"].apply(str)

# Dropping the outliers for extension provider
aty_gv = atypical_data(
    corn_subset, "yield", "extension_provider", "National Government"
)
corn_subset = corn_subset.drop(aty_gv.index)

# Transformation of the data type in the variable 'acreage':
corn_subset["acreage"] = corn_subset["acreage"].apply(str)

# The final cleaned data is the following:
categorical_columns = [
    "education",
    "age_bracket",
    "household_size",
    "laborers",
    "main_advisory_source",
    "acreage",
]

# The significant variables identifyed were:
significant_var = [
    "education",
    "age_bracket",
    "household_size",
    "laborers",
    "main_advisory_source",
    "acreage",
    "fertilizer_amount",
    "yield",
]

## Step 4: Model identification
# The cleaned dataset is filtered to include the significant variables identified above.
corn_cleaned = corn_subset[significant_var]
corn.reset_index

# The time elapsed until cleaning the dataset
s1 = time.time()
fs1 = time.gmtime(s1)
strf1 = time.strftime("%D %T", fs1)
print("The data was cleaned at: ", strf1)

# Working dataset is prepared and splitted as follows:
X = corn_cleaned.drop("yield", axis=1)
y = corn_cleaned["yield"]

# Turning the dataframes into dictionaries:
X_dic = X.to_dict(orient="records")
# Instanciating the vectorizer for Hot Encoding:
dv = DictVectorizer(sparse=False)
# Applying the vectorizer:
X_encoded = dv.fit_transform(X_dic)

# Dataset splitted as follows: 60% for training, 20% for validation, and 20% for testing.
# We first split for testing
X_full_train, X_test, y_full_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
# Then we split again for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_full_train, y_full_train, test_size=0.25, random_state=42
)


## Training the best regression model for this problem:
s2 = time.time()
fs2 = time.gmtime(s2)
strf2 = time.strftime("%D %T", fs2)
print(f"The model training stated at:", strf2)
# Gradient Boosted Trees (GBT) Model:
# Tunning the hyperparameters is crucial for the optimized model. In this case, I'll define the followings:
param_grid = {
    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.5],
    "n_estimators": [20, 50, 100, 200, 300],
    "max_depth": [3, 5, 7, 10, 20],
}

# I'll use GridSearchCV for exhaustive tuning
grid_gbt = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
)
grid_gbt.fit(X_train, y_train)


## Step 5. Exporting the model
s3 = time.time()
fs3 = time.gmtime(s3)
strf3 = time.strftime("%D %T", fs3)
print(f"The model training finished at:", strf3)
# The selected model will be exported to a binary file (.bin) for later usage:
# The params for the selected model are:
model_params = grid_gbt.best_params_
print(f"The parameters used in the model are:{model_params}")
# Defining the model name:
output_file = f"model_Grid_GBT_learnig={model_params['learning_rate']}_depth={model_params['max_depth']}.bin"
# Saving the model for external usage
with open(output_file, "wb") as f_out:
    pickle.dump((dv, grid_gbt), f_out)

# Final time stamp of the process
s4 = time.time()
fs4 = time.gmtime(s4)
strf4 = time.strftime("%D %T", fs4)
print(f"The model was exported at:", strf4)
