# The first step involves importing the libraries required for the process:
import os
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

# Then the dataset is loaded as:
corn = pd.read_csv("corn.csv", sep=",")


# Step 1: Understanding the data
print("The dataset size is: ", corn.shape)

print("The columns in the dataset size are: ", corn.columns)

# Using the info() method, we can quickly identify the data type of each
# column and detect null values:"
corn.info()

print("The number of null values in the dataset is confirmed as: ", corn.isna().sum())


# Step 2: Data preparation
# Now that I have a general understanding of the data, some cleaning is
# needed before proceeding with further analysis.


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

# Column names in our refined dataframe are converted to lowercase, and
# spaces are removed for consistency and usability:
corn_subset.columns = [name.lower() for name in corn_subset.columns]
corn_subset.columns = [name.replace(" ", "_") for name in corn_subset.columns]


# Those registries represent:
missing_land = corn_subset["acreage"].isna().sum()
amount_ml = (missing_land / corn.shape[0]) * 100
print(f"The percentage of registries with missing values of cultivated land represent {amount_ml}")
# While removing a large number of missing values is generally not
# advisable, the lack of access to the research team for clarification and
# the limited usefulness of this data for our model, these rows will be
# removed from the dataframe.

# The resulting dataframe is:
filter = corn_subset["acreage"].isna()
corn_subset = corn_subset[~filter]

# It makes sense that farmers in a developing country might have little to no formal education. Therefore, we can reasonably infer that many of them have not achieved any formal academic qualifications.
# We populate the missing values in the education column with "No educated":
corn_subset.loc[corn_subset["education"].isna()] = corn_subset.loc[
    corn_subset["education"].isna()
].fillna("No educated")

print(
    "The main statistics for out clean dataset are: ",
    corn_subset.describe(include="all"),
)


# Step 3: Feature understanding

# Now, it is important to understand how the selected variables behave:
print("The target variable is Yield")
print("There are no outliers visible at first glance")

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

# The not significant variables identifyed were:
not_significant_var = [
    "gender",
    "main_credit_source",
    "farm_records",
    "extension_provider",
    "advisory_format",
    "advisory_language",
]

# The cleaned dataset is filtered to include the significant variables identified above.
corn_cleaned = corn_subset[significant_var]
corn.reset_index
print("The final cleaned data is the following: ")
print(corn_cleaned.head())


# the Working dataset is prepared and splitted as follows:

# Preparation dataset
X = corn_cleaned.drop("yield", axis=1)
y = corn_cleaned["yield"]

# Turning the dataframes into dictionaries:
X_dic = X.to_dict(orient="records")


# Instanciating the vectorizer for Hot Encoding:
dv = DictVectorizer(sparse=False)

# Applying the vectorizer:
X_encoded = dv.fit_transform(X_dic)

print(
    "The Dataset splitted as follows: 60 percent for training, 20 percent for validation, and 20 percent for testing."
)

# We first split for testing
X_full_train, X_test, y_full_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
# Then we split again for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_full_train, y_full_train, test_size=0.25, random_state=42
)

# The lenght of the datasets can be validated as:
print(
    f"The number of registries in the train dataset is {
        len(X_train)}, in the validation dataset is {
            len(X_val)}, and in the test dataset is {
                len(X_test)}.")

# Create output directory
os.makedirs("data_splits", exist_ok=True)


# More robust saving that handles both DataFrames and numpy arrays
def save_to_csv(data, filename):
    if hasattr(data, "to_csv"):  # It's already a DataFrame
        data.to_csv(filename, index=False)
    else:  # It's a numpy array or similar
        pd.DataFrame(data).to_csv(filename, index=False)


# Save feature datasets
save_to_csv(X_train, "data_splits/X_train.csv")
save_to_csv(X_val, "data_splits/X_val.csv")
save_to_csv(X_test, "data_splits/X_test.csv")

# Save target datasets
pd.DataFrame({"target": y_train}).to_csv("data_splits/y_train.csv", index=False)
pd.DataFrame({"target": y_val}).to_csv("data_splits/y_val.csv", index=False)
pd.DataFrame({"target": y_test}).to_csv("data_splits/y_test.csv", index=False)

# Save the vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(dv, f)
