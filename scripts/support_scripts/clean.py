import pandas as pd
import os
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


def load_and_understand_data(filepath):
    """
    Load the corn dataset and provide initial data understanding.
    
    Args:
        filepath: Path to the corn.csv file
        
    Returns:
        pd.DataFrame: Loaded corn dataset
    """
    corn = pd.read_csv(filepath, sep=",")
    
    print("The dataset size is:", corn.shape)
    print("The columns in the dataset size are:", corn.columns)
    
    corn.info()
    print("The number of null values in the dataset is confirmed as:", corn.isna().sum())
    
    return corn


def prepare_data(corn):
    """
    Prepare and clean the corn dataset by selecting relevant columns,
    handling missing values, and standardizing column names.
    
    Args:
        corn: Raw corn DataFrame
        
    Returns:
        pd.DataFrame: Cleaned corn dataset with significant variables only
    """
    # Select subset of columns
    corn_subset = corn[['Education', 'Gender', 'Age bracket',
                        'Household size', 'Acreage', 'Fertilizer amount', 'Laborers',
                        'Yield', 'Main credit source', 'Farm records', 
                        'Main advisory source', 'Extension provider', 'Advisory format', 
                        'Advisory language']]
    
    # Standardize column names
    corn_subset.columns = [name.lower() for name in corn_subset.columns]
    corn_subset.columns = [name.replace(" ", "_") for name in corn_subset.columns]
    
    # Handle missing values in acreage
    missing_land = corn_subset['acreage'].isna().sum()
    amount_ml = (missing_land / corn.shape[0]) * 100
    print(f'The percentage of registries with missing values of cultivated land represent {amount_ml}')
    
    filter_mask = corn_subset['acreage'].isna()
    corn_subset = corn_subset[~filter_mask]
    
    # Handle missing values in education
    corn_subset.loc[corn_subset['education'].isna(), 'education'] = 'No educated'
    
    print("The main statistics for out clean dataset are:", corn_subset.describe(include='all'))
    
    return corn_subset


def select_significant_features(corn_subset):
    """
    Filter dataset to include only significant variables.
    
    Args:
        corn_subset: Cleaned corn DataFrame
        
    Returns:
        pd.DataFrame: Dataset with only significant features
    """
    print("The target variable is Yield")
    print("There are no outliers visible at first glance")
    
    significant_var = ['education', 'age_bracket', 'household_size', 'laborers', 
                       'main_advisory_source', 'acreage', 'fertilizer_amount', 'yield']
    
    not_significant_var = ['gender', 'main_credit_source', 'farm_records', 
                          'extension_provider', 'advisory_format', 'advisory_language']
    
    corn_cleaned = corn_subset[significant_var]
    corn_cleaned.reset_index(drop=True, inplace=True)
    
    print("The final cleaned data is the following:")
    print(corn_cleaned.head())
    
    return corn_cleaned


def encode_and_split_data(corn_cleaned, test_size=0.2, val_size=0.25, random_state=42):
    """
    Encode categorical features and split data into train, validation, and test sets.
    
    Args:
        corn_cleaned: Cleaned DataFrame with significant features
        test_size: Proportion for test set (default: 0.2)
        val_size: Proportion of remaining data for validation (default: 0.25)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, dv)
    """
    # Prepare features and target
    X = corn_cleaned.drop('yield', axis=1)
    y = corn_cleaned['yield']
    
    # Convert to dictionaries for encoding
    X_dic = X.to_dict(orient='records')
    
    # Initialize and apply DictVectorizer
    dv = DictVectorizer(sparse=False)
    X_encoded = dv.fit_transform(X_dic)
    
    print(f"The Dataset splitted as follows: {int((1-test_size)*100)} percent for training, "
          f"{int(test_size*val_size*100)} percent for validation, and {int(test_size*100)} percent for testing.")
    
    # Split for test set
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state
    )
    
    # Split remaining data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_full_train, y_full_train, test_size=val_size, random_state=random_state
    )
    
    print(f'The number of registries in the train dataset is {len(X_train)}, '
          f'in the validation dataset is {len(X_val)}, and in the test dataset is {len(X_test)}.')
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dv


def save_to_csv(data, filename):
    """
    Save data to CSV, handling both DataFrames and numpy arrays.
    
    Args:
        data: Data to save (DataFrame or array)
        filename: Output filename
    """
    if hasattr(data, 'to_csv'):
        data.to_csv(filename, index=False)
    else:
        pd.DataFrame(data).to_csv(filename, index=False)


def save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, dv, output_dir='data_splits'):
    """
    Save all outputs (train/val/test splits and vectorizer) to files.
    
    Args:
        X_train, X_val, X_test: Feature datasets
        y_train, y_val, y_test: Target datasets
        dv: Fitted DictVectorizer
        output_dir: Directory to save outputs (default: 'data_splits')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save feature datasets
    save_to_csv(X_train, f'{output_dir}/X_train.csv')
    save_to_csv(X_val, f'{output_dir}/X_val.csv')
    save_to_csv(X_test, f'{output_dir}/X_test.csv')
    
    # Save target datasets
    pd.DataFrame({'target': y_train}).to_csv(f'{output_dir}/y_train.csv', index=False)
    pd.DataFrame({'target': y_val}).to_csv(f'{output_dir}/y_val.csv', index=False)
    pd.DataFrame({'target': y_test}).to_csv(f'{output_dir}/y_test.csv', index=False)
    
    # Save the vectorizer
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(dv, f)
    
    print(f"\nAll outputs saved successfully to '{output_dir}/' directory and 'vectorizer.pkl'")


def process_corn_data(input_filepath, output_dir='data_splits', test_size=0.2, 
                      val_size=0.25, random_state=42):
    """
    Main function to process corn data from CSV to train/val/test splits.
    
    Args:
        input_filepath: Path to corn.csv file
        output_dir: Directory to save outputs (default: 'data_splits')
        test_size: Proportion for test set (default: 0.2)
        val_size: Proportion of remaining data for validation (default: 0.25)
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing all splits and vectorizer
            {
                'X_train': array, 'X_val': array, 'X_test': array,
                'y_train': Series, 'y_val': Series, 'y_test': Series,
                'vectorizer': DictVectorizer
            }
    """
    # Step 1: Load and understand data
    corn = load_and_understand_data(input_filepath)
    
    # Step 2: Prepare and clean data
    corn_subset = prepare_data(corn)
    
    # Step 3: Select significant features
    corn_cleaned = select_significant_features(corn_subset)
    
    # Step 4: Encode and split data
    X_train, X_val, X_test, y_train, y_val, y_test, dv = encode_and_split_data(
        corn_cleaned, test_size, val_size, random_state
    )
    
    # Step 5: Save outputs
    save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, dv, output_dir)
    
    # Return all components
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'vectorizer': dv
    }


if __name__ == "__main__":
    # Run the complete pipeline
    results = process_corn_data("corn.csv")
    print("\nProcessing complete!")