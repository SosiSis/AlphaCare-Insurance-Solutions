import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import argparse
import yaml
import sys

def train_model(input_csv, model_output):
    try:
        # Read parameters from params.yaml
        with open("params.yaml", 'r') as params_file:
            params = yaml.safe_load(params_file)
        
        # Get all model parameters
        model_params = params['train']['model_params']
        print(f"Training model with parameters: {model_params}")
        
        # Read the processed CSV
        df = pd.read_csv(input_csv)
        if df.empty:
            raise ValueError("The input CSV file is empty")
        
        # List of feature columns
        feature_columns = [
            'Bank', 'AccountType', 'MaritalStatus', 'Gender', 'mmcode', 'VehicleType',
            'make', 'Model', 'Cylinders', 'cubiccapacity', 'kilowatts', 'bodytype',
            'NumberOfDoors', 'VehicleIntroDate', 'CustomValueEstimate', 'CapitalOutstanding',
            'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'NumberOfVehiclesInFleet'
        ]
        
        # Handle missing values if any
        df = df.dropna(subset=feature_columns)  # Drop rows with missing values in feature columns
        
        # Handle categorical variables
        categorical_columns = ['Bank', 'AccountType', 'MaritalStatus', 'Gender', 'mmcode', 'VehicleType', 'make', 'Model', 'bodytype']
        
        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        # Ensure 'target' is your actual target variable (replace this with your real target column name)
        target = 'target_column'  # Replace with your target column name
        
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in the input data")

        # Define features (X) and target (y)
        X = df.drop(target, axis=1)  # Drop target column from features
        y = df[target]

        # Split the data into training and test sets (optional, based on your use case)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the model
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)

        # Save the trained model
        with open(model_output, "wb") as f:
            pickle.dump(model, f)
            
        print("Model training completed and saved.")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the processed CSV")
    parser.add_argument("--model", required=True, help="Path to save the model file")
    args = parser.parse_args()

    train_model(args.input, args.model)
