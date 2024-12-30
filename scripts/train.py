import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import sys
import yaml

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
        
        # Check for missing values and handle them
        df.fillna(value={'CustomValueEstimate': 0, 'WrittenOff': 'No'}, inplace=True)
        
        # Drop non-numeric or irrelevant columns
        df = df.drop(['CustomValueEstimate', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'NumberOfVehiclesInFleet'], axis=1)
        
        # Convert categorical features to numeric if necessary
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['WrittenOff'] = le.fit_transform(df['WrittenOff'])  # Example for categorical columns
        
        if df.empty:
            raise ValueError("The input CSV file is empty")
            
        X = df.drop("species", axis=1)
        y = df["species"]

        # Create and train model
        model = LogisticRegression(**model_params)
        model.fit(X, y)

        # Save model
        with open(model_output, "wb") as f:
            pickle.dump(model, f)
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    train_model("data/pre_proccessed/pre_proccessed.csv", "model.pkl")
