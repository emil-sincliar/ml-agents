import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from typing import Optional, Union, List, Dict, Any
import joblib

class DataScientistAgent:
    """
    A data scientist agent that helps with building and evaluating machine learning models.
    """
    
    def __init__(self, name: str):
        """
        Initialize the DataScientistAgent.
        
        Args:
            name (str): Name of the agent
        """
        self.name = name
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, data_path: str) -> None:
        """
        Load data from a CSV file.
        
        Args:
            data_path (str): Path to the CSV file
        """
        try:
            self.data = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print("\nFirst few rows of the data:")
            print(self.data.head())
            print("\nData info:")
            print(self.data.info())
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def preprocess_data(self, 
                       target_column: str,
                       features: List[str],
                       test_size: float = 0.2,
                       random_state: int = 42) -> tuple:
        """
        Preprocess data and split into train/test sets.
        
        Args:
            target_column (str): Name of the target column
            features (List[str]): List of feature columns
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        self.feature_names = features
        X = self.data[features]
        y = self.data[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nData split complete:")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model, X_train, y_train) -> None:
        """
        Train the specified model.
        
        Args:
            model: The machine learning model to train
            X_train: Training features
            y_train: Training target
        """
        try:
            self.model = model
            self.model.fit(X_train, y_train)
            print(f"\nModel {type(model).__name__} trained successfully")
        except Exception as e:
            print(f"Error training model: {e}")
            
    def evaluate_model(self, X_test, y_test, task_type: str = 'classification') -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            task_type (str): Either 'classification' or 'regression'
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model trained. Please train model first.")
            
        y_pred = self.model.predict(X_test)
        metrics = {}
        
        if task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            print(f"\nClassification metrics:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
        else:
            mse = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(mse)
            metrics['r2'] = r2_score(y_test, y_pred)
            print(f"\nRegression metrics:")
            print(f"Root Mean Squared Error: {metrics['rmse']:.4f}")
            print(f"R² Score: {metrics['r2']:.4f}")
            
        return metrics
    
    def make_predictions(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            new_data (pd.DataFrame): New data to make predictions on
            
        Returns:
            np.ndarray: Model predictions
        """
        if self.model is None:
            raise ValueError("No model trained. Please train model first.")
            
        if not all(feature in new_data.columns for feature in self.feature_names):
            raise ValueError("New data doesn't contain all required features")
            
        new_data = new_data[self.feature_names]
        new_data = new_data.fillna(new_data.mean())
        scaled_data = self.scaler.transform(new_data)
        
        predictions = self.model.predict(scaled_data)
        print(f"\nPredictions made for {len(predictions)} samples")
        return predictions
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and scaler to disk with compression.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Please train model first.")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        # Use high compression to reduce file size
        joblib.dump(model_data, filepath, compress=('gzip', 9))
        print(f"\nModel saved successfully to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            print(f"\nModel loaded successfully from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
