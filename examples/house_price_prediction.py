"""
End-to-end example of using DataScientistAgent for house price prediction.
This example demonstrates:
1. Data loading and preprocessing
2. Feature engineering
3. Model training and evaluation
4. Model persistence
5. Making predictions on new data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ml_agents.agents import DataScientistAgent
from sklearn.datasets import fetch_california_housing

def main():
    # Create our data scientist agent
    agent = DataScientistAgent("HousePricePredictor")
    
    # Load California Housing dataset
    california = fetch_california_housing()
    data = pd.DataFrame(
        california.data,
        columns=california.feature_names
    )
    data['Price'] = california.target
    
    # Save data to CSV for demonstration
    data.to_csv('california_housing.csv', index=False)
    
    # Load data using our agent
    agent.load_data('california_housing.csv')
    
    # Define features and target
    features = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    target = 'Price'
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = agent.preprocess_data(
        target_column=target,
        features=features
    )
    
    # Create and train model
    print("Training model...")
    # Using fewer trees to reduce model size while maintaining good performance
    model = RandomForestRegressor(
        n_estimators=50,  # Reduced from 100 to make the model file smaller
        max_depth=10,     # Limit tree depth to reduce model size
        random_state=42
    )
    agent.train_model(model, X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = agent.evaluate_model(X_test, y_test, task_type='regression')
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save the model
    print("\nSaving model...")
    agent.save_model('house_price_model.joblib')
    
    # Example of making predictions on new data
    print("\nMaking predictions on new data...")
    # Create sample new houses
    new_houses = pd.DataFrame({
        'MedInc': [8.5, 3.0],
        'HouseAge': [20, 40],
        'AveRooms': [6, 4],
        'AveBedrms': [2, 1],
        'Population': [2000, 3000],
        'AveOccup': [2.5, 3.0],
        'Latitude': [34.0, 35.0],
        'Longitude': [-118.0, -119.0]
    })
    
    predictions = agent.make_predictions(new_houses)
    print("\nPredicted house prices for new data:")
    for i, price in enumerate(predictions):
        print(f"House {i+1}: ${price:.2f}k")
    
    # Feature importance analysis
    if hasattr(agent.model, 'feature_importances_'):
        print("\nFeature Importances:")
        importances = pd.DataFrame({
            'Feature': features,
            'Importance': agent.model.feature_importances_
        })
        importances = importances.sort_values('Importance', ascending=False)
        print(importances)

if __name__ == "__main__":
    main()
