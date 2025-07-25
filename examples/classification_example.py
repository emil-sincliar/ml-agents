from sklearn.ensemble import RandomForestClassifier
from ml_agents.agents import DataScientistAgent
import pandas as pd
import numpy as np

def classification_example():
    """Example of using DataScientistAgent for a classification task"""
    
    # Create synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    })
    
    # Generate target (binary classification)
    y = (X['feature1'] + X['feature2'] > 0).astype(int)
    
    # Save to CSV
    data = X.copy()
    data['target'] = y
    data.to_csv('example_classification.csv', index=False)
    
    # Create and use the agent
    agent = DataScientistAgent("ClassificationAgent")
    
    # Load data
    agent.load_data('example_classification.csv')
    
    # Prepare data
    features = ['feature1', 'feature2', 'feature3']
    X_train, X_test, y_train, y_test = agent.preprocess_data(
        target_column='target',
        features=features
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    agent.train_model(model, X_train, y_train)
    
    # Evaluate model
    metrics = agent.evaluate_model(X_test, y_test, task_type='classification')
    
    # Save model
    agent.save_model('classification_model.joblib')
    
    # Make predictions on new data
    new_data = pd.DataFrame({
        'feature1': [0.5, -0.5],
        'feature2': [0.2, -0.2],
        'feature3': [0.1, -0.1]
    })
    predictions = agent.make_predictions(new_data)
    print(f"\nPredictions for new data: {predictions}")

if __name__ == "__main__":
    classification_example()
