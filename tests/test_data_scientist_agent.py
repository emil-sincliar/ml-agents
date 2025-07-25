import pytest
import pandas as pd
import numpy as np
from ml_agents.agents import DataScientistAgent
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
    })
    y = (X['feature1'] + X['feature2'] > 0).astype(int)
    
    data = X.copy()
    data['target'] = y
    return data

@pytest.fixture
def agent():
    """Create a DataScientistAgent instance"""
    return DataScientistAgent("TestAgent")

def test_agent_initialization(agent):
    """Test agent initialization"""
    assert agent.name == "TestAgent"
    assert agent.data is None
    assert agent.model is None

def test_data_preprocessing(agent, sample_data, tmp_path):
    """Test data preprocessing functionality"""
    # Save sample data to temporary file
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Load and preprocess data
    agent.load_data(str(data_path))
    features = ['feature1', 'feature2']
    X_train, X_test, y_train, y_test = agent.preprocess_data(
        target_column='target',
        features=features
    )
    
    # Check shapes
    assert X_train.shape[1] == len(features)
    assert len(X_train) + len(X_test) == len(sample_data)

def test_model_training(agent, sample_data, tmp_path):
    """Test model training functionality"""
    # Save sample data to temporary file
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    
    # Prepare and train model
    agent.load_data(str(data_path))
    features = ['feature1', 'feature2']
    X_train, X_test, y_train, y_test = agent.preprocess_data(
        target_column='target',
        features=features
    )
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    agent.train_model(model, X_train, y_train)
    
    # Test model evaluation
    metrics = agent.evaluate_model(X_test, y_test, task_type='classification')
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1

def test_model_persistence(agent, sample_data, tmp_path):
    """Test model saving and loading"""
    # Save sample data to temporary file
    data_path = tmp_path / "test_data.csv"
    model_path = tmp_path / "test_model.joblib"
    sample_data.to_csv(data_path, index=False)
    
    # Train and save model
    agent.load_data(str(data_path))
    features = ['feature1', 'feature2']
    X_train, X_test, y_train, y_test = agent.preprocess_data(
        target_column='target',
        features=features
    )
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    agent.train_model(model, X_train, y_train)
    agent.save_model(str(model_path))
    
    # Load model and make predictions
    new_agent = DataScientistAgent("LoadedAgent")
    new_agent.load_model(str(model_path))
    
    # Test predictions
    new_data = pd.DataFrame({
        'feature1': [0.5],
        'feature2': [0.2]
    })
    predictions = new_agent.make_predictions(new_data)
    assert len(predictions) == 1
