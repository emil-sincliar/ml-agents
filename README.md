# ML Agents

A collection of intelligent agents for various data science and machine learning tasks.

## Online Model Utilities

Access our online model splitting and combining tool:
[ML Agents Utility App](https://ml-agents-iowu6wwfewhqgeeazyvuci.streamlit.app/)

This web application allows you to:
- Split large model files into smaller chunks for easier sharing
- Combine previously split model chunks back into a complete model
- Handle models larger than GitHub's 100MB file limit

## Installation

```bash
pip install -e .
```

## Agents Available

1. DataScientistAgent - Helps with building and evaluating machine learning models
   - Data loading and preprocessing
   - Model training and evaluation
   - Feature scaling
   - Model persistence
   - Prediction making

## Usage

```python
from ml_agents.agents import DataScientistAgent

# Create an agent
agent = DataScientistAgent("ModelBuilder")

# Load data
agent.load_data("your_dataset.csv")

# Preprocess and train
features = ["feature1", "feature2", "feature3"]
X_train, X_test, y_train, y_test = agent.preprocess_data(
    target_column="target",
    features=features
)

# See examples directory for more detailed usage examples
```

## Model File Management

### Using the Web Interface

1. Visit our [Streamlit App](https://ml-agents-iowu6wwfewhqgeeazyvuci.streamlit.app/)
2. For splitting large models:
   - Upload your model file
   - Choose chunk size (max 90MB per chunk)
   - Download the resulting ZIP file containing model chunks
3. For combining chunks:
   - Upload the ZIP file containing model chunks
   - Download the reconstructed model file

### Using Command Line

Split a model:
```bash
python scripts/model_utils.py split path/to/model.joblib
```

Combine chunks:
```bash
python scripts/model_utils.py combine path/to/chunks/directory
```

## Development

1. Clone the repository
2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest tests/
```

## License

MIT License
