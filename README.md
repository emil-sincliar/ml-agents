# ML Agents

A collection of intelligent agents for various data science and machine learning tasks.

## Model Manager: Your Tool for Handling Large Models

Our user-friendly Model Manager helps you share large machine learning models easily! Try it here:
[ML Agents Utility App](https://ml-agents-iowu6wwfewhqgeeazyvuci.streamlit.app/)

### What Can It Do?

1. **Split Large Models into Smaller Pieces**
   - Have a model that's too big to email or upload?
   - Our tool breaks it into smaller, manageable pieces
   - All pieces are neatly packed into one zip file
   - Perfect for sharing models that are larger than platform limits (like GitHub's 100MB limit)

2. **Put the Pieces Back Together**
   - Got a zip file with model pieces?
   - Simply upload it to our tool
   - The tool automatically rebuilds your model
   - Download your fully reconstructed model, ready to use!

### Why Is This Useful?

- ✅ Share large models easily through email or messaging
- ✅ Work around file size limits on platforms like GitHub
- ✅ More reliable transfers (if one piece fails, only resend that piece)
- ✅ Perfect for collaborative projects where you need to share models

## Model Files

The trained models are not included in this repository due to file size limitations. You can download them from the following locations:

### House Price Prediction Model
- File: `house_price_model.joblib`
- Size: 138.04 MB
- Storage: Use our [Streamlit App](https://ml-agents-iowu6wwfewhqgeeazyvuci.streamlit.app/) to download the model in chunks
- Usage: After downloading all chunks, use the app or command-line tools to reconstruct the model file
## Installation

1. Clone the repository:
```bash
git clone https://github.com/emil-sincliar/ml-agents.git
cd ml-agents
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the required model files as described above.

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

## How to Use the Model Manager

### 🌐 Using the Web Interface (Recommended for Beginners)

1. **To Split a Large Model:**
   - Go to our [Streamlit App](https://ml-agents-iowu6wwfewhqgeeazyvuci.streamlit.app/)
   - Click on "Model File Manager"
   - Upload your model file (supports .joblib, .pkl, or .bin files)
   - Choose how small you want the pieces to be (up to 90MB each)
   - Click "Split Model" and download your zip file!

2. **To Combine Model Pieces:**
   - Go to the same app and click "Combine Model"
   - Upload the zip file containing your model pieces
   - Click "Combine Chunks"
   - Download your reconstructed model!

### Using Command Line

Split a model:
```bash
python scripts/model_utils.py split path/to/model.joblib
```

Combine chunks:
```bash
python scripts/model_utils.py combine path/to/chunks/directory
```

<<<<<<< HEAD
=======
## Project Structure
```
ml-agents/
├── examples/
│   ├── classification_example.py
│   ├── house_price_prediction.py
│   └── test_scenarios.py
├── ml_agents/
│   └── agents/
│       └── data_scientist_agent.py
├── tests/
│   └── test_data_scientist_agent.py
├── requirements.txt
└── README.md
```

>>>>>>> feature/add-model
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
