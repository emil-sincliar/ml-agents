# ML Agents

A collection of intelligent agents for various data science and machine learning tasks.

## Model Manager: Your Tool for Handling Large Models

Our user-friendly Model Manager helps you share large machine learning models easily! Try it here:
[ML Agents Utility App](https://ml-agents-yjftpngbooy2vgoappydwpe.streamlit.app/)

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
- Storage: Use our [Streamlit App](https://ml-agents-yjftpngbooy2vgoappydwpe.streamlit.app/) to download the model in chunks
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

## Our Tools Explained

### 1. House Price Predictor 🏠

Ever wondered how much a house in California might cost? Our House Price Predictor helps you figure it out! Here's how it works:

#### What It Does
- Takes information about a house and its location
- Uses smart technology (machine learning) to estimate the price
- Explains why it thinks the house would cost that much

#### How to Use It
1. **Enter Location Details:**
   - Area income level (helps understand neighborhood value)
   - How many people live in the area
   - Exact location in California (using latitude and longitude sliders)

2. **Describe the House:**
   - How old the house is
   - Number of rooms and bedrooms
   - How many people typically live in houses in that area

3. **Get Your Estimate:**
   - Click "Predict Price"
   - See the estimated price
   - Get a smart analysis of what makes this price high or low
   - See which features matter most for the price

#### Cool Features
- 📊 Shows which factors affect the price the most
- 💡 Gives insights about the property and location
- 🎯 Uses real California housing data for accuracy

### 2. Model Manager 🔄

#### What It Does
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
## Understanding Your House Price Prediction

### 📊 What the Numbers Mean

1. **Area Median Income**
   - Input as tens of thousands (e.g., 5.0 means $50,000)
   - Higher income areas usually mean higher house prices
   - This is often the most important factor!

2. **House Age**
   - Newer isn't always better
   - Very new houses (< 10 years) might be more expensive
   - Historic homes (> 40 years) might have special value

3. **Rooms and Space**
   - More rooms usually mean higher prices
   - The app looks at both total rooms and bedrooms
   - Balance between rooms matters (e.g., 3 bedrooms in 6 rooms is typical)

4. **Location Factors**
   - California covers a large area (32°N to 42°N latitude)
   - Coastal areas (around -124°W to -114°W longitude) often cost more
   - Population density affects prices

### 💡 Understanding the Insights

The app gives you helpful insights like:
- Whether the area's income level is boosting or lowering the price
- If the house is unusually spacious or compact
- Whether the house age is a positive or negative factor

## Behind the Scenes: How It Works 🔬

### The Data 📊
Our house price predictor learns from the famous California Housing Dataset, which includes:
- Information about 20,000+ houses across California
- Real data collected from the 1990 California census
- Carefully cleaned and verified information

### What We Know About Each House 🏠
The dataset tells us:
1. **MedInc**: Typical income in the area (in tens of thousands of dollars)
2. **HouseAge**: How old the house is (in years)
3. **AveRooms**: Average number of rooms
4. **AveBedrms**: Average number of bedrooms
5. **Population**: Number of people in the area
6. **AveOccup**: Average number of people in each household
7. **Latitude**: North-South position in California
8. **Longitude**: East-West position in California
9. **Price**: The house price (what we're trying to predict)

### Our Smart Prediction System 🧠

We use a technique called "Random Forest" to predict house prices. Think of it like this:
- It's like having 25 expert real estate agents
- Each expert looks at the house details in a slightly different way
- They all make their best guess about the price
- The final price is the average of all their guesses

#### Why Random Forest?
- It's really good at understanding complex patterns
- It can tell us which features matter most
- It's less likely to make wild guesses
- It can explain its decisions (through feature importance)

### How Accurate Is It? 📏

Our model is tested on houses it hasn't seen before, and we measure its performance in several ways:
1. **R² Score**: Shows how well our predictions match reality (closer to 1 is better)
2. **Mean Absolute Error**: The average difference between predicted and actual prices
3. **Root Mean Squared Error**: Similar to MAE, but penalizes big mistakes more

The model has shown strong performance with:
- R² scores typically above 0.80 (meaning it explains 80%+ of price variations)
- Predictions usually within 15% of actual prices
- Better accuracy in areas with more similar houses

### Tips for Getting Better Predictions 💫
1. Be as accurate as possible with the location (latitude/longitude)
2. Use reliable area income data
3. Be precise about the number of rooms and bedrooms
4. Consider seasonal trends (prices can vary by season)

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
