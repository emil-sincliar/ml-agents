# California House Price Predictor - Deployment Guide

## Overview
This guide explains how to deploy and run the California House Price Predictor web application.

## Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. First, clone the repository:
```bash
git clone https://github.com/neerajchavan/ml-agents.git
cd ml-agents
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -e .
pip install streamlit
```

## Running the Application

1. Make sure you have trained the model first:
```bash
python examples/house_price_prediction.py
```

2. Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501

## Usage

The web interface allows users to:
- Input house and location details
- Get price predictions
- View feature importance analysis
- Receive insights about the prediction

## Features

- Interactive input fields for all house features
- Real-time price predictions
- Visual feature importance analysis
- Automated insights based on input values
- Responsive design that works on both desktop and mobile

## Deployment Options

### Local Deployment
- Follow the installation and running instructions above
- Suitable for development and testing

### Cloud Deployment (Streamlit Cloud)
1. Push your code to GitHub
2. Visit https://share.streamlit.io/
3. Connect your GitHub repository
4. Select app.py as the main file
5. Deploy

### Docker Deployment
1. Build the Docker image:
```bash
docker build -t house-price-predictor .
```

2. Run the container:
```bash
docker run -p 8501:8501 house-price-predictor
```

## Maintenance

- Regularly update the model with new data
- Monitor prediction accuracy
- Keep dependencies updated
- Backup the model file regularly

## Support

For any issues or questions:
- Create an issue in the GitHub repository
- Contact the development team

## Security Notes

- Ensure input validation is maintained
- Regularly update dependencies
- Back up the model file
- Monitor system resources

## Future Improvements

- Add user authentication
- Implement API endpoints
- Add more visualization options
- Include historical price trends
- Add batch prediction capabilities
