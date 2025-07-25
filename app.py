"""
Streamlit web application for the House Price Prediction model
"""

import streamlit as st
import pandas as pd
import numpy as np
from ml_agents.agents import DataScientistAgent
import joblib

# Set page configuration
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

def load_agent():
    """Load the trained agent"""
    agent = DataScientistAgent("HousePricePredictor")
    agent.load_model('house_price_model.joblib')
    return agent

def format_price(price):
    """Format price in thousands to a readable format with commas"""
    return f"${price*100:,.2f}"

def main():
    # Add a title and description
    st.title("🏠 California House Price Predictor")
    st.markdown("""
    This application predicts house prices in California based on various features.
    Enter the details below to get a price estimate.
    """)
    
    try:
        agent = load_agent()
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Location and Community Details")
            median_income = st.number_input(
                "Area Median Income (in tens of thousands $)",
                min_value=1.0,
                max_value=15.0,
                value=5.0,
                help="Median income in the area, e.g., 5.0 means $50,000"
            )
            
            population = st.number_input(
                "Block Population",
                min_value=100,
                max_value=10000,
                value=2000,
                help="Population of the block"
            )
            
            avg_occupancy = st.number_input(
                "Average Occupancy",
                min_value=1.0,
                max_value=6.0,
                value=2.8,
                help="Average number of people per household"
            )
            
            latitude = st.slider(
                "Latitude",
                min_value=32.0,
                max_value=42.0,
                value=37.0,
                help="Location latitude (32°N to 42°N covers California)"
            )
            
            longitude = st.slider(
                "Longitude",
                min_value=-124.0,
                max_value=-114.0,
                value=-119.0,
                help="Location longitude (-124°W to -114°W covers California)"
            )
        
        with col2:
            st.subheader("House Details")
            house_age = st.number_input(
                "House Age (years)",
                min_value=0,
                max_value=100,
                value=20,
                help="Age of the house in years"
            )
            
            avg_rooms = st.number_input(
                "Average Rooms",
                min_value=2.0,
                max_value=12.0,
                value=6.0,
                help="Average number of rooms"
            )
            
            avg_bedrooms = st.number_input(
                "Average Bedrooms",
                min_value=1.0,
                max_value=6.0,
                value=3.0,
                help="Average number of bedrooms"
            )
        
        # Add a predict button
        if st.button("Predict Price", type="primary"):
            # Create input data
            input_data = pd.DataFrame({
                'MedInc': [median_income],
                'HouseAge': [house_age],
                'AveRooms': [avg_rooms],
                'AveBedrms': [avg_bedrooms],
                'Population': [population],
                'AveOccup': [avg_occupancy],
                'Latitude': [latitude],
                'Longitude': [longitude]
            })
            
            # Make prediction
            prediction = agent.make_predictions(input_data)[0]
            
            # Display result in a nice box
            st.success(f"### Estimated House Price: {format_price(prediction)}")
            
            # Show feature importance
            if hasattr(agent.model, 'feature_importances_'):
                st.subheader("Feature Importance Analysis")
                features = [
                    'Median Income', 'House Age', 'Average Rooms',
                    'Average Bedrooms', 'Population', 'Average Occupancy',
                    'Latitude', 'Longitude'
                ]
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': agent.model.feature_importances_
                })
                importance_df = importance_df.sort_values('Importance', ascending=True)
                
                # Plot feature importance
                st.bar_chart(importance_df.set_index('Feature'))
            
            # Add some insights
            st.subheader("Price Insights")
            insights = []
            if median_income > 8.0:
                insights.append("📈 The high median income in this area significantly increases the price.")
            elif median_income < 3.0:
                insights.append("📉 The lower median income in this area affects the price negatively.")
                
            if avg_rooms > 8:
                insights.append("🏰 The large number of rooms suggests this is a spacious property.")
            elif avg_rooms < 4:
                insights.append("🏠 This appears to be a more compact property.")
                
            if house_age < 10:
                insights.append("✨ This is a relatively new property.")
            elif house_age > 40:
                insights.append("🏛️ This is an established property in a mature neighborhood.")
                
            for insight in insights:
                st.markdown(insight)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.markdown("""
        ### Troubleshooting
        - Make sure the model file 'house_price_model.joblib' is present
        - Verify that all required dependencies are installed
        - Check if the input values are within expected ranges
        """)

if __name__ == "__main__":
    main()
