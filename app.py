"""
Streamlit web application for ML Agents utilities and House Price Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
from ml_agents.agents import DataScientistAgent
from scripts.model_utils import split_model, combine_model
import joblib
import os
import tempfile
import shutil
import zipfile

# Initialize session state for model training
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

@st.cache_resource
def load_or_train_model():
    """Load or train the model with caching"""
    try:
        # Try to load the model
        agent = DataScientistAgent("HousePricePredictor")
        if os.path.exists('house_price_model.joblib'):
            agent.load_model('house_price_model.joblib')
            return agent
        else:
            # Train new model if it doesn't exist
            import examples.house_price_prediction as trainer
            trainer.main()
            agent.load_model('house_price_model.joblib')
            return agent
    except Exception as e:
        st.error(f"Error loading/training model: {str(e)}")
        return None

# Set page configuration
st.set_page_config(
    page_title="ML Agents Toolkit",
    page_icon="🤖",
    layout="wide"
)

def load_agent():
    """Load the trained agent"""
    agent = DataScientistAgent("HousePricePredictor")
    agent.load_model('house_price_model.joblib')
    return agent

def model_splitter_app():
    """Model splitting and combining interface"""
    st.title("🔄 Model File Manager")
    st.markdown("""
    This tool helps you manage large model files by splitting them into smaller chunks 
    and combining them back together. Useful for sharing models that exceed size limits.
    """)
    
    tab1, tab2 = st.tabs(["Split Model", "Combine Model"])
    
    with tab1:
        st.header("Split Large Model File")
        uploaded_model = st.file_uploader("Choose a model file to split", type=['joblib', 'pkl', 'bin'])
        chunk_size = st.number_input("Chunk size (MB)", min_value=1, max_value=90, value=90)
        
        if uploaded_model is not None:
            if st.button("Split Model"):
                with st.spinner("Splitting model file..."):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_model.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Create a temporary directory for chunks
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            try:
                                # Split the model
                                total_chunks = split_model(tmp_path, chunk_size_mb=chunk_size, output_dir=tmp_dir)
                                
                                # Create a zip file of all chunks
                                zip_path = "model_chunks.zip"
                                shutil.make_archive("model_chunks", 'zip', tmp_dir)
                                
                                # Offer the zip file for download
                                with open(zip_path, "rb") as fp:
                                    st.success(f"Model split into {total_chunks} chunks successfully!")
                                    st.download_button(
                                        label="Download Split Model Chunks",
                                        data=fp,
                                        file_name="model_chunks.zip",
                                        mime="application/zip"
                                    )
                            finally:
                                # Clean up zip file if it was created
                                if 'zip_path' in locals() and os.path.exists(zip_path):
                                    os.remove(zip_path)
                    except Exception as e:
                        st.error(f"Error processing model: {str(e)}")
                    finally:
                        os.unlink(tmp_path)
    
    with tab2:
        st.header("Combine Model Chunks")
        uploaded_chunks = st.file_uploader("Upload ZIP file containing model chunks", type=['zip'])
        
        if uploaded_chunks is not None:
            if st.button("Combine Chunks"):
                with st.spinner("Combining model chunks..."):
                    # Save the uploaded zip file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                        tmp_zip.write(uploaded_chunks.getvalue())
                        zip_path = tmp_zip.name
                    
                    try:
                        # Create a temporary directory and extract zip
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            try:
                                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                    zip_ref.extractall(tmp_dir)
                                
                                # Combine the chunks
                                output_path = os.path.join(tmp_dir, "reconstructed_model.joblib")
                                combine_model(tmp_dir, output_path)
                                
                                # Offer the combined model for download
                                with open(output_path, "rb") as fp:
                                    st.success("Model chunks combined successfully!")
                                    st.download_button(
                                        label="Download Combined Model",
                                        data=fp,
                                        file_name="reconstructed_model.joblib",
                                        mime="application/octet-stream"
                                    )
                            except Exception as e:
                                st.error(f"Error combining chunks: {str(e)}")
                    finally:
                        os.unlink(zip_path)

def format_price(price):
    """Format price in thousands to a readable format with commas"""
    return f"${price*100:,.2f}"

def house_price_predictor():
    """House price prediction interface"""
    st.title("🏠 California House Price Predictor")
    st.markdown("""
    This application predicts house prices in California based on various features.
    Enter the details below to get a price estimate.
    """)
    
    with st.spinner('Loading model... This may take a moment on first run...'):
        agent = load_or_train_model()
    
    if agent is None:
        st.error("Could not initialize the model. Please try refreshing the page.")
        return

def main():
    """Main application with navigation"""
    st.sidebar.title("🤖 ML Agents Toolkit")
    
    # Navigation
    page = st.sidebar.radio(
        "Select Tool",
        ["Model File Manager", "House Price Predictor"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This toolkit provides:
    - Model file splitting and combining
    - House price prediction model
    """)
    
    # Display selected page
    if page == "Model File Manager":
        model_splitter_app()
    else:
        house_price_predictor()
    
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
