"""
Testing the house price prediction model with various scenarios
"""

import pandas as pd
from ml_agents.agents import DataScientistAgent
import joblib

def format_price(price):
    """Format price in thousands to a readable format with commas"""
    return f"${price*100:,.2f}k"

def main():
    # Load the trained agent
    agent = DataScientistAgent("HousePricePredictor")
    agent.load_model('house_price_model.joblib')
    
    # Test Scenario 1: Luxury Houses in Different Areas
    print("\nScenario 1: Luxury Houses in Different Areas")
    print("-------------------------------------------")
    luxury_houses = pd.DataFrame({
        'MedInc': [12.5, 12.5, 12.5],  # High income areas
        'HouseAge': [5, 15, 30],        # New to established
        'AveRooms': [8, 8, 8],          # Large houses
        'AveBedrms': [4, 4, 4],         # Many bedrooms
        'Population': [1500, 1500, 1500],
        'AveOccup': [2.5, 2.5, 2.5],
        'Latitude': [37.85, 37.75, 37.65],  # Different Bay Area locations
        'Longitude': [-122.25, -122.45, -122.15]
    })
    predictions = agent.make_predictions(luxury_houses)
    print("\nPredicted prices for luxury houses in different locations:")
    for i, price in enumerate(predictions):
        area = ["San Francisco Bay", "Silicon Valley", "East Bay"][i]
        print(f"Location: {area} - {format_price(price)}")

    # Test Scenario 2: Starter Homes
    print("\nScenario 2: Starter Homes")
    print("-------------------------")
    starter_homes = pd.DataFrame({
        'MedInc': [4.0, 3.5, 3.0],      # Lower-middle income areas
        'HouseAge': [40, 35, 45],        # Older homes
        'AveRooms': [5, 4, 4],           # Smaller houses
        'AveBedrms': [2, 2, 2],          # Fewer bedrooms
        'Population': [2500, 3000, 2800],
        'AveOccup': [3.0, 3.2, 3.1],
        'Latitude': [34.0, 34.1, 34.2],  # Different LA area locations
        'Longitude': [-118.2, -118.3, -118.4]
    })
    predictions = agent.make_predictions(starter_homes)
    print("\nPredicted prices for starter homes:")
    for i, price in enumerate(predictions):
        condition = ["Good condition", "Needs minor repairs", "Fixer-upper"][i]
        print(f"Home type: {condition} - {format_price(price)}")

    # Test Scenario 3: Impact of Income Level
    print("\nScenario 3: Same House, Different Income Areas")
    print("--------------------------------------------")
    income_test = pd.DataFrame({
        'MedInc': [2.0, 5.0, 8.0, 12.0],  # Various income levels
        'HouseAge': [25, 25, 25, 25],      # Same house age
        'AveRooms': [6, 6, 6, 6],          # Same size
        'AveBedrms': [3, 3, 3, 3],         # Same bedrooms
        'Population': [2000, 2000, 2000, 2000],
        'AveOccup': [2.8, 2.8, 2.8, 2.8],
        'Latitude': [36.0, 36.0, 36.0, 36.0],  # Same location
        'Longitude': [-119.0, -119.0, -119.0, -119.0]
    })
    predictions = agent.make_predictions(income_test)
    print("\nPredicted prices based on area median income:")
    for inc, price in zip([2.0, 5.0, 8.0, 12.0], predictions):
        print(f"Area Median Income: ${inc*10:,.0f}k - House Price: {format_price(price)}")

    # Test Scenario 4: Age Impact
    print("\nScenario 4: Age Impact on Similar Houses")
    print("---------------------------------------")
    age_test = pd.DataFrame({
        'MedInc': [6.0, 6.0, 6.0, 6.0],    # Same income area
        'HouseAge': [5, 15, 30, 50],        # Different ages
        'AveRooms': [6, 6, 6, 6],           # Same size
        'AveBedrms': [3, 3, 3, 3],          # Same bedrooms
        'Population': [2000, 2000, 2000, 2000],
        'AveOccup': [2.8, 2.8, 2.8, 2.8],
        'Latitude': [35.5, 35.5, 35.5, 35.5],  # Same location
        'Longitude': [-119.5, -119.5, -119.5, -119.5]
    })
    predictions = agent.make_predictions(age_test)
    print("\nPredicted prices based on house age:")
    for age, price in zip([5, 15, 30, 50], predictions):
        print(f"House Age: {age} years - {format_price(price)}")

    # Test Scenario 5: Size Impact
    print("\nScenario 5: Size Impact")
    print("----------------------")
    size_test = pd.DataFrame({
        'MedInc': [7.0, 7.0, 7.0, 7.0],      # Same income area
        'HouseAge': [20, 20, 20, 20],         # Same age
        'AveRooms': [4, 6, 8, 10],            # Different sizes
        'AveBedrms': [2, 3, 4, 5],            # Proportional bedrooms
        'Population': [2000, 2000, 2000, 2000],
        'AveOccup': [2.8, 2.8, 2.8, 2.8],
        'Latitude': [35.0, 35.0, 35.0, 35.0], # Same location
        'Longitude': [-118.5, -118.5, -118.5, -118.5]
    })
    predictions = agent.make_predictions(size_test)
    print("\nPredicted prices based on house size:")
    for rooms, beds, price in zip([4, 6, 8, 10], [2, 3, 4, 5], predictions):
        print(f"Rooms: {rooms}, Bedrooms: {beds} - {format_price(price)}")

if __name__ == "__main__":
    main()
