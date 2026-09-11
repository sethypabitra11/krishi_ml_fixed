import os
import pickle
import pandas as pd


# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# -----------------------------
# Load saved models
# -----------------------------

with open(os.path.join(MODEL_DIR, "price_model.pkl"), "rb") as f:
    price_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "demand_model.pkl"), "rb") as f:
    demand_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "location_encoder.pkl"), "rb") as f:
    location_encoder = pickle.load(f)

with open(os.path.join(MODEL_DIR, "product_encoder.pkl"), "rb") as f:
    product_encoder = pickle.load(f)


print("Models loaded successfully!")


# -----------------------------
# Test input
# -----------------------------

location = "Unit-1 Market"
product = "Tomato"
market_supply = 1000
festival_season = 0


# Encode categorical values
location_enc = location_encoder.transform([location])[0]
product_enc = product_encoder.transform([product])[0]


# Create input DataFrame
X_test = pd.DataFrame({
    "Location_Enc": [location_enc],
    "Product_Enc": [product_enc],
    "Market_Supply_kg": [market_supply],
    "Festival_Season": [festival_season]
})


# -----------------------------
# Predictions
# -----------------------------

predicted_price = price_model.predict(X_test)[0]
predicted_demand = demand_model.predict(X_test)[0]

expected_revenue = predicted_price * predicted_demand


# -----------------------------
# Display results
# -----------------------------

print("\nPrediction Test")
print("----------------------------")
print(f"Location: {location}")
print(f"Product: {product}")
print(f"Market Supply: {market_supply} kg")
print(f"Festival Season: {festival_season}")

print("\nResults")
print("----------------------------")
print(f"Expected Price: ₹{predicted_price:.2f}/kg")
print(f"Expected Demand: {predicted_demand:.2f} kg")
print(f"Expected Revenue: ₹{expected_revenue:.2f}")