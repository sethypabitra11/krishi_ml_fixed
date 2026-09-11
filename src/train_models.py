import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "bhubaneswar_market_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. BASIC DATA CHECK
# ============================================================

required_columns = [
    "Date",
    "Location",
    "Product",
    "Market_Supply_kg",
    "Festival_Season",
    "Selling_Price_per_kg",
    "Quantity_Sold_kg"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

print("\nDataset validation: OK")


# ============================================================
# 4. ENCODE CATEGORICAL FEATURES
# ============================================================

location_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df["Location_Enc"] = location_encoder.fit_transform(
    df["Location"]
)

df["Product_Enc"] = product_encoder.fit_transform(
    df["Product"]
)


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

features = [
    "Location_Enc",
    "Product_Enc",
    "Market_Supply_kg",
    "Festival_Season"
]

X = df[features]

# Price target
y_price = df["Selling_Price_per_kg"]

# Demand target
y_demand = df["Quantity_Sold_kg"]


print("\nFeatures:")
print(features)

print("\nNumber of products:",
      len(product_encoder.classes_))

print("Products:")
print(list(product_encoder.classes_))

print("\nNumber of locations:",
      len(location_encoder.classes_))

print("Locations:")
print(list(location_encoder.classes_))


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train_price, X_test_price, y_train_price, y_test_price = (
    train_test_split(
        X,
        y_price,
        test_size=0.20,
        random_state=42
    )
)

X_train_demand, X_test_demand, y_train_demand, y_test_demand = (
    train_test_split(
        X,
        y_demand,
        test_size=0.20,
        random_state=42
    )
)


# ============================================================
# 7. PRICE MODEL
# ============================================================

print("\nTraining Price Model...")

price_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

price_model.fit(
    X_train_price,
    y_train_price
)

print("Price model training completed.")


# ============================================================
# 8. DEMAND MODEL
# ============================================================

print("\nTraining Demand Model...")

demand_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

demand_model.fit(
    X_train_demand,
    y_train_demand
)

print("Demand model training completed.")


# ============================================================
# 9. PRICE MODEL EVALUATION
# ============================================================

price_predictions = price_model.predict(X_test_price)

price_mae = mean_absolute_error(
    y_test_price,
    price_predictions
)

price_rmse = np.sqrt(
    mean_squared_error(
        y_test_price,
        price_predictions
    )
)

price_r2 = r2_score(
    y_test_price,
    price_predictions
)


print("\n==============================")
print("PRICE MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {price_mae:.2f}")
print(f"RMSE : {price_rmse:.2f}")
print(f"R2   : {price_r2:.2f}")


# ============================================================
# 10. DEMAND MODEL EVALUATION
# ============================================================

demand_predictions = demand_model.predict(
    X_test_demand
)

demand_mae = mean_absolute_error(
    y_test_demand,
    demand_predictions
)

demand_rmse = np.sqrt(
    mean_squared_error(
        y_test_demand,
        demand_predictions
    )
)

demand_r2 = r2_score(
    y_test_demand,
    demand_predictions
)


print("\n==============================")
print("DEMAND MODEL PERFORMANCE")
print("==============================")

print(f"MAE  : {demand_mae:.2f}")
print(f"RMSE : {demand_rmse:.2f}")
print(f"R2   : {demand_r2:.2f}")


# ============================================================
# 11. SAVE PRICE MODEL
# ============================================================

price_model_path = os.path.join(
    MODEL_DIR,
    "price_model.pkl"
)

with open(price_model_path, "wb") as file:
    pickle.dump(price_model, file)


# ============================================================
# 12. SAVE DEMAND MODEL
# ============================================================

demand_model_path = os.path.join(
    MODEL_DIR,
    "demand_model.pkl"
)

with open(demand_model_path, "wb") as file:
    pickle.dump(demand_model, file)


# ============================================================
# 13. SAVE LOCATION ENCODER
# ============================================================

location_encoder_path = os.path.join(
    MODEL_DIR,
    "location_encoder.pkl"
)

with open(location_encoder_path, "wb") as file:
    pickle.dump(location_encoder, file)


# ============================================================
# 14. SAVE PRODUCT ENCODER
# ============================================================

product_encoder_path = os.path.join(
    MODEL_DIR,
    "product_encoder.pkl"
)

with open(product_encoder_path, "wb") as file:
    pickle.dump(product_encoder, file)


print("\n==============================")
print("MODELS SAVED")
print("==============================")

print(price_model_path)
print(demand_model_path)
print(location_encoder_path)
print(product_encoder_path)


# ============================================================
# 15. TEST PRODUCT RECOMMENDATION
# ============================================================

print("\n==============================")
print("PRODUCT RECOMMENDATION TEST")
print("==============================")


# Use the first location from the dataset
test_location = df["Location"].iloc[0]

future_products = list(
    product_encoder.classes_
)


# ------------------------------------------------------------
# Create test scenarios
# ------------------------------------------------------------

test_scenarios = pd.DataFrame({
    "Location": [test_location] * len(future_products),
    "Product": future_products,

    # Example market supply values
    "Market_Supply_kg": [
        1200,
        2800,
        2000,
        900,
        700,
        1000,
        400
    ],

    # 0 = normal season
    # 1 = festival season
    "Festival_Season": [0] * len(future_products)
})


# ============================================================
# 16. ENCODE TEST DATA
# ============================================================

test_scenarios["Location_Enc"] = (
    location_encoder.transform(
        test_scenarios["Location"]
    )
)

test_scenarios["Product_Enc"] = (
    product_encoder.transform(
        test_scenarios["Product"]
    )
)


X_future = test_scenarios[features]


# ============================================================
# 17. PREDICT PRICE
# ============================================================

test_scenarios["Expected_Selling_Price"] = (
    price_model.predict(X_future)
)


# ============================================================
# 18. PREDICT DEMAND
# ============================================================

test_scenarios["Expected_Quantity"] = (
    demand_model.predict(X_future)
)


# ============================================================
# 19. CALCULATE EXPECTED REVENUE
# ============================================================

test_scenarios["Total_Expected_Revenue"] = (
    test_scenarios["Expected_Selling_Price"]
    *
    test_scenarios["Expected_Quantity"]
)


# ============================================================
# 20. CALCULATE MODEL CONSISTENCY SCORE
# ============================================================

tree_predictions = np.array([
    tree.predict(X_future.to_numpy())
    for tree in price_model.estimators_
])

std_dev = tree_predictions.std(axis=0)

mean_pred = tree_predictions.mean(axis=0)

# Avoid division by zero
mean_pred = np.maximum(mean_pred, 0.000001)

confidence_scores = np.maximum(
    0,
    100 - (std_dev / mean_pred) * 200
)

test_scenarios["Confidence_Score_%"] = (
    confidence_scores.round(2)
)


# ============================================================
# 21. TOP 3 RECOMMENDATIONS
# ============================================================

top_3 = (
    test_scenarios
    .sort_values(
        by="Total_Expected_Revenue",
        ascending=False
    )
    .head(3)
)


# ============================================================
# 22. DISPLAY RESULTS
# ============================================================

print(
    f"\nTest Location: {test_location}"
)

print("\nTOP 3 PRODUCTS TO RECOMMEND:")

for rank, (_, row) in enumerate(
    top_3.iterrows(),
    start=1
):

    print(f"\n#{rank} {row['Product']}")

    print(
        f"  Expected Price: "
        f"₹{row['Expected_Selling_Price']:.2f}/kg"
    )

    print(
        f"  Expected Quantity: "
        f"{int(row['Expected_Quantity'])} kg"
    )

    print(
        f"  Expected Revenue: "
        f"₹{row['Total_Expected_Revenue']:.2f}"
    )

    print(
        f"  Model Consistency: "
        f"{row['Confidence_Score_%']}%"
    )


print("\nTraining and testing completed successfully.")