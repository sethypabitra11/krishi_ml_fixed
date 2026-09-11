import os
import pickle
from typing import Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# Project paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# --------------------------------------------------
# Load trained models once when the Render service starts
# --------------------------------------------------
def load_pickle(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


price_model = load_pickle("price_model.pkl")
demand_model = load_pickle("demand_model.pkl")
location_encoder = load_pickle("location_encoder.pkl")
product_encoder = load_pickle("product_encoder.pkl")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------
app = FastAPI(
    title="Krishi Setu ML API",
    description="Agricultural price, demand and recommendation service",
    version="1.1.0",
)

# Krishi Setu is calling this API directly from the Vercel/Expo web client.
# Wildcard origins are intentional here because the frontend domain can change.
# Credentials are NOT used by the ML API, so keep allow_credentials=False.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request models - these exactly match Krishi Setu
# --------------------------------------------------
class PredictionRequest(BaseModel):
    location: str = Field(min_length=1)
    product: str = Field(min_length=1)
    market_supply_kg: float = Field(ge=0)
    festival_season: int = Field(ge=0, le=1)


class RecommendationRequest(BaseModel):
    location: str = Field(min_length=1)
    festival_season: int = Field(ge=0, le=1)
    product_supplies: Dict[str, float]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def validate_location(location: str):
    if location not in location_encoder.classes_:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown location: {location}. Use one of: {list(location_encoder.classes_)}",
        )


def validate_product(product: str):
    if product not in product_encoder.classes_:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown product: {product}. Use one of: {list(product_encoder.classes_)}",
        )


def prepare_input(request: PredictionRequest):
    validate_location(request.location)
    validate_product(request.product)

    location_enc = location_encoder.transform([request.location])[0]
    product_enc = product_encoder.transform([request.product])[0]

    return pd.DataFrame(
        {
            "Location_Enc": [location_enc],
            "Product_Enc": [product_enc],
            "Market_Supply_kg": [request.market_supply_kg],
            "Festival_Season": [request.festival_season],
        }
    )


# --------------------------------------------------
# Health/root endpoints
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Krishi Setu ML API is running",
        "status": "ok",
        "endpoints": [
            "/predict-price",
            "/predict-demand",
            "/recommend-products",
        ],
    }


@app.get("/health")
def health():
    # Lightweight endpoint for Render health checks and manual wake-up checks.
    return {"status": "healthy"}


# --------------------------------------------------
# Price prediction - used by Krishi Setu
# --------------------------------------------------
@app.post("/predict-price")
def predict_price(request: PredictionRequest):
    X = prepare_input(request)
    predicted_price = price_model.predict(X)[0]

    return {
        "location": request.location,
        "product": request.product,
        "expected_price_per_kg": round(float(predicted_price), 2),
    }


# --------------------------------------------------
# Demand prediction - used by Krishi Setu
# --------------------------------------------------
@app.post("/predict-demand")
def predict_demand(request: PredictionRequest):
    X = prepare_input(request)
    predicted_demand = demand_model.predict(X)[0]

    return {
        "location": request.location,
        "product": request.product,
        "expected_demand_kg": round(float(predicted_demand), 2),
    }


# --------------------------------------------------
# Product recommendation - used by Krishi Setu
# --------------------------------------------------
@app.post("/recommend-products")
def recommend_products(request: RecommendationRequest):
    validate_location(request.location)

    for product in request.product_supplies:
        validate_product(product)

    products = list(product_encoder.classes_)

    scenarios = pd.DataFrame(
        {
            "Location": [request.location] * len(products),
            "Product": products,
            "Market_Supply_kg": [
                max(0.0, float(request.product_supplies.get(product, 0)))
                for product in products
            ],
            "Festival_Season": [request.festival_season] * len(products),
        }
    )

    scenarios["Location_Enc"] = location_encoder.transform(scenarios["Location"])
    scenarios["Product_Enc"] = product_encoder.transform(scenarios["Product"])

    features = [
        "Location_Enc",
        "Product_Enc",
        "Market_Supply_kg",
        "Festival_Season",
    ]

    X = scenarios[features]

    scenarios["Expected_Price"] = price_model.predict(X)
    scenarios["Expected_Demand"] = demand_model.predict(X)
    scenarios["Expected_Revenue"] = (
        scenarios["Expected_Price"] * scenarios["Expected_Demand"]
    )

    top_products = scenarios.sort_values(
        by="Expected_Revenue", ascending=False
    ).head(3)

    recommendations = []
    for _, row in top_products.iterrows():
        recommendations.append(
            {
                "product": row["Product"],
                "market_supply_kg": round(float(row["Market_Supply_kg"]), 2),
                "expected_price_per_kg": round(float(row["Expected_Price"]), 2),
                "expected_demand_kg": round(float(row["Expected_Demand"]), 2),
                "expected_revenue": round(float(row["Expected_Revenue"]), 2),
            }
        )

    return {
        "location": request.location,
        "festival_season": request.festival_season,
        "recommendations": recommendations,
    }
