from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import joblib
import numpy as np
from rules import apply_rules
import os

# Load models and tools
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")
rf_model = joblib.load("models/rf_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")

# Load graph features if available
GRAPH_FEATURES_PATH = "Dataset/graph_features.csv"
graph_features_df = pd.read_csv(GRAPH_FEATURES_PATH) if os.path.exists(GRAPH_FEATURES_PATH) else None

# Feature order
with open("feature_order.txt") as f:
    feature_order = f.read().splitlines()

# FastAPI app
app = FastAPI()

# Input schema
class UserRequest(BaseModel):
    data: Dict[str, Any]

# Config
WEIGHTS = {
    "rf": 0.6,
    "iso": 0.1,
    "rules": 0.3
}
THRESHOLD = 0.35
cat_cols = ["device_os", "source"]
num_cols = [col for col in feature_order if col not in encoder.get_feature_names_out(cat_cols)]

@app.post("/predict")
def predict(request: UserRequest):
    try:
        # Parse input
        user_row_df = pd.DataFrame([request.data])
        user_id = request.data.get("email", None)  # or another identifier

        # Encode categorical
        encoded_cat = encoder.transform(user_row_df[cat_cols])
        encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))

        # Combine and scale
        combined = pd.concat([user_row_df[num_cols].reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)
        combined = combined[feature_order]
        X = pd.DataFrame(scaler.transform(combined), columns=feature_order)

        # Predict scores
        rf_score = rf_model.predict_proba(X)[0][1]
        iso_score = 1 - (iso_model.decision_function(X)[0] - iso_model.offset_)
        rule_score, reasons = apply_rules(user_row_df.iloc[0])

        # Ensemble score
        ensemble_score = (
            WEIGHTS["rf"] * rf_score +
            WEIGHTS["iso"] * iso_score +
            WEIGHTS["rules"] * rule_score
        )

        is_fraud = int(ensemble_score >= THRESHOLD)

        result = {
            "ensemble_score": round(ensemble_score, 3),
            "is_fraud": is_fraud,
            "rule_based_reasons": reasons
        }

        # Append graph features if available
        if graph_features_df is not None and user_id is not None:
            match = graph_features_df[graph_features_df["user_id"] == user_id]
            if not match.empty:
                result["graph_features"] = match.iloc[0].to_dict()

        return result

    except Exception as e:
        return {"error": str(e)}
