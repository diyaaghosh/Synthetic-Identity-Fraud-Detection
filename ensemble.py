import joblib
import numpy as np
import pandas as pd
from rules import apply_rules
from graph_feature import load_graph, extract_graph_features_for_row

# Load models and scalers
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")
rf_model = joblib.load("models/rf_model.pkl")
iso_model = joblib.load("models/iso_model.pkl")

# Load graph
G = load_graph("models/user_graph.gpickle")

# Load feature order
with open("feature_order.txt") as f:
    feature_order = f.read().splitlines()

# Weights for ensemble
WEIGHTS = {
    "rf": 0.5,
    "iso": 0.1,
    "rules": 0.2,
    "graph": 0.2  # new graph weight
}
THRESHOLD = 0.35

cat = ["device_os", "source"]
num = [col for col in feature_order if col not in encoder.get_feature_names_out(cat)]

def predict_user_risk(user_row_df):
    """
    Predicts fraud risk for a single user row (raw input) with graph support.
    Returns: dict with score, fraud label, reasons.
    """
    # Encode categorical
    encoded_cat = encoder.transform(user_row_df[cat])
    encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat))

    # Combine numerical + encoded
    combined = pd.concat([user_row_df[num].reset_index(drop=True), encoded_cat_df.reset_index(drop=True)], axis=1)

    # Fill missing graph columns if absent
    graph_feats = extract_graph_features_for_row(G, user_row_df.iloc[0])
    for key, value in graph_feats.items():
        combined[key] = value

    # Reorder and scale
    combined = combined[feature_order]
    X = scaler.transform(combined)
    X = pd.DataFrame(X, columns=feature_order)

    # Sub-model predictions
    rf_score = rf_model.predict_proba(X)[0][1]
    iso_raw = iso_model.decision_function(X)[0]
    iso_score = 1 - (iso_raw - iso_model.offset_)
    rule_score, reasons = apply_rules(user_row_df.iloc[0])

    # Graph score: simple heuristic for now (e.g., based on fraud neighbors)
    graph_score = min(1.0, graph_feats["fraud_ratio_neighbors"] + graph_feats["fraud_neighbors"] * 0.1)

    # Final ensemble score
    final_score = (
        WEIGHTS["rf"] * rf_score +
        WEIGHTS["iso"] * iso_score +
        WEIGHTS["rules"] * rule_score +
        WEIGHTS["graph"] * graph_score
    )

    is_fraud = int(final_score >= THRESHOLD)

    return {
        "ensemble_score": round(final_score, 3),
        "is_fraud": int(final_score >= THRESHOLD),
        "rf_score": round(rf_score, 3),
        "iso_score": round(iso_score, 3),
        "rule_score": round(rule_score, 3),
        "graph_score": round(graph_score, 3),
        "reasons": reasons
    }
