import pandas as pd
from ensemble import predict_user_risk

# Load data with identifiers
df = pd.read_csv("Dataset/Base_with_graph_features.csv")

# Drop unwanted columns but keep identifier + useful ones
df = df.drop(columns=['employment_status', 'housing_status', 'payment_type', 'fraud_bool'], errors='ignore')

# Ensure required fields for graph exist
required_fields = ['device_os', 'source', 'email', 'phone_number', 'device_id', 'ip_address']
df = df.dropna(subset=required_fields)

# Pick a sample with full identifier and feature info
sample_raw = df.iloc[[0]]

# Predict fraud risk
result = predict_user_risk(sample_raw)

# Print result
print("Prediction Result:")
for key, value in result.items():
    print(f"{key}: {value}")
