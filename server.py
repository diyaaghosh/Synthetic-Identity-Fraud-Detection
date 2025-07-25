
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from ensemble import predict_user_risk

app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    
    df = pd.DataFrame([data])
    try:
        result = predict_user_risk(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
