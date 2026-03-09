import json
import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, request, jsonify

# ── App setup ────────────────────────────────────────────────────
app = Flask(__name__)

# ── Load model & features at startup ─────────────────────────────
MODEL_NAME = "churn-model"
MODEL_VERSION = "1"

print(f"Loading model: {MODEL_NAME} v{MODEL_VERSION}...")
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

with open("models/feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

print(f"Model loaded! Expecting {len(FEATURE_COLUMNS)} features.")


# ── Helper: preprocess input ──────────────────────────────────────
def preprocess(data: dict) -> pd.DataFrame:
    raw = pd.DataFrame([data])

    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    raw = pd.get_dummies(raw, columns=cat_cols, drop_first=True)
    raw['charges_per_tenure'] = raw['TotalCharges'] / (raw['tenure'] + 1)
    raw = raw.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return raw


# ── Routes ────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Churn Prediction API is running!"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": MODEL_NAME, "version": MODEL_VERSION})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        X = preprocess(data)

        churn_prob = model.predict_proba(X)[0][1]
        churn_pred = bool(churn_prob >= 0.5)

        return jsonify({
            "churn": churn_pred,
            "churn_probability": round(float(churn_prob), 4),
            "risk_level": "High" if churn_prob >= 0.7 else "Medium" if churn_prob >= 0.4 else "Low"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)