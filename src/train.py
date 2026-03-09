import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import os

# ── 1. Load & preprocess ──────────────────────────────────────────
df = pd.read_csv("data/telco_churn.csv")

df = df.drop(columns=['customerID'])
df['TotalCharges'] = df['TotalCharges'].replace(' ', '0')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, columns=df.select_dtypes(include='object').columns.tolist(), drop_first=True)
df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)

X = df.drop(columns=['Churn'])
y = df['Churn']

# Save feature columns for the API later
os.makedirs("models", exist_ok=True)
with open("models/feature_columns.json", "w") as f:
    json.dump(X.columns.tolist(), f)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Define parameters ──────────────────────────────────────────
params = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "scale_pos_weight": round((y_train == 0).sum() / (y_train == 1).sum(), 2),
    "random_state": 42
}

# ── 3. MLflow experiment ──────────────────────────────────────────
mlflow.set_experiment("churn-prediction")

with mlflow.start_run(run_name="lgbm-baseline"):

    # Log parameters
    mlflow.log_params(params)

    # Train
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log metrics
    mlflow.log_metric("roc_auc", round(roc_auc, 4))
    mlflow.log_metric("recall_churned", round(report['1']['recall'], 4))
    mlflow.log_metric("precision_churned", round(report['1']['precision'], 4))
    mlflow.log_metric("f1_churned", round(report['1']['f1-score'], 4))

    from mlflow.models.signature import infer_signature

    signature = infer_signature(X_train, model.predict(X_train))
    # Log model
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=X_train.iloc[:3],
        registered_model_name="churn-model"   # ← this registers it
    )

    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"Recall (churned): {report['1']['recall']:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("✅ Run logged to MLflow!")