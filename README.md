# Churn Prediction MLOps Pipeline

An end-to-end ML pipeline that predicts customer churn using LightGBM, tracked with MLflow, served via Flask, containerised with Docker, and deployed on AWS.

## Tech Stack
- **Model**: LightGBM (ROC-AUC: 0.84)
- **Experiment Tracking**: MLflow
- **API**: Flask
- **Containerisation**: Docker
- **Cloud**: AWS

## Project Structure
```
churn-mlops/
├── src/
│   ├── train.py      # Model training + MLflow logging
│   └── app.py        # Flask prediction API
├── data/             # Raw data (gitignored)
├── models/           # Saved artifacts (gitignored)
├── notebooks/        # EDA notebook
├── Dockerfile
└── requirements.txt
```

## Quickstart
```bash
# Train model
python src/train.py

# Run API
python src/app.py

# Test prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 85.0, ...}'
```

## API Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/health` | GET | Model status |
| `/predict` | POST | Predict churn |
