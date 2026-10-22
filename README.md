# Crypto Volatility Prediction API

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikit-learn)
![Render](https://img.shields.io/badge/Deployed%20on-Render-purple)

**Live API:** [https://crypto-volatility-api-645g.onrender.com](https://crypto-volatility-api-645g.onrender.com)  
**Interactive Docs:** [https://crypto-volatility-api-645g.onrender.com/docs](https://crypto-volatility-api-645g.onrender.com/docs)

## Overview
Production FastAPI backend that serves a **logistic regression model** trained to predict high crypto volatility from **12 market features**.

| Detail | Value |
|--------|--------|
| Model | Logistic Regression |
| Features | 12 (returns, rolling volatility, volume, range, etc.) |
| Threshold | 0.275 (27.5%) |
| Output | Probability + High / Low-Moderate label |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Feature list + threshold |
| `POST` | `/predict` | Run prediction |

### Example request
```json
POST /predict
{
  "features": {
    "daily_return": 0.045,
    "return_volatility_3d": 0.038,
    "return_volatility_7d": 0.042,
    "return_volatility_14d": 0.036,
    "return_volatility_30d": 0.031,
    "return_mean_7d": 0.018,
    "return_mean_14d": 0.012,
    "return_mean_30d": 0.008,
    "daily_range": 0.062,
    "open_close_change": 0.041,
    "volume_change": 0.85,
    "log_volume": 18.4
  }
}
```
### Example response
```json

{
  "prediction": 1,
  "label": "High volatility",
  "high_volatility_probability": 0.541,
  "threshold": 0.275,
  "disclaimer": "Experimental model; not financial advice."
}
