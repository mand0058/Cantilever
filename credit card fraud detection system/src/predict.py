# task2/src/predict.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

model_path = Path("models/fraud_model.pkl")
scaler_path = Path("models/scaler.pkl")

if not model_path.exists() or not scaler_path.exists():
    print("Model or scaler missing. Run: python src/train_model.py")
    exit(1)

model = joblib.load(str(model_path))
scaler = joblib.load(str(scaler_path))

def predict_transaction(tx_series):
    # tx_series: pandas Series or 1D array with same features as original X (Time, V1..V28, Amount)
    x = pd.DataFrame([tx_series])
    # scale Time and Amount
    x[['Time','Amount']] = scaler.transform(x[['Time','Amount']])
    prob = model.predict_proba(x)[:,1][0]
    pred = model.predict(x)[0]
    return {"probability_fraud": float(prob), "prediction": int(pred)}

if __name__ == "__main__":
    # Example: load sample from data after training
    df = pd.read_csv("data/creditcard.csv")
    sample = df.drop("Class", axis=1).iloc[0]
    print("Sample prediction:", predict_transaction(sample))
