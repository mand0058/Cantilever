# task2/src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path="data/creditcard.csv"):
    df = pd.read_csv(path)
    # Basic sanity
    df = df.dropna()
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale 'Amount' and 'Time' (other PCA columns V1..V28 already scaled)
    scaler = StandardScaler()
    X[['Time','Amount']] = scaler.fit_transform(X[['Time','Amount']])
    return X, y, scaler
