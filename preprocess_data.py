import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

if __name__ == "__main__":
    from load_data import load_dataset
    df = load_dataset("data/breast_cancer_dataset.csv")
    X, y, _ = preprocess(df)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
