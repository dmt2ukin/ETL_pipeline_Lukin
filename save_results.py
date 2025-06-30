import json
import os

def save_metrics(metrics: dict, filepath="results/metrics.json"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    from train_model import train
    from evaluate_model import evaluate
    from load_data import load_dataset
    from preprocess_data import preprocess

    df = load_dataset("data/breast_cancer_dataset.csv")
    X, y, _ = preprocess(df)
    model = train(X, y)
    metrics = evaluate(model, X, y)
    save_metrics(metrics)
