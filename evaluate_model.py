from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

def evaluate(model, X, y):
    preds = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1_score": f1_score(y, preds),
    }
    return metrics
