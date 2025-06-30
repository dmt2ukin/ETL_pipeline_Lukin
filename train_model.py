
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import joblib

def train(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

if __name__ == "__main__":
    from load_data import load_dataset
    from preprocess_data import preprocess_breast_cancer_data
    df = load_dataset("data/breast_cancer_dataset.csv")
    X, y, _ = preprocess(df)
    model = train(X, y)
    joblib.dump(model, "results/model.joblib")
