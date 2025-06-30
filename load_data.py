import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_dataset("data/breast_cancer_dataset.csv")
    print(df.info())
