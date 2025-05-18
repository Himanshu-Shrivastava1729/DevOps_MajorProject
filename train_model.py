# Script to train model and store it.

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def load_and_preprocess():
    print("Loading and preprocessing data...")
    df = pd.read_csv("Assignment-2_Data.csv")

    # Convert target to binary
    df["y"] = df["y"].apply(lambda x: 1 if x == "yes" else 0)

    # Remove outliers
    df = df[df["age"] <= 100]
    for col in ["age", "balance"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # One-hot encoding
    df = pd.get_dummies(
        df, columns=["job", "marital", "month", "poutcome", "education", "contact"]
    )

    # Encode binary columns
    for col in ["default", "housing", "loan"]:
        df[col] = df[col].map({"yes": 1, "no": 0})

    print("Data preprocessing complete.")
    return df

def train_models(df):
    print("Starting model training...")
    X = df.drop("y", axis=1)
    y = df["y"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "gnb": GaussianNB(),
        "dt": DecisionTreeClassifier(),
    }

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump((model, X.columns), f"models/{name}.pkl")
        print(f"Trained and saved model: {name}")

    print("All models trained and saved successfully.")

if __name__ == "__main__":
    df = load_and_preprocess()
    train_models(df)

