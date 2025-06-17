import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 
import argparse

def main(data_path):
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop("TravelInsurance", axis=1)
    y = df["TravelInsurance"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("travel_insurance_experiment")
    mlflow.sklearn.autolog()

    # Training model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

    # Simpan dan log model
    joblib.dump(model, "trained_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="TravelInsurancePrediction_preprocessed.csv")
    args = parser.parse_args()
    main(args.data_path)
