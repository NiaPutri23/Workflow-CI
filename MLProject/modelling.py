import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 

# Inisialisasi tracking MLflow DagsHub
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("travel_insurance_experiment")

# Gunakan autolog
mlflow.sklearn.autolog()

# Load data
df = pd.read_csv("TravelInsurancePrediction_preprocessed.csv")
X = df.drop("TravelInsurance", axis=1)
y = df["TravelInsurance"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    
    # Simpan model
    output_path = "trained_model.pkl"
    joblib.dump(model, output_path)

    # Log model sebagai artifact
    mlflow.log_artifact(output_path)
    mlflow.log_metric("accuracy_manual", acc)
