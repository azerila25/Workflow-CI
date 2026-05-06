import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os

if __name__ == "__main__":
    # Load dataset (pastikan file dataset ada di folder yang sama)
    df = pd.read_csv("dataset_preprocessing.csv")
    
    # Skenario sederhana: pisahkan fitur dan target
    # Sesuaikan 'target' dengan nama kolom asli di dataset kamu
    X = df.drop(columns=['target']) 
    y = df['target']
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        
        # Log model ke MLflow
        mlflow.sklearn.log_model(model, "model")
        print("Model berhasil dilatih dan disimpan ke MLflow.")