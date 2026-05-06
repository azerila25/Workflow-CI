import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # Ubah dari Classifier ke Regressor
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("dataset_preprocessing.csv")
    
    # Pastikan nama kolom 'target' sesuai dengan yang ada di dataset_preprocessing.csv kamu
    X = df.drop(columns=['target']) 
    y = df['target']
    
    with mlflow.start_run():
        # Gunakan Regressor karena data kamu terdeteksi sebagai 'continuous'
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        # Log model ke MLflow
        mlflow.sklearn.log_model(model, "model")
        print("Model Regressor berhasil dilatih dan disimpan ke MLflow.")