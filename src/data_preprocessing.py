import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(raw_data_path="data/raw/Dry_Bean_Dataset.xlsx", output_dir="data/processed/"):
    print("Iniciando preprocesamiento de datos...")

    # 1. Cargar el dataset
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"No se encontró el archivo en {raw_data_path}")

    df = pd.read_excel(raw_data_path)
    print(f"Dataset cargado exitosamente. Forma original: {df.shape}")

    # Limpiar posibles valores nulos
    df = df.dropna()

    # 2. Separar características (X) y etiqueta (Y)
    # La columna objetivo se llama 'Class' en este dataset
    X = df.drop(columns=['Class'])
    y = df['Class']

    # 3. Dividir en 80% entrenamiento y 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # 4. Escalar los datos (Crucial para K-Means y SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Solo transformamos para evitar data leakage
    X_test_scaled = scaler.transform(X_test)

    # Convertir de vuelta a DataFrames para guardarlos fácilmente
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 5. Guardar los datos procesados
    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled_df.to_csv(os.path.join(
        output_dir, "X_train.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(
        output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print("Preprocesamiento completado. Archivos guardados en:", output_dir)
    print(f"Dimensiones X_train: {X_train_scaled_df.shape}")
    print(f"Dimensiones X_test: {X_test_scaled_df.shape}")


if __name__ == "__main__":
    preprocess_data()
