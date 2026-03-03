import yaml
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data(data_dir="data/processed/"):
    X_train = pd.read_csv(f"{data_dir}X_train.csv")
    X_test = pd.read_csv(f"{data_dir}X_test.csv")
    y_train = pd.read_csv(f"{data_dir}y_train.csv")['Class']
    y_test = pd.read_csv(f"{data_dir}y_test.csv")['Class']
    return X_train, X_test, y_train, y_test


def load_configs(config_path="configs/supervised_params.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_batches(X, y, batch_size):
    """Generador simple para extraer minibatches de los datos."""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        yield X.iloc[batch_idx], y.iloc[batch_idx]


def run_supervised_engine():
    print("Iniciando Entrenamiento Concurrente y Successive Halving...")
    X_train_full, X_test, y_train_full, y_test = load_data()
    configs = load_configs()

    # Extraemos un pequeño set de validación del train set para evaluar las épocas
    # y decidir a qué modelo eliminar sin tocar los datos de testing.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    classes = np.unique(y_train_full)
    max_epochs = configs['training']['max_epochs']
    prune_every = configs['training']['prune_every_n_epochs']

    # 1. Instanciar los modelos
    active_models = []

    for cfg in configs['logistic_regression']:
        model = SGDClassifier(loss='log_loss', alpha=cfg['alpha'], learning_rate=cfg['learning_rate'],
                              eta0=cfg['eta0'], random_state=42)
        active_models.append(
            {"name": cfg['name'], "model": model, "batch_size": cfg['batch_size'], "type": "LR"})

    for cfg in configs['svm']:
        model = SGDClassifier(loss='hinge', alpha=cfg['alpha'], learning_rate=cfg['learning_rate'],
                              eta0=cfg['eta0'], random_state=42)
        active_models.append(
            {"name": cfg['name'], "model": model, "batch_size": cfg['batch_size'], "type": "SVM"})

    print(
        f"\nIniciando entrenamiento con {len(active_models)} modelos concurrentes.")
    print(f"Épocas máximas: {max_epochs} | Poda cada: {prune_every} épocas.\n")

    # 2. Bucle de Entrenamiento por Épocas
    for epoch in range(1, max_epochs + 1):
        # Entrenar cada modelo activo una época completa usando sus respectivos minibatches
        for item in active_models:
            model = item['model']
            batch_size = item['batch_size']

            for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
                model.partial_fit(X_batch, y_batch, classes=classes)

        # 3. Lógica de Evaluación y Poda (Pruning)
        if epoch % prune_every == 0 and len(active_models) > 2:
            print(f"--- Época {epoch}: Evaluando modelos para poda ---")

            # Evaluar todos en el set de validación
            for item in active_models:
                y_val_pred = item['model'].predict(X_val)
                item['val_acc'] = accuracy_score(y_val, y_val_pred)
                print(f"  {item['name']} -> Val Acc: {item['val_acc']:.4f}")

            # Ordenar por accuracy de validación (de mejor a peor)
            active_models.sort(key=lambda x: x['val_acc'], reverse=True)

            # Descartar el peor
            worst_model = active_models.pop()
            print(
                f"PODA: Se descarta '{worst_model['name']}' por bajo desempeño.\n")

    # 4. Evaluación Final en Testing (20%) de los 2 mejores
    print("\n" + "="*50)
    print("RESULTADOS FINALES DE LOS 2 MEJORES MODELOS (TEST SET)")
    print("="*50)

    for i, item in enumerate(active_models[:2]):
        final_model = item['model']
        y_test_pred = final_model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)

        print(f"Top {i+1}: {item['name']} ({item['type']})")
        print(
            f" -> Accuracy en Test (20% invisible): {test_acc:.4f} ({test_acc*100:.2f}%)")
        print("-" * 50)


if __name__ == "__main__":
    run_supervised_engine()
