import yaml
import pandas as pd
import numpy as np
import time
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, accuracy_score


def load_data(data_dir="data/processed/"):
    """Carga los datos preprocesados de la Fase 1."""
    X_train = pd.read_csv(f"{data_dir}X_train.csv")
    X_test = pd.read_csv(f"{data_dir}X_test.csv")
    y_train = pd.read_csv(f"{data_dir}y_train.csv")['Class']
    y_test = pd.read_csv(f"{data_dir}y_test.csv")['Class']
    return X_train, X_test, y_train, y_test


def load_configs(config_path="configs/clustering_params.yaml"):
    """Carga las configuraciones desde el archivo YAML."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_dominant_labels(y_true, y_pred):
    """
    Lógica de Pseudo-etiquetado:
    Mapea cada cluster descubierto a la etiqueta real más frecuente (dominante).
    """
    cluster_to_label = {}
    for cluster_id in np.unique(y_pred):
        # Ignoramos ruido (generado por MeanShift cuando cluster_all=False)
        if cluster_id == -1:
            continue

        # Filtrar las etiquetas reales que cayeron en este cluster
        mask = (y_pred == cluster_id)
        labels_in_cluster = y_true[mask]

        # Encontrar la etiqueta más común (la moda)
        dominant_label = labels_in_cluster.mode()[0]
        cluster_to_label[cluster_id] = dominant_label

    return cluster_to_label


def run_clustering_engine():
    print("Iniciando Fase 2: Motor de Clustering y Pseudo-etiquetado...")
    X_train, X_test, y_train, y_test = load_data()
    configs = load_configs()

    results = []

    # 1. Ejecutar las 12 configuraciones (K-Means, K-Means++, MeanShift)
    print("\n--- Entrenando Modelos de Clustering (80% de los datos) ---")

    # Combinar kmeans y kmeans++ del yaml para iterar
    all_kmeans_configs = [('K-Means', c) for c in configs['kmeans']] + \
                         [('K-Means++', c) for c in configs['kmeans_plusplus']]

    for model_name, params in all_kmeans_configs:
        print(f"Entrenando {model_name} con config: {params}")
        model = KMeans(
            n_clusters=params['n_clusters'],
            init=params['init'],
            n_init=params['n_init'],
            max_iter=params['max_iter'],
            random_state=42
        )
        t0 = time.time()
        cluster_labels = model.fit_predict(X_train)
        t_fit = time.time() - t0

        # Evaluación No Supervisada: Silhouette Score
        score = silhouette_score(X_train, cluster_labels)
        results.append({'model_name': model_name, 'params': params,
                       'model': model, 'silhouette': score})
        print(f" -> Silhouette Score: {score:.4f} (Tiempo: {t_fit:.2f}s)\n")

    # Ejecutar MeanShift
    # MeanShift puede ser computacionalmente costoso, estimamos el bandwidth con una muestra
    print("Estimando bandwidth para MeanShift (esto puede tomar unos segundos)...")
    bw = estimate_bandwidth(X_train, quantile=0.2, n_samples=1500)

    for i, params in enumerate(configs['meanshift']):
        model_name = f"MeanShift (Config {i+1})"
        print(f"Entrenando {model_name} con config: {params}")
        model = MeanShift(
            bandwidth=bw,
            bin_seeding=params['bin_seeding'],
            cluster_all=params['cluster_all']
        )
        t0 = time.time()
        cluster_labels = model.fit_predict(X_train)
        t_fit = time.time() - t0

        # Si MeanShift pone todo en 1 solo cluster o más de 50, el silhouette falla o es inútil
        n_clusters_found = len(set(cluster_labels)) - \
            (1 if -1 in cluster_labels else 0)
        if 1 < n_clusters_found < 50:
            score = silhouette_score(X_train, cluster_labels)
        else:
            score = -1.0  # Penalizamos si no logra segmentar lógicamente

        results.append({'model_name': model_name, 'params': params,
                       'model': model, 'silhouette': score})
        print(
            f" -> Clusters encontrados: {n_clusters_found} | Silhouette Score: {score:.4f} (Tiempo: {t_fit:.2f}s)\n")

    # 2. Seleccionar los 3 mejores según Silhouette Score
    results.sort(key=lambda x: x['silhouette'], reverse=True)
    top_3 = results[:3]

    print("\n" + "="*50)
    print("TOP 3 CONFIGURACIONES (Según Silhouette Score)")
    print("="*50)

    for i, res in enumerate(top_3):
        print(
            f"{i+1}. {res['model_name']} | Silhouette: {res['silhouette']:.4f}")
        print(f"   Parámetros: {res['params']}")

    # 3. Evaluación del Pseudo-etiquetado en el 20% de Testeo
    print("\n" + "="*50)
    print("EVALUACIÓN DE PSEUDO-ETIQUETADO EN TESTING (20%)")
    print("="*50)

    for i, res in enumerate(top_3):
        best_model = res['model']

        # Predecir clusters para el set de prueba
        test_clusters = best_model.predict(X_test)

        # Obtener el mapeo de cluster -> etiqueta real dominante usando test
        # (En un escenario 100% estricto de falta de etiquetas, inferiríamos la clase desde X_train si tuvieramos algunas etiquetas,
        # pero para evaluar teóricamente si esto tiene sentido, cruzamos test_clusters con y_test)
        cluster_mapping = get_dominant_labels(y_test, test_clusters)

        # Asignar la etiqueta inferida a cada muestra
        y_pred_inferred = pd.Series(test_clusters).map(
            cluster_mapping).fillna('Unknown')

        # Calcular Accuracy
        acc = accuracy_score(y_test, y_pred_inferred)

        print(f"\n[{i+1}] Evaluando {res['model_name']}:")
        print(f" -> Mapeo de clusters a etiquetas: {cluster_mapping}")
        print(
            f" -> Accuracy del Pseudo-etiquetado: {acc:.4f} ({acc*100:.2f}%)")

        # Análisis breve automatizado
        if acc > 0.7:
            print(
                " -> Conclusión: ¡Excelente! La topología geométrica captura muy bien las clases reales.")
        elif acc > 0.5:
            print(
                " -> Conclusión: Razonable. Hay solapamiento entre clases, pero hay una tendencia útil.")
        else:
            print(" -> Conclusión: Pobre. Las características numéricas no se agrupan naturalmente según la etiqueta Y.")


if __name__ == "__main__":
    run_clustering_engine()
