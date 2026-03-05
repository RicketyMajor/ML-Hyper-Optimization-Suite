import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd


def plot_pca_clusters(X, y_real, y_kmeans, title="PCA: Clases Reales vs Clusters Descubiertos"):
    """
    Reduce los datos a 2 dimensiones usando PCA y grafica las clases reales
    junto a los clusters descubiertos por K-Means para comparación visual.
    """
    # Aplicar PCA para reducir a 2 componentes
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Crear un DataFrame para seaborn
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df_pca['Real Class'] = y_real.values
    df_pca['K-Means Cluster'] = y_kmeans

    # Configurar la figura
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)

    # Gráfico 1: Clases Reales
    sns.scatterplot(ax=axes[0], x='PC1', y='PC2', hue='Real Class',
                    data=df_pca, palette='tab10', s=50, alpha=0.7)
    axes[0].set_title('Etiquetas Reales (Ground Truth)')

    # Gráfico 2: Clusters de K-Means
    sns.scatterplot(ax=axes[1], x='PC1', y='PC2', hue='K-Means Cluster',
                    data=df_pca, palette='Set2', s=50, alpha=0.7)
    axes[1].set_title('Clusters descubiertos (No Supervisado)')

    plt.tight_layout()
    plt.show()

    print(
        f"Varianza explicada por las 2 componentes principales: {sum(pca.explained_variance_ratio_) * 100:.2f}%")


def plot_successive_halving(history_dict):
    """
    Grafica la evolución de la precisión de validación de los modelos
    y visualiza en qué época fueron podados.
    """
    plt.figure(figsize=(12, 7))

    # Paleta de colores para diferenciar bien los modelos
    colors = sns.color_palette("husl", len(history_dict))

    for (model_name, data), color in zip(history_dict.items(), colors):
        epochs = data['epochs']
        acc = data['acc']

        # Graficar la línea de vida del modelo
        plt.plot(epochs, acc, marker='o', label=model_name,
                 linewidth=2.5, markersize=8, color=color)

        # Si el modelo fue podado antes de la época 20, marcamos el final con una 'X'
        if max(epochs) < 20:
            plt.plot(max(epochs), acc[-1], marker='X',
                     color='red', markersize=12)

    # Líneas verticales indicando los momentos de poda (cada 5 épocas)
    for prune_epoch in [5, 10, 15, 20]:
        plt.axvline(x=prune_epoch, color='gray', linestyle='--', alpha=0.5)
        plt.text(prune_epoch + 0.2, 0.75, f'Poda',
                 color='gray', fontsize=10, rotation=90)

    plt.title(
        "Evolución de Modelos (Successive Halving) - Accuracy vs Épocas", fontsize=16)
    plt.xlabel("Épocas de Entrenamiento", fontsize=12)
    plt.ylabel("Accuracy en Validación", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
