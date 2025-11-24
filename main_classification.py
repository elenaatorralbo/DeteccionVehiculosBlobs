import cv2
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# IMPORTAMOS LAS FASES ANTERIORES
from preprocess import run_preprocessing
from feature_extraction import run_feature_extraction


def classify_vehicles(X_train, Y_train, X_test, Y_test, test_data_visual):
    """Entrena y predice con KNN y K-Means, y facilita la inspección visual."""

    print("\n=======================================================")
    print("FASE 4: CLASIFICACIÓN Y PREDICCIÓN")
    print("=======================================================")

    # --- 1. CLASIFICACIÓN SUPERVISADA: KNN ---
    start_knn = time.time()
    k_value = 5
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(X_train, Y_train)
    Y_pred_knn = knn_model.predict(X_test)
    elapsed_knn = time.time() - start_knn

    print(f"\n--- KNN (K={k_value}) ---")
    print(f"Tiempo de entrenamiento y predicción: {elapsed_knn:.2f} segundos.")

    # --- 2. AGRUPAMIENTO NO SUPERVISADO: K-MEANS ---
    start_kmeans = time.time()
    k_clusters = 2
    kmeans_model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    kmeans_model.fit(X_train)
    Y_pred_kmeans = kmeans_model.predict(X_test)
    elapsed_kmeans = time.time() - start_kmeans

    print(f"\n--- K-MEANS (K={k_clusters}) ---")
    print(f"Tiempo de entrenamiento y predicción: {elapsed_kmeans:.2f} segundos.")

    # --- 3. ANÁLISIS Y PREPARACIÓN VISUAL ---

    # Creamos un DataFrame para un análisis fila por fila
    results_df = pd.DataFrame({
        'Ground_Truth': Y_test,
        'Pred_KNN': Y_pred_knn,
        'Pred_KMeans': Y_pred_kmeans,
        'Image_Data': test_data_visual
    })

    # Añadir las clases de texto para mejor lectura
    results_df['GT_Class'] = results_df['Ground_Truth'].apply(lambda x: 'Truck' if x == 1 else 'Car')
    results_df['KNN_Class'] = results_df['Pred_KNN'].apply(lambda x: 'Truck' if x == 1 else 'Car')

    print("\n--- RESULTADOS (Primeras 10 predicciones) ---")
    print(results_df[['GT_Class', 'KNN_Class', 'Pred_KMeans']].head(10))

    return results_df


def inspect_predictions(results_df, model_type='KNN', num_samples=5):
    """Muestra visualmente algunas predicciones (ej. las primeras 5)."""

    print(f"\n--- INSPECCIÓN VISUAL DE PREDICCIONES ({model_type}) ---")
    print("Ventana de OpenCV: Presiona cualquier tecla para pasar a la siguiente imagen.")

    for i in range(min(num_samples, len(results_df))):
        row = results_df.iloc[i]

        img = row['Image_Data']['image']
        gt = row['GT_Class']

        if model_type == 'KNN':
            pred = row['KNN_Class']
            result_label = f"KNN: {pred} | GT: {gt}"
        else:  # K-Means
            cluster = row['Pred_KMeans']
            result_label = f"K-Means: Cluster {cluster} | GT: {gt}"

        cv2.imshow(result_label, img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 1. FASE 1 & 2: Preprocesamiento y Recorte
    train_set, test_set = run_preprocessing()

    if train_set and test_set:
        # 2. FASE 3: Extracción y Normalización de Características
        X_train_scaled, Y_train, X_test_scaled, Y_test, test_data_visual = run_feature_extraction(train_set, test_set)

        # 3. FASE 4: Clasificación y Obtención de Resultados
        results_df = classify_vehicles(X_train_scaled, Y_train, X_test_scaled, Y_test, test_data_visual)

        # 4. INSPECCIÓN VISUAL
        # Puedes cambiar 'KNN' a 'K-Means' para ver las asignaciones de clústeres.
        # Puedes cambiar 'num_samples' para ver más o menos ejemplos.
        inspect_predictions(results_df, model_type='KNN', num_samples=5)