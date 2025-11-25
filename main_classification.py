import cv2
import numpy as np
import pandas as pd
import time
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# IMPORTAMOS LAS FASES ANTERIORES
from preprocess import run_preprocessing
from feature_extraction import run_feature_extraction

# --- CONFIGURACIÓN ---
# Ruta RELATIVA donde están las imágenes de prueba (Ajusta si es necesario)
BASE_IMAGE_DIR = 'data_object_image_2/training/image_2/'
K_VALUE = 5  # Valor para el modelo KNN


# =========================================================================
# UTILITY: Función para clasificar un objeto individual (Usa el modelo entrenado)
# =========================================================================

def classify_single_object(cropped_img, scaler, knn_model):
    """Extrae features y clasifica una única imagen recortada con el modelo KNN."""

    # 1. Extracción de Features (Copia de la lógica de feature_extraction.py)
    h, w = cropped_img.shape[:2]
    aspect_ratio = w / h

    if h <= 0 or w <= 0: return "N/A"

    try:
        img_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return "N/A"

    h_avg = np.mean(img_hsv[:, :, 0])
    s_avg = np.mean(img_hsv[:, :, 1])
    v_avg = np.mean(img_hsv[:, :, 2])

    try:
        detector = cv2.SIFT_create(nfeatures=500)
    except AttributeError:
        detector = cv2.FastFeatureDetector_create()

    keypoints, _ = detector.detectAndCompute(cropped_img, None)
    num_keypoints = len(keypoints)

    # 2. Vector de Features
    single_feature_vector = np.array([[aspect_ratio, h_avg, s_avg, v_avg, num_keypoints]])

    # 3. Normalización (USANDO EL SCALER ENTRENADO)
    # Se debe aplicar la transformación con el scaler que se entrenó en la Fase 3
    single_scaled_feature = scaler.transform(single_feature_vector)

    # 4. Predicción
    prediction = knn_model.predict(single_scaled_feature)[0]

    # 5. Interpretación (0=Car, 1=Truck)
    predicted_class = 'Truck' if prediction == 1 else 'Car'
    return predicted_class


# =========================================================================
# FUNCIÓN DE CLASIFICACIÓN Y ENTRENAMIENTO (FASE 4)
# =========================================================================

def classify_vehicles(X_train, Y_train, X_test, Y_test, test_data_visual):
    """Entrena y predice con KNN y K-Means, y genera el DataFrame de resultados."""

    print("\n=======================================================")
    print("FASE 4: CLASIFICACIÓN Y PREDICCIÓN")
    print("=======================================================")

    # 1. CLASIFICACIÓN SUPERVISADA: KNN
    start_knn = time.time()
    knn_model = KNeighborsClassifier(n_neighbors=K_VALUE)
    knn_model.fit(X_train, Y_train)
    Y_pred_knn = knn_model.predict(X_test)
    elapsed_knn = time.time() - start_knn

    print(f"\n--- KNN (K={K_VALUE}) ---")
    print(f"Tiempo de entrenamiento y predicción: {elapsed_knn:.4f} segundos.")

    # 2. AGRUPAMIENTO NO SUPERVISADO: K-MEANS
    start_kmeans = time.time()
    kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_model.fit(X_train)
    Y_pred_kmeans = kmeans_model.predict(X_test)
    elapsed_kmeans = time.time() - start_kmeans

    print(f"\n--- K-MEANS (K=2) ---")
    print(f"Tiempo de entrenamiento y predicción: {elapsed_kmeans:.4f} segundos.")

    # GENERACIÓN DE RESULTADOS
    results_df = pd.DataFrame({
        'Ground_Truth': Y_test,
        'Pred_KNN': Y_pred_knn,
        'Pred_KMeans': Y_pred_kmeans,
        'Image_Data': test_data_visual
    })

    results_df['GT_Class'] = results_df['Ground_Truth'].apply(lambda x: 'Truck' if x == 1 else 'Car')
    results_df['KNN_Class'] = results_df['Pred_KNN'].apply(lambda x: 'Truck' if x == 1 else 'Car')
    results_df['KMeans_Cluster'] = results_df['Pred_KMeans'].apply(lambda x: f'Cluster {x}')

    print("\n--- RESULTADOS (Primeras 10 predicciones) ---")
    print(results_df[['GT_Class', 'KNN_Class', 'KMeans_Cluster']].head(10))

    return results_df, knn_model


# =========================================================================
# FUNCIÓN DE DIBUJADO EN IMAGEN COMPLETA (Visualización Final del TFG)
# =========================================================================

def run_realtime_detection(knn_model, scaler, results_df, num_samples=3):
    """
    Usa los Bounding Boxes (BBox) del Ground Truth para simular la detección
    y clasifica cada región con el modelo KNN, dibujando el recuadro.
    """

    print("\n--- INICIO DE LA INSPECCIÓN VISUAL EN IMÁGENES COMPLETAS ---")
    print(f"Mostrando {num_samples} predicciones del Test Set usando BBOX del Ground Truth.")
    print("Ventana de OpenCV: Presiona cualquier tecla para pasar a la siguiente imagen.")

    # 1. Itera sobre las muestras de resultados del Test Set
    for i in range(min(num_samples, len(results_df))):
        row = results_df.iloc[i]

        # Obtener datos para la visualización
        image_id = row['Image_Data']['image_id']
        bbox_float = row['Image_Data']['bbox']

        # Ruta de la imagen completa
        test_image_path = os.path.join(BASE_IMAGE_DIR, f'{image_id}.png')

        img = cv2.imread(test_image_path)
        if img is None:
            continue

        # Coordenadas BBox del Ground Truth (GT)
        xmin, ymin, xmax, ymax = map(int, bbox_float)

        # Recorte (Región de Interés) para el clasificador KNN
        detected_roi = img[ymin:ymax, xmin:xmax]

        # Clasificar la región usando tu modelo KNN
        predicted_class = classify_single_object(detected_roi, scaler, knn_model)

        # Etiqueta del GT para comparación
        gt_class = row['GT_Class']

        # 2. Dibujar el recuadro y el texto

        # Color: Verde si acierta (o si es Car) Rojo si falla (o si es Truck)
        is_correct = (predicted_class == gt_class)
        color = (0, 255, 0) if is_correct else (0, 0, 255)  # Verde si acierta, Rojo si falla
        thickness = 3

        # Dibujar rectángulo (el recuadro que querías)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        # Poner el texto: PREDICCIÓN / (VERDAD FUNDAMENTAL)
        text_label = f"PRED: {predicted_class} (GT: {gt_class})"
        cv2.putText(img, text_label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

        # Mostrar la imagen
        cv2.imshow(f"Clasificacion KNN en Imagen Completa (ID: {image_id})", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 1. FASE 1 & 2: Preprocesamiento y Recorte
    train_set, test_set = run_preprocessing()

    if train_set and test_set:
        # 2. FASE 3: Extracción y Normalización de Características
        X_train_scaled, Y_train, X_test_scaled, Y_test, test_data_visual, scaler = run_feature_extraction(train_set,
                                                                                                          test_set)

        # 3. FASE 4: Clasificación y Obtención de Resultados
        results_df, knn_model = classify_vehicles(X_train_scaled, Y_train, X_test_scaled, Y_test, test_data_visual)

        # 4. INSPECCIÓN VISUAL (Muestra 5 ejemplos usando los BBoxes de Ground Truth)
        run_realtime_detection(knn_model, scaler, results_df, num_samples=20)