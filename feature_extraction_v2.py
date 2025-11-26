import cv2
import numpy as np
import pandas as pd
import time
import os
# --- CAMBIO IMPORTANTE: Usamos SVC en lugar de KNN ---
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# --- MÉTRICAS ---
from sklearn.metrics import accuracy_score, classification_report
# -----------------
from sklearn.utils.class_weight import compute_class_weight
from skimage.feature import hog

# IMPORTAMOS LAS FASES ANTERIORES
# Importamos la nueva fase 3 y la configuración HOG_IMAGE_SIZE
from preprocess import run_preprocessing
from feature_extraction_v2 import run_feature_extraction, HOG_IMAGE_SIZE


# --- CONFIGURACIÓN ---
BASE_IMAGE_DIR = 'data_object_image_2/training/image_2/'
# K_VALUE ya no se usa, pero se define CLASS_ID_TO_NAME
CLASS_ID_TO_NAME = {0: 'Car', 1: 'Truck', 2: 'Pedestrian', 3: 'Cyclist'}


# =========================================================================
# UTILITY: Función para clasificar un objeto individual (Actualizada para HOG+PCA)
# =========================================================================

def classify_single_object(cropped_img, scaler, pca_model, svc_model):
    """Extrae features (HOG incluido), normaliza, aplica PCA y clasifica con SVC."""

    h, w = cropped_img.shape[:2]
    aspect_ratio = w / h

    if h <= 0 or w <= 0: return "N/A"

    # --- HOG: Redimensionamiento y escala de grises ---
    try:
        resized_img = cv2.resize(cropped_img, HOG_IMAGE_SIZE)
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return "N/A"
    # --------------------------------------------------

    h_avg = np.mean(img_hsv[:, :, 0])
    s_avg = np.mean(img_hsv[:, :, 1])
    v_avg = np.mean(img_hsv[:, :, 2])

    try:
        detector = cv2.SIFT_create(nfeatures=500)
    except AttributeError:
        detector = cv2.FastFeatureDetector_create()

    keypoints, _ = detector.detectAndCompute(cropped_img, None)
    num_keypoints = len(keypoints)

    # 2. Cálculo de HOG
    hog_features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm='L2-Hys',
                        visualize=False, feature_vector=True)


    # 3. Vector de Features (Base + HOG)
    base_features = np.array([aspect_ratio, h_avg, s_avg, v_avg, num_keypoints])
    single_feature_vector = np.concatenate((base_features, hog_features)).reshape(1, -1)


    # 4. Normalización
    single_scaled_feature = scaler.transform(single_feature_vector)

    # 5. Aplicar PCA
    single_pca_feature = pca_model.transform(single_scaled_feature)

    # 6. Predicción con SVC
    prediction = svc_model.predict(single_pca_feature)[0]

    # 7. Interpretación
    predicted_class = CLASS_ID_TO_NAME.get(prediction, "N/A")
    return predicted_class


# =========================================================================
# FUNCIÓN DE CLASIFICACIÓN Y ENTRENAMIENTO (FASE 4)
# =========================================================================

def classify_vehicles(X_train, Y_train, X_test, Y_test, test_data_visual):
    print("\n=======================================================")
    print("FASE 4: CLASIFICACIÓN Y PREDICCIÓN (SVC + PCA)")
    print("=======================================================")

    # 1. CLASIFICACIÓN SUPERVISADA: SVC
    start_svc = time.time()

    # Usamos SVC con kernel RBF y class_weight='balanced'
    svc_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)

    # Entrenar el modelo con las características reducidas por PCA
    svc_model.fit(X_train, Y_train)

    Y_pred_svc = svc_model.predict(X_test)
    elapsed_svc = time.time() - start_svc

    print(f"\n--- SVC (kernel='rbf', class_weight='balanced') ---")
    print(f"Tiempo de entrenamiento y predicción: {elapsed_svc:.4f} segundos.")

    # --- CÁLCULO DE MÉTRICAS DE RENDIMIENTO (NUEVO) ---
    svc_accuracy = accuracy_score(Y_test, Y_pred_svc)
    print(f"\nPrecisión General (Accuracy): {svc_accuracy:.4f} ({svc_accuracy*100:.2f}%)")

    # Muestra métricas detalladas por clase
    target_names = list(CLASS_ID_TO_NAME.values())
    print("\nReporte de Clasificación (SVC - Precisión, Recall, F1-Score):")
    print(classification_report(Y_test, Y_pred_svc, target_names=target_names))
    # -------------------------------------------------


    # 2. AGRUPAMIENTO NO SUPERVISADO: K-MEANS
    start_kmeans = time.time()
    # K-Means sigue usando los datos con PCA (4 clusters)
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_model.fit(X_train)
    Y_pred_kmeans = kmeans_model.predict(X_test)
    elapsed_kmeans = time.time() - start_kmeans

    print(f"\n--- K-MEANS (K=4 sobre datos PCA) ---")
    print(f"Tiempo de entrenamiento y predicción: {elapsed_kmeans:.4f} segundos.")


    # GENERACIÓN DE RESULTADOS
    results_df = pd.DataFrame({
        'Ground_Truth': Y_test,
        'Pred_SVC': Y_pred_svc,
        'Pred_KMeans': Y_pred_kmeans,
        'Image_Data': test_data_visual
    })

    map_to_class_name = lambda x: CLASS_ID_TO_NAME.get(x, 'Desconocido')

    results_df['GT_Class'] = results_df['Ground_Truth'].apply(map_to_class_name)
    results_df['SVC_Class'] = results_df['Pred_SVC'].apply(map_to_class_name)
    results_df['KMeans_Cluster'] = results_df['Pred_KMeans'].apply(lambda x: f'Cluster {x}')

    print("\n--- RESULTADOS (Primeras 10 predicciones) ---")
    print(results_df[['GT_Class', 'SVC_Class', 'KMeans_Cluster']].head(10))

    return results_df, svc_model


# =========================================================================
# FUNCIONES DE VISUALIZACIÓN REFORZADAS
# =========================================================================

# Esta función ahora muestra la predicción SVC
def inspect_predictions(results_df, num_samples=5):
    """Muestra visualmente algunas predicciones (solo recortes) del Test Set."""

    print(f"\n--- INSPECCIÓN VISUAL DE RECORTES ({num_samples} MUESTRAS) ---")
    print("Ventana de OpenCV: Presiona cualquier tecla para pasar al siguiente recorte.")

    cv2.waitKey(1)

    for i in range(min(num_samples, len(results_df))):
        row = results_df.iloc[i]

        img = row['Image_Data']['image']
        gt = row['GT_Class']
        pred = row['SVC_Class']

        color = (0, 255, 0) if (pred == gt) else (0, 0, 255)
        text_label = f"RECORTES PRED: {pred} (GT: {gt})"

        cv2.imshow(text_label, img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# Esta función ahora usa el modelo SVC y el objeto PCA
def run_realtime_detection(svc_model, scaler, pca_model, results_df, num_samples=5):
    """
    Usa los Bounding Boxes (BBox) del Ground Truth para simular la detección
    y clasifica cada región con el modelo SVC + PCA, dibujando el recuadro en la imagen completa.
    """

    print("\n--- INICIO DE LA INSPECCIÓN VISUAL EN IMÁGENES COMPLETAS ---")
    print(f"Mostrando {num_samples} predicciones del Test Set usando BBOX del Ground Truth.")
    print("Ventana de OpenCV: Presiona cualquier tecla para pasar a la siguiente imagen.")

    cv2.waitKey(1)

    for i in range(min(num_samples, len(results_df))):
        row = results_df.iloc[i]

        image_id = row['Image_Data']['image_id']
        bbox_float = row['Image_Data']['bbox']

        test_image_path = os.path.join(BASE_IMAGE_DIR, f'{image_id}.png')

        img = cv2.imread(test_image_path)
        if img is None:
            continue

        xmin, ymin, xmax, ymax = map(int, bbox_float)
        detected_roi = img[ymin:ymax, xmin:xmax]

        # Clasificar la región usando el modelo SVC y PCA
        predicted_class = classify_single_object(detected_roi, scaler, pca_model, svc_model)

        gt_class = row['GT_Class']

        is_correct = (predicted_class == gt_class)
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        thickness = 3

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        text_label = f"PRED: {predicted_class} (GT: {gt_class})"
        cv2.putText(img, text_label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

        cv2.imshow(f"FINAL - ID: {image_id}", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 1. FASE 1 & 2: Preprocesamiento y Recorte
    train_set, test_set = run_preprocessing()

    if train_set and test_set:
        # 2. FASE 3: Extracción, Normalización y PCA (Retorna el objeto PCA)
        X_train_pca, Y_train, X_test_pca, Y_test, test_data_visual, scaler, pca = run_feature_extraction(train_set,
                                                                                                          test_set)

        # 3. FASE 4: Clasificación y Obtención de Resultados con SVC
        results_df, svc_model = classify_vehicles(X_train_pca, Y_train, X_test_pca, Y_test, test_data_visual)

        # 4. INSPECCIÓN VISUAL
        # La función de visualización final debe recibir el SVC y el PCA
        run_realtime_detection(svc_model, scaler, pca, results_df, num_samples=40)