import os
import cv2
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Importamos la librería para la barra de progreso

# --- CONFIGURACIÓN DE RUTAS ---
# Ajusta estas rutas si es necesario
BASE_IMAGE_DIR = 'data_object_image_2/training/image_2/'
BASE_LABEL_DIR = 'label_2/training/label_2'  # ¡OJO! Revisa si la carpeta es 'label_2' o 'label_2/'
TARGET_CLASSES = ['Car', 'Truck']
TEST_SIZE = 0.2  # Usaremos 80% para entrenar y 20% para probar
RANDOM_SEED = 42


def parse_kitti_labels():
    """Lee y procesa todos los archivos de etiquetas de KITTI."""
    print("Fase 1: Leyendo y filtrando archivos de etiquetas...")
    all_annotations = []

    if not os.path.isdir(BASE_LABEL_DIR):
        print(f"Error de ruta: No se encuentra el directorio de etiquetas: {BASE_LABEL_DIR}")
        return pd.DataFrame()

    # Usamos tqdm en el bucle de archivos para mostrar progreso
    label_files = [f for f in os.listdir(BASE_LABEL_DIR) if f.endswith('.txt')]
    for filename in tqdm(label_files, desc="Leyendo etiquetas"):
        file_path = os.path.join(BASE_LABEL_DIR, filename)
        image_id = filename.replace('.txt', '')

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue

                obj_class = parts[0]

                if obj_class in TARGET_CLASSES:
                    try:
                        # BBox 2D: xmin, ymin, xmax, ymax (índices 4 a 7)
                        xmin = float(parts[4])
                        ymin = float(parts[5])
                        xmax = float(parts[6])
                        ymax = float(parts[7])

                        all_annotations.append({
                            'image_id': image_id,
                            'class': obj_class,
                            'bbox': [xmin, ymin, xmax, ymax],
                        })
                    except (ValueError, IndexError):
                        continue

    return pd.DataFrame(all_annotations)


def create_datasets_and_crop(df):
    """Divide el DataFrame en Train/Test y realiza el recorte de imágenes."""

    print("\nFase 2: Dividiendo y preparando los conjuntos de datos...")

    # 1. DIVISIÓN INTERNA (80/20)
    image_ids = df['image_id'].unique()
    train_ids, test_ids = train_test_split(
        image_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    train_df = df[df['image_id'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['image_id'].isin(test_ids)].reset_index(drop=True)

    print(f"Total de Anotaciones filtradas (Car/Truck): {len(df)}")
    print(f"Imágenes únicas para entrenamiento: {len(train_ids)}")
    print(f"Imágenes únicas para prueba: {len(test_ids)}")

    # 2. RECORTES (Añadimos tqdm aquí, donde se lee y procesa OpenCV)

    def crop_images_from_df(dataset_df, set_name):
        cropped_data = []

        # tqdm itera sobre las filas del DataFrame y muestra el progreso
        for index, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc=f"Recortando {set_name}"):
            image_filename = row['image_id'] + '.png'
            image_path = os.path.join(BASE_IMAGE_DIR, image_filename)

            img = cv2.imread(image_path)
            if img is None: continue

            # Coordenadas BBox (convertidas a entero)
            xmin, ymin, xmax, ymax = map(int, row['bbox'])

            # Recorte: [ymin:ymax, xmin:xmax]
            cropped_img = img[ymin:ymax, xmin:xmax]

            if cropped_img.size == 0 or cropped_img.shape[0] < 20 or cropped_img.shape[1] < 20:
                continue

            # Almacenar la imagen recortada, la clase y el ID
            cropped_data.append({
                'image': cropped_img,
                'class': row['class'],
                'image_id': row['image_id'],
                'bbox': row['bbox']
            })

        print(f"Recortes válidos generados para {set_name}: {len(cropped_data)}")
        return cropped_data

    train_data = crop_images_from_df(train_df, "Train")
    test_data = crop_images_from_df(test_df, "Test")

    return train_data, test_data


# --- FUNCIÓN PRINCIPAL CON TEMPORIZADOR ---
def run_preprocessing():
    start_time = time.time()  # Iniciar contador de tiempo

    kitti_df = parse_kitti_labels()
    if kitti_df.empty:
        # Finalizar el temporizador incluso si hay un error de ruta
        elapsed = time.time() - start_time
        print(f"\n--- TIEMPO TOTAL: {elapsed:.2f} segundos ---")
        return [], []

    train_set, test_set = create_datasets_and_crop(kitti_df)

    # Finalizar el temporizador si el proceso fue exitoso
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n=======================================================")
    print(f"PREPROCESAMIENTO COMPLETO. FASE 2 LISTA.")
    print(f"TIEMPO TOTAL: {int(minutes)} minutos y {seconds:.2f} segundos.")
    print(f"=======================================================")

    return train_set, test_set


if __name__ == '__main__':
    train_set, test_set = run_preprocessing()
    if train_set:
        print("Mostrando ejemplo de recorte...")
        # Ejemplo: Muestra el primer recorte
        cv2.imshow("Primer Recorte (Presiona cualquier tecla para cerrar)", train_set[0]['image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()