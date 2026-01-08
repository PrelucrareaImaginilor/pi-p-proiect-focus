import numpy as np
import cv2
import os
from tqdm import tqdm
from config import INPUT_DIR, OUTPUT_DIR, TARGET_SIZE


VALID_EXT = ('.png', '.bmp', '.jpg', '.jpeg')


def generate_gei(silhouette_folder, output_path):
    """
    Genereaza GEI din toate siluetele dintr-un folder.
    Optimizat pentru viteza.
    """

    # Lista de imagini valide
    silhouette_files = [
        f for f in os.listdir(silhouette_folder)
        if f.lower().endswith(VALID_EXT)
    ]

    # Daca folderul nu are imagini, il sarim instant
    if not silhouette_files:
        return False

    silhouettes = []

    for file_name in silhouette_files:
        path = os.path.join(silhouette_folder, file_name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # Resize rapid
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        silhouettes.append(img.astype(np.float32))

    if not silhouettes:
        return False

    # GEI = media siluetelor
    gei = np.mean(silhouettes, axis=0)

    # Salvare imagine
    cv2.imwrite(output_path, gei.astype(np.uint8))
    return True


def process_casia_b_dataset():
    """
    Proceseaza TOT datasetul CASIA-B si genereaza GEI-uri pentru toate unghiurile.
    Optimizat pentru viteza.
    """

    print("=== Incepe procesarea dataset-ului CASIA-B (optimizat) ===")

    gei_info = []

    subjects = sorted([
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ])

    for subject_id in tqdm(subjects, desc="Subiecti"):
        subject_path = os.path.join(INPUT_DIR, subject_id)

        for seq_type in os.listdir(subject_path):
            seq_path = os.path.join(subject_path, seq_type)
            if not os.path.isdir(seq_path):
                continue

            for angle in os.listdir(seq_path):
                angle_path = os.path.join(seq_path, angle)
                if not os.path.isdir(angle_path):
                    continue

                # Verificam rapid daca folderul are imagini
                if not any(f.lower().endswith(VALID_EXT) for f in os.listdir(angle_path)):
                    continue

                # Normalizam unghiul
                try:
                    angle_int = int(angle)
                except:
                    continue

                output_filename = f"{subject_id}_{seq_type}_{angle_int:03d}_gei.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)

                if generate_gei(angle_path, output_path):
                    gei_info.append({
                        "path": output_path,
                        "subject_id": subject_id,
                        "condition": seq_type,
                        "angle": angle_int
                    })

    # Salvam informatiile
    np.save(os.path.join(OUTPUT_DIR, "gei_info.npy"), gei_info)

    print(f"\n=== Generate {len(gei_info)} GEI-uri (optimizat) ===")
    print(f"Fisier salvat: {os.path.join(OUTPUT_DIR, 'gei_info.npy')}\n")

    return gei_info
