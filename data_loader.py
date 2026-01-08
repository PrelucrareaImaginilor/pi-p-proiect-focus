"""Incarcare si pregatire date pentru antrenare."""

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import OUTPUT_DIR, RANDOM_SEED, VALIDATION_SPLIT, TEST_SPLIT
import os


def parse_gei_filename(filename, folder="GEI_Images"):
    """
    Format acceptat: 001_nm_090_gei.png
    Returneaza subject_id, condition, angle, path
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")

    # Format corect: 4 parti
    # ex: ['001', 'nm', '090', 'gei']
    if len(parts) != 4:
        print(f"[AVERTISMENT] Nume fisier invalid: {filename}")
        return None

    subject = parts[0]
    condition = parts[1]

    try:
        angle = int(parts[2])
    except:
        print(f"[EROARE] Unghi invalid in fisier: {filename}")
        return None

    return {
        "subject_id": subject,
        "condition": condition,
        "angle": angle,
        "path": os.path.join(folder, filename)
    }


def load_gei_info(folder="GEI_Images", save=True):
    """
    Parcurge folderul GEI_Images si extrage metadatele din numele fisierelor.
    Returneaza o lista de dict-uri cu subject_id, condition, angle si path.
    """
    gei_info = []

    for file in os.listdir(folder):
        if file.endswith("_gei.png"):
            info = parse_gei_filename(file, folder)
            if info:
                gei_info.append(info)

    print(f"Incarcate {len(gei_info)} fisiere GEI din {folder}")

    if save:
        np.save(os.path.join(folder, "gei_info.npy"), gei_info)
        print(f"gei_info.npy salvat in {folder}")

    return gei_info


def load_gei_dataset(gei_info, angle_filter=None, condition_filter=None):
    """
    Incarca GEI-urile generate si le pregateste pentru antrenare.
    """

    print("Incarcare date...")

    images = []
    labels = []

    for info in tqdm(gei_info, desc="Incarcare GEI"):
        if angle_filter and info['angle'] not in angle_filter:
            continue
        if condition_filter and info['condition'] not in condition_filter:
            continue

        img = cv2.imread(info['path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[EROARE] Nu pot incarca imaginea: {info['path']}")
            continue

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)

        images.append(img)
        labels.append(info['subject_id'])

    images = np.array(images)
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    print(f"Incarcate {len(images)} imagini din {len(np.unique(labels))} subiecti")

    return images, labels_encoded, label_encoder


def split_dataset(X, y, gei_info):
    """
    Imparte dataset-ul in train, validation si test.
    """

    total_test_size = VALIDATION_SPLIT + TEST_SPLIT

    X_train, X_temp, y_train, y_temp, gei_info_train, gei_info_temp = train_test_split(
        X, y, gei_info, test_size=total_test_size,
        random_state=RANDOM_SEED, stratify=y
    )

    val_size_adjusted = VALIDATION_SPLIT / total_test_size

    X_val, X_test, y_val, y_test, gei_info_val, gei_info_test = train_test_split(
        X_temp, y_temp, gei_info_temp,
        test_size=(1 - val_size_adjusted),
        random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"\nDistributia datelor:")
    print(f"   Train: {len(X_train)} imagini ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} imagini ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} imagini ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   Numar clase: {len(np.unique(y))}\n")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        gei_info_train, gei_info_val, gei_info_test
    )
