"""Încărcare și pregătire date pentru antrenare."""

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import OUTPUT_DIR, RANDOM_SEED, VALIDATION_SPLIT, TEST_SPLIT
import os
import re


def parse_gei_filename(filename, folder="GEI_Images"):
    """
    Exemplu: '001_bg-90_000_gei.png'
    Returnează: subject_id, condition, angle, frame, path
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r"(\d{3})_(\w+)-(\d{1,3})_(\d{3})_gei", name)
    if match:
        subject, condition, angle, frame = match.groups()
        return {
            "subject_id": subject,
            "condition": condition,   # nm, bg, cl
            "angle": int(angle),      # unghi real (0, 18, 36, ..., 180)
            "frame": frame,
            "path": os.path.join(folder, filename)
        }
    return None


def load_gei_info(folder="GEI_Images", save=True):
    """
    Parcurge folderul GEI_Images și extrage metadatele din numele fișierelor.
    Returnează o listă de dict-uri cu subject_id, condition, angle, frame și path.
    """
    gei_info = []
    for file in os.listdir(folder):
        if file.endswith("_gei.png"):
            info = parse_gei_filename(file, folder)
            if info:
                gei_info.append(info)
    print(f"Incarcate {len(gei_info)} fisiere GEI din {folder}")

    if save:
        np.save(os.path.join(folder, "gei_info_new.npy"), gei_info)
        print(f"gei_info.npy salvat in {folder}")

    return gei_info


def load_gei_dataset(gei_info, angle_filter=None, condition_filter=None):
    """
    Încarcă GEI-urile generate și le pregătește pentru antrenare.
    
    Args:
        gei_info: Lista cu informații despre GEI-uri
        angle_filter: Listă cu unghiuri de filtrat (ex: [0, 90, 180])
        condition_filter: Listă cu condiții de filtrat (ex: ['nm', 'bg', 'cl'])
    
    Returns:
        Tuple (images, labels_encoded, label_encoder)
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
        if img is not None:
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
    Împarte dataset-ul în train, validation și test, inclusiv metadatele gei_info.
    
    Args:
        X: Array cu imagini
        y: Array cu etichete
        gei_info: Lista cu metadate pentru fiecare imagine
    
    Returns:
        Tuple (X_train, X_val, X_test, y_train, y_val, y_test,
               gei_info_train, gei_info_val, gei_info_test)
    """
    total_test_size = VALIDATION_SPLIT + TEST_SPLIT
    
    X_train, X_temp, y_train, y_temp, gei_info_train, gei_info_temp = train_test_split(
        X, y, gei_info, test_size=total_test_size, random_state=RANDOM_SEED, stratify=y
    )
    
    val_size_adjusted = VALIDATION_SPLIT / total_test_size
    X_val, X_test, y_val, y_test, gei_info_val, gei_info_test = train_test_split(
        X_temp, y_temp, gei_info_temp, test_size=(1 - val_size_adjusted),
        random_state=RANDOM_SEED, stratify=y_temp
    )
    
    print(f"\nDistributia datelor:")
    print(f"   Train: {len(X_train)} imagini ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} imagini ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test)} imagini ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   Numar clase: {len(np.unique(y))}\n")
    
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            gei_info_train, gei_info_val, gei_info_test)
