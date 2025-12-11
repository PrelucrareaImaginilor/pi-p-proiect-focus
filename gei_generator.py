import numpy as np
import cv2
import os
from tqdm import tqdm
from config import INPUT_DIR, OUTPUT_DIR, TARGET_SIZE


def generate_gei(silhouette_folder, output_path):
    """
    Generează Gait Energy Image (GEI) din siluete.
    
    Args:
        silhouette_folder: Calea către folderul cu siluete
        output_path: Calea unde va fi salvat GEI
    
    Returns:
        True dacă GEI a fost generat cu succes
    """
    silhouette_files = sorted([f for f in os.listdir(silhouette_folder) 
                              if f.endswith('.png')])
    
    if not silhouette_files:
        return False
    
    silhouettes = []
    for file_name in silhouette_files:
        path = os.path.join(silhouette_folder, file_name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            img = cv2.resize(img, TARGET_SIZE)
            silhouettes.append(img.astype(np.float32) / 255.0)
    
    if not silhouettes:
        return False
    
    # Calculează GEI (media tuturor siluetelor)
    gei = np.mean(silhouettes, axis=0)
    
    # Salvează GEI
    cv2.imwrite(output_path, (gei * 255).astype(np.uint8))
    return True


def process_casia_b_dataset():
    """
    Procesează întregul dataset CASIA-B și generează GEI-uri.
    
    Structura CASIA-B: INPUT_DIR/subject_id/sequence_type/angle/frames
    
    Returns:
        Lista cu dicționare conținând info despre fiecare GEI generat
    """
    print("Incepe procesarea dataset-ului CASIA-B...")
    
    subject_folders = sorted([d for d in os.listdir(INPUT_DIR) 
                             if os.path.isdir(os.path.join(INPUT_DIR, d))])
    
    gei_info = []
    
    for subject_id in tqdm(subject_folders, desc="Procesare subiecti"):
        subject_path = os.path.join(INPUT_DIR, subject_id)
        
        for seq_type in os.listdir(subject_path):
            seq_path = os.path.join(subject_path, seq_type)
            if not os.path.isdir(seq_path):
                continue
            
            for angle in os.listdir(seq_path):
                angle_path = os.path.join(seq_path, angle)
                if not os.path.isdir(angle_path):
                    continue
                
                output_filename = f"{subject_id}_{seq_type}_{angle}_gei.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                if generate_gei(angle_path, output_path):
                    gei_info.append({
                        'path': output_path,
                        'subject_id': subject_id,
                        'seq_type': seq_type,
                        'angle': angle
                    })
    
    print(f"Generate {len(gei_info)} imagini GEI")
    
    # Salvează informațiile pentru refolosire
    np.save(os.path.join(OUTPUT_DIR, 'gei_info.npy'), gei_info)
    
    return gei_info