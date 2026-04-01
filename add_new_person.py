import os
import numpy as np
import cv2
from video_inference import generate_gei_from_video

GEI_DIR = "GEI_Images"
GEI_INFO_PATH = os.path.join(GEI_DIR, "gei_info.npy")

def add_split_videos_to_dataset(splits_folder):
    os.makedirs(GEI_DIR, exist_ok=True)

    # Load existing gei_info or create new
    if os.path.exists(GEI_INFO_PATH):
        gei_info = np.load(GEI_INFO_PATH, allow_pickle=True).tolist()
    else:
        gei_info = []

    for file in os.listdir(splits_folder):
        if not file.endswith(".mp4"):
            continue

        video_path = os.path.join(splits_folder, file)
        print(f"Processing {video_path}")

        # Generate GEI
        gei = generate_gei_from_video(video_path)
        gei_img = (gei * 255).astype(np.uint8)

        # Save GEI in GEI_Images
        gei_name = file.replace(".mp4", ".png")
        gei_path = os.path.join(GEI_DIR, gei_name)
        cv2.imwrite(gei_path, gei_img)

        # Parse CASIA-B style name
        subject, rest = file.split("_", 1)
        condition, seq_angle = rest.split("-", 1)
        seq, angle_mp4 = seq_angle.split("_", 1)
        angle = angle_mp4.replace(".mp4", "")

        # Add entry to gei_info
        gei_info.append({
            "path": gei_path,
            "subject": subject,
            "condition": condition,
            "sequence": seq,
            "angle": angle
        })

    # 🔥 Remove duplicates based on "path"
    unique = {}
    for entry in gei_info:
        unique[entry["path"]] = entry

    gei_info = list(unique.values())

    # Save updated gei_info
    np.save(GEI_INFO_PATH, np.array(gei_info, dtype=object))
    print(f"Dataset updated successfully! Total GEIs: {len(gei_info)}")

add_split_videos_to_dataset("splits")
