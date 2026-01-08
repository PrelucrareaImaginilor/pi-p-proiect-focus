import os
import json
import cv2
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder

from video_inference import generate_gei_from_video
from model import GaitRecognitionCNN
from data_loader import load_gei_dataset
from config import MODEL_DIR


# ============================================================
# 1. Generate GEIs for a new person
# ============================================================

def generate_geis_for_new_person(video_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for video in os.listdir(video_folder):
        if not video.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        video_path = os.path.join(video_folder, video)
        print(f"Processing video: {video_path}")

        gei = generate_gei_from_video(video_path)
        gei_img = (gei * 255).astype(np.uint8)

        save_path = os.path.join(output_folder, video.replace(".mp4", ".png"))
        cv2.imwrite(save_path, gei_img)

    print("GEI generation complete.")


# ============================================================
# 2. Update gei_info.json with new entries
# ============================================================

def update_gei_info(gei_info_path, new_person_folder, new_label):
    if os.path.exists(gei_info_path):
        with open(gei_info_path, "r") as f:
            gei_info = json.load(f)
    else:
        gei_info = []

    for img in os.listdir(new_person_folder):
        if img.endswith(".png"):
            gei_info.append({
                "path": os.path.join(new_person_folder, img),
                "label": new_label,
                "angle": "090",
                "condition": "nm"
            })

    with open(gei_info_path, "w") as f:
        json.dump(gei_info, f, indent=4)

    print("gei_info.json updated.")


# ============================================================
# 3. Retrain only the final layer
# ============================================================

def retrain_last_layer(model_path, gei_info_path):
    print("Loading updated dataset...")

    # Load dataset with new person included
    X, y, label_encoder = load_gei_dataset(gei_info_path)

    num_classes = len(label_encoder.classes_)
    print(f"New number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load old model
    model = GaitRecognitionCNN(num_classes=num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # Freeze all layers except final FC
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(128, num_classes).to(device)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Retraining last layer...")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/10 - Loss: {loss.item():.4f}")

    # Save updated model
    save_path = os.path.join(MODEL_DIR, "model_with_new_person.pth")
    torch.save(model.state_dict(), save_path)

    print(f"Model saved to: {save_path}")


# ============================================================
# 4. Main pipeline
# ============================================================

def add_new_person(person_name, video_folder):
    new_person_folder = f"data/new_subjects/{person_name}"
    gei_info_path = "gei_info.json"

    print("=== Step 1: Generating GEIs ===")
    generate_geis_for_new_person(video_folder, new_person_folder)

    print("=== Step 2: Updating gei_info.json ===")
    update_gei_info(gei_info_path, new_person_folder, person_name)

    print("=== Step 3: Retraining last layer ===")
    retrain_last_layer(f"{MODEL_DIR}/best_model_epoch50.pth", gei_info_path)

    print("=== Done! New person added successfully. ===")


if __name__ == "__main__":
    # Example usage:
    # add_new_person("person_001", "videos/person_001")
    pass
