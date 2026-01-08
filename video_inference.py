import cv2
import numpy as np
import torch

# ============================
# 1. Extract silhouettes from video
# ============================

def extract_silhouettes_from_video(video_path, size=(64, 64)):
    """
    Extract silhouettes from a video using background subtraction.
    Returns an array of binary silhouettes 64x64.
    """
    cap = cv2.VideoCapture(video_path)
    silhouettes = []

    # Background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)

        # Binarization and noise cleaning
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.medianBlur(thresh, 5)

        # Resize to 64x64
        resized = cv2.resize(thresh, size)
        silhouettes.append(resized)

    cap.release()

    silhouettes = np.array(silhouettes)
    return silhouettes


# ============================
# 2. Generate GEI from silhouettes
# ============================

def generate_gei_from_silhouettes(silhouettes):
    """
    Create GEI by averaging silhouettes.
    """
    silhouettes = silhouettes.astype(np.float32) / 255.0
    gei = np.mean(silhouettes, axis=0)
    gei = np.expand_dims(gei, axis=-1)  # (H, W, 1)
    return gei


# ============================
# 3. Video -> GEI
# ============================

def generate_gei_from_video(video_path):
    """
    Full pipeline: video -> silhouettes -> GEI.
    """
    silhouettes = extract_silhouettes_from_video(video_path)

    if len(silhouettes) == 0:
        raise ValueError("No silhouettes could be extracted from the video.")

    gei = generate_gei_from_silhouettes(silhouettes)
    return gei


# ============================
# 4. Evaluate model on video
# ============================

def evaluate_video(model, video_path, label_encoder, device="cuda"):
    """
    Run the model on a video and return the prediction.
    """
    gei = generate_gei_from_video(video_path)

    # Prepare for PyTorch
    gei = gei.astype(np.float32)
    gei = np.expand_dims(gei, axis=0)  # batch
    gei = np.transpose(gei, (0, 3, 1, 2))  # NCHW

    tensor = torch.tensor(gei).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    subject_id = label_encoder.inverse_transform([pred])[0]

    return subject_id, probs.cpu().numpy()
