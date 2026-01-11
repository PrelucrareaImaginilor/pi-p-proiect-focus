import cv2
import numpy as np
import torch

def extract_silhouettes_from_video(video_path, size=(64, 64), min_area=3000):
    cap = cv2.VideoCapture(video_path)
    silhouettes = []

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=40,
        detectShadows=False
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)

        fgmask = cv2.medianBlur(fgmask, 5)
        _, binary = cv2.threshold(fgmask, 180, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < min_area or area > 50000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        silhouette = binary[y:y+h, x:x+w]

        # normalize aspect ratio
        h0, w0 = silhouette.shape
        scale = min(size[0] / h0, size[1] / w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(silhouette, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        padded = np.zeros(size, dtype=np.uint8)
        y_off = (size[0] - new_h) // 2
        x_off = (size[1] - new_w) // 2
        padded[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        silhouettes.append((padded / 255.0).astype(np.float32))

    cap.release()
    return np.array(silhouettes)



def generate_gei_from_silhouettes(silhouettes):
    silhouettes = silhouettes.astype(np.float32)
    gei = np.mean(silhouettes, axis=0)
    gei = np.expand_dims(gei, axis=-1)
    return gei


def generate_gei_from_video(video_path):
    silhouettes = extract_silhouettes_from_video(video_path)

    if len(silhouettes) == 0:
        raise ValueError("No silhouettes could be extracted from the video.")

    return generate_gei_from_silhouettes(silhouettes)

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
