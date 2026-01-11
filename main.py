import numpy as np
import torch
import argparse
import os

from train import train_model
from evaluate import (
    evaluate_by_angle,
    evaluate_by_condition,
    evaluate_model,
    plot_training_history,
    save_results
)
from data_loader import load_gei_dataset, split_dataset, load_gei_info
from utils import check_gei_exists, print_banner, print_config
from gei_generator import process_casia_b_dataset
from config import OUTPUT_DIR, MODEL_DIR, EPOCHS
from model import GaitRecognitionCNN
from video_inference import evaluate_video


TRAIN = False   # True = train, False = load saved model


def main():

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to input video for inference")
    parser.add_argument("--add-person", type=str, help="Name of new person to add")
    parser.add_argument("--videos", type=str, help="Folder with videos for new person")
    args = parser.parse_args()

    print_banner("GAIT RECOGNITION SYSTEM - CASIA-B + GEI (PyTorch)")
    print_config()

    # ============================================================
    # NORMAL PIPELINE (TRAIN / LOAD / EVALUATE)
    # ============================================================

    print("Loading GEI metadata...")
    gei_info = load_gei_info()

    if len(gei_info) == 0:
        print("No GEI metadata found. Generating CASIA-B GEIs...")
        process_casia_b_dataset()
        gei_info = load_gei_info()

    X, y, label_encoder = load_gei_dataset(gei_info, angle_filter=None)

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        gei_info_train, gei_info_val, gei_info_test
    ) = split_dataset(X, y, gei_info)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if TRAIN:
        print("Training model from scratch...")
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            X_train, y_train, X_val, y_val, num_classes=len(np.unique(y))
        )

        plot_training_history(
            train_losses, val_losses, train_accs, val_accs,
            save_path=f"{MODEL_DIR}/training_history.png"
        )

    else:
        print("Loading saved model...")
        model = GaitRecognitionCNN(num_classes=len(np.unique(y))).to(device)
        state_dict = torch.load(
            f"{MODEL_DIR}/best_model_epoch100.pth",
            map_location=device
        )
        model.load_state_dict(state_dict)
        model.eval()

    # ============================================================
    # MODE: VIDEO INFERENCE
    # ============================================================
    if args.video:
        print(f"\n=== Running inference on video: {args.video} ===\n")
        subject, probs = evaluate_video(model, args.video, label_encoder, device=device)
        print(f"Predicted subject: {subject}")
        print("\nInference complete.")
        return

    # ============================================================
    # NORMAL EVALUATION
    # ============================================================
    print("\nEvaluating model on test set...")

    metrics = evaluate_model(model, X_test, y_test, label_encoder)
    angle_results = evaluate_by_angle(model, X_test, y_test, gei_info_test)
    condition_results = evaluate_by_condition(model, X_test, y_test, gei_info_test)

    save_results(
        metrics,
        label_encoder,
        model_dir=MODEL_DIR,
        angle_results=angle_results,
        condition_results=condition_results
    )

    print_banner("PROCESS COMPLETE!")
    print(f"Model and results saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
