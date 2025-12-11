import numpy as np
import torch
from train import train_model
from evaluate import evaluate_by_angle, evaluate_by_condition, evaluate_model, plot_training_history, save_results
from data_loader import load_gei_dataset, split_dataset, load_gei_info
from utils import check_gei_exists, print_banner, print_config
from gei_generator import process_casia_b_dataset
from config import OUTPUT_DIR, MODEL_DIR, EPOCHS
from model import GaitRecognitionCNN

# === FLAG pentru control ===
TRAIN = False   # True = antrenează, False = încarcă modelul salvat

def main():
    print_banner("SISTEM DE RECUNOASTERE A MERSULUI - CASIA-B + GEI (PyTorch)")
    print_config()
    
    # Pasul 1: Generează sau încarcă GEI-uri
    gei_info = load_gei_info()
    if not check_gei_exists():
        print("Generare GEI-uri...")
        gei_info = process_casia_b_dataset()
    else:
        print("GEI-urile exista deja, le incarcam...")
        gei_info = load_gei_info()
    
    # Pasul 2: Încarcă dataset-ul
    X, y, label_encoder = load_gei_dataset(gei_info, angle_filter=None)
    
    # Pasul 3: Împarte dataset-ul
    X_train, X_val, X_test, y_train, y_val, y_test, gei_info_train, gei_info_val, gei_info_test = split_dataset(X, y, gei_info)
    
    # Pasul 4: Alegere între train sau load
    if TRAIN:
        print("Antrenam modelul de la zero...")
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            X_train, y_train, X_val, y_val, num_classes=len(np.unique(y))
        )
        # Vizualizare istoric
        plot_training_history(train_losses, val_losses, train_accs, val_accs,
                              save_path=f"{MODEL_DIR}/training_history.png")
    else:
        print("Incarcam modelul salvat...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GaitRecognitionCNN(num_classes=len(np.unique(y))).to(device)
        model.load_state_dict(torch.load(f"{MODEL_DIR}/best_model_epoch50.pth", map_location=device))
        model.eval()
    
    # Pasul 5: Evaluează modelul
    metrics = evaluate_model(model, X_test, y_test, label_encoder)
    angle_results = evaluate_by_angle(model, X_test, y_test, gei_info_test)
    condition_results = evaluate_by_condition(model, X_test, y_test, gei_info_test)
    
    # Pasul 6: Salvează rezultatele
    save_results(metrics, label_encoder, model_dir=MODEL_DIR,angle_results=angle_results, condition_results=condition_results)
    
    print_banner("PROCESARE COMPLETA!")
    print(f"Modelul si rezultatele sunt salvate in: {MODEL_DIR}")

if __name__ == "__main__":
    main()
