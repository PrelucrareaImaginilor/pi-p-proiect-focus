import torch
import numpy as np

import matplotlib.pyplot as plt

import os

def save_results(metrics, label_encoder, model_dir, angle_results=None, condition_results=None):
    import os
    results_path = os.path.join(model_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write("=== Rezultate evaluare ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        
        if angle_results:
            f.write("\n=== Acuratete pe unghiuri ===\n")
            for angle, acc in angle_results.items():
                f.write(f"Unghi {int(angle)}°: {acc:.3f}\n")
        
        if condition_results:
            f.write("\n=== Acuratete pe conditii ===\n")
            for cond, acc in condition_results.items():
                f.write(f"{cond.upper()}: {acc:.3f}\n")
    print(f"Rezultatele au fost salvate în {results_path}")



def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path="training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    
    # Accuracy
    ax1.plot(train_accs, label="Train", linewidth=2)
    ax1.plot(val_accs, label="Validation", linewidth=2)
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(train_losses, label="Train", linewidth=2)
    ax2.plot(val_losses, label="Validation", linewidth=2)
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Grafice salvate in {save_path}")


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluează modelul pe setul de test.
    
    Args:
        model: Model PyTorch antrenat
        X_test, y_test: Date de test (numpy arrays)
        label_encoder: LabelEncoder folosit
    
    Returns:
        Dict cu metrici
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Transformă numpy arrays în tensori
    test_inputs = torch.tensor(X_test, dtype=torch.float32).permute(0,3,1,2).to(device)
    test_labels = torch.tensor(y_test, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(test_inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
    
    # Acuratețe
    accuracy = (preds == test_labels).float().mean().item()
    
    # Top-k accuracy
    def top_k_accuracy(probs, labels, k=5):
        topk = torch.topk(probs, k, dim=1).indices
        correct = topk.eq(labels.view(-1,1)).sum().item()
        return correct / len(labels)
    
    top5_accuracy = top_k_accuracy(probs, test_labels, k=5)
    top10_accuracy = top_k_accuracy(probs, test_labels, k=10)
    
    print(f"Acuratete: {accuracy*100:.2f}%")
    print(f"Top-5 Acuratete: {top5_accuracy*100:.2f}%")
    print(f"Top-10 Acuratete: {top10_accuracy*100:.2f}%")
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'top10_accuracy': top10_accuracy,
        'predictions': preds.cpu().numpy(),
        'true_labels': test_labels.cpu().numpy()
    }

def evaluate_by_angle(model, X_test, y_test, gei_info, device="cuda"):
    """
    Calculează acuratețea pe fiecare unghi (0, 18, ..., 180).
    """
    model.eval()
    results = {}
    with torch.no_grad():
        for angle in sorted(np.unique([info['angle'] for info in gei_info])):
            print("DEBUG angle:", angle)
            idx = [i for i, info in enumerate(gei_info) if info['angle'] == angle]
            if not idx: 
                continue
            X_angle = torch.tensor(X_test[idx], dtype=torch.float32).permute(0,3,1,2).to(device)
            y_angle = torch.tensor(y_test[idx], dtype=torch.long).to(device)

            outputs = model(X_angle)
            preds = outputs.argmax(1)
            acc = (preds == y_angle).sum().item() / len(y_angle)
            results[angle] = acc
    return results


def evaluate_by_condition(model, X_test, y_test, gei_info, device="cuda"):
    """
    Calculează acuratetea pe fiecare conditie (nm, bg, cl).
    """
    model.eval()
    results = {}
    with torch.no_grad():
        for cond in sorted(np.unique([info['condition'] for info in gei_info])):
            idx = [i for i, info in enumerate(gei_info) if info['condition'] == cond]
            if not idx: 
                continue
            X_cond = torch.tensor(X_test[idx], dtype=torch.float32).permute(0,3,1,2).to(device)
            y_cond = torch.tensor(y_test[idx], dtype=torch.long).to(device)

            outputs = model(X_cond)
            preds = outputs.argmax(1)
            acc = (preds == y_cond).sum().item() / len(y_cond)
            results[cond] = acc
    return results
