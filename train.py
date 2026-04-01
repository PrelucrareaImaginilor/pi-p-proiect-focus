import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import GaitRecognitionCNN
import os
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE

def train_model(X_train, y_train, X_val, y_val, num_classes, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitRecognitionCNN(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    
    # Transformă numpy arrays în tensori
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).permute(0,3,1,2),
                                  torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32).permute(0,3,1,2),
                                torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Liste pentru istoric
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # === Training ===
        model.train()
        epoch_train_loss, correct = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        
        train_acc = correct / len(train_dataset)
        train_losses.append(epoch_train_loss / len(train_loader))
        train_accs.append(train_acc)
        
        # === Validation ===
        model.eval()
        epoch_val_loss, val_correct = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        val_acc = val_correct / len(val_dataset)
        best_val_acc = val_acc
        val_losses.append(epoch_val_loss / len(val_loader))
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.3f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.3f}")
        
    
    save_checkpoint(model, epochs)
    
    return model, train_losses, val_losses, train_accs, val_accs

def save_checkpoint(model, epoch, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, f"best_model_epoch{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint salvat: {checkpoint_path}")