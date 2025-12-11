import os

# Căi directoare
INPUT_DIR = "D:\\Facultate\\Anul3\\Sem1\\PI-p\\CASIA-B"
OUTPUT_DIR = "GEI_Images"
MODEL_DIR = "models"

# Parametri imagini
TARGET_SIZE = (64, 64)

# Hiperparametri antrenare
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Random seed pentru reproducibilitate
RANDOM_SEED = 42

# Creează directoarele necesare
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)