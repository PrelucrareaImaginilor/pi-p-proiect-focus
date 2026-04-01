import os
import numpy as np


def check_gei_exists():
    """Verifică dacă GEI-urile au fost deja generate."""
    from config import OUTPUT_DIR
    gei_info_path = os.path.join(OUTPUT_DIR, 'gei_info.npy')
    return os.path.exists(gei_info_path)


def print_banner(text):
    """Afișează un banner formatat."""
    length = len(text) + 4
    print("\n" + "=" * length)
    print(f"  {text}")
    print("=" * length + "\n")


def print_config():
    """Afișează configurația curentă."""
    from config import (TARGET_SIZE, BATCH_SIZE, EPOCHS, 
                        LEARNING_RATE, VALIDATION_SPLIT, TEST_SPLIT)
    
    print("Configuratie:")
    print(f"   - Dimensiune imagini: {TARGET_SIZE}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Epoci: {EPOCHS}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Validation split: {VALIDATION_SPLIT}")
    print(f"   - Test split: {TEST_SPLIT}")
    print()
