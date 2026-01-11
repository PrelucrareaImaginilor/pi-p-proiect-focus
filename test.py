import numpy as np
from collections import Counter

info = np.load("GEI_Images/gei_info.npy", allow_pickle=True)

# extragem toate subject_id-urile
subjects = [x["subject_id"] for x in info]

# numărăm câte GEI-uri are fiecare subiect
counts = Counter(subjects)

print("Număr GEI-uri per subiect:")
for s, c in sorted(counts.items(), key=lambda x: int(x[0])):
    print(f"{s}: {c}")

print("\nSubiecți cu mai puțin de 2 GEI-uri:")
for s, c in counts.items():
    if c < 2:
        print(f"{s}: {c}")
