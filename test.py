import numpy as np

data = np.load("GEI_Images/gei_info.npy", allow_pickle=True)
angles = sorted(set([d["angle"] for d in data]))
print("Unghiuri gasite:", angles)
print("Numar unghiuri:", len(angles))
