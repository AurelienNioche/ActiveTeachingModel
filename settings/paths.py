import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = os.path.join(BASE_DIR, "fig")
BKP_DIR = os.path.join(BASE_DIR, "bkp")

for folder in FIG_DIR, BKP_DIR:
    os.makedirs(folder, exist_ok=True)
