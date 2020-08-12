import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = os.path.join(BASE_DIR, "fig")
BKP_DIR = os.path.join(BASE_DIR, "bkp")

for directory in FIG_DIR, BKP_DIR:
    os.makedirs(directory, exist_ok=True)
