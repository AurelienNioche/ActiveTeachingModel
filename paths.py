import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FIG_DIR = os.path.join(BASE_DIR, "fig")
os.makedirs(FIG_DIR, exist_ok=True)
BKP_DIR = os.path.join(BASE_DIR, "bkp")
os.makedirs(BKP_DIR, exist_ok=True)
