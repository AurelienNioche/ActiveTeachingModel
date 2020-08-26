import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = os.path.join(BASE_DIR, "fig")

BKP_DIR = os.path.join(BASE_DIR, "bkp")

JSON_DIR = os.path.join(BASE_DIR, "config")

CONFIG_CLUSTER_DIR = os.path.join(JSON_DIR, "triton")

DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_CLUSTER_DIR = os.path.join(DATA_DIR, "triton")

for directory in FIG_DIR, BKP_DIR, JSON_DIR, CONFIG_CLUSTER_DIR, DATA_DIR:
    os.makedirs(directory, exist_ok=True)
