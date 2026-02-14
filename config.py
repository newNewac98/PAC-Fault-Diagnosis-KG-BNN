"""
Central configuration for the KG + BNN fault diagnosis project.
All hyperparameters and experimental settings are defined here.
"""

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data"
SEED = 42

NUM_SENSOR_FEATURES = 20       
NUM_KG_FEATURES = 10           
NUM_FEATURES = NUM_SENSOR_FEATURES + NUM_KG_FEATURES   # 30
NUM_CLASSES = 4                # PAC fault types

FAULT_NAMES = [
    "Refrigerant Leakage",
    "Excessive Refrigerant",
    "Filter Blockage",
    "Poor Indoor Ventilation",
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


ALPHA = 1e-3       # weight prior precision
BETA = 1e-1        # noise / likelihood precision


LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 200

N_FOLDS = 5       
