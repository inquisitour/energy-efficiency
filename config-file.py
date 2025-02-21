import os
from datetime import datetime

# Base Paths
BASE_DIR = os.path.abspath('.')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data Paths
RAW_PATH = os.path.join(DATA_DIR, 'raw/building_energy.csv')
PROCESSED_PATH = os.path.join(DATA_DIR, 'processed/processed_data.csv')
SYNTHETIC_PATH = os.path.join(DATA_DIR, 'synthetic/synthetic_data.csv')
PREPROCESSOR_PATH = os.path.join(DATA_DIR, 'processed/preprocessor.joblib')

# Model Hyperparameters
BATCH_SIZE = 256
LATENT_DIM = 128
SEQ_LENGTH = 24
EPOCHS = 500
LEARNING_RATE = 0.0001

# Privacy Parameters
NOISE_MULTIPLIER = 1.3
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
MAX_EPSILON = 10.0  # Maximum allowed privacy budget

# Feature Engineering
TEMPORAL_FEATURES = True
HANDLE_SPARSE = True

# Business Rules
ENERGY_ALERT = 75  # kWh
MIN_COMFORT_TEMP = 20  # °C
MAX_COMFORT_TEMP = 28  # °C
ENERGY_COST_PER_KWH = 0.30  # EUR
CO2_PER_KWH = 0.4  # kg CO2/kWh

# Validation
VALIDATION_SPLIT = 0.2
ANOMALY_ZSCORE = 3.0