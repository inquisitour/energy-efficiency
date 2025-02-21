# Cell 1: Load Data with Validation
import pandas as pd
import numpy as np
from config import *
from sklearn.exceptions import NotFittedError

def validate_data(df):
    """Validate data quality and constraints"""
    assert not df['energy_kWh'].isnull().any(), "Missing energy values"
    assert df['energy_kWh'].min() >= 0, "Negative energy values"
    assert 0 <= df['occupancy_pct'].max() <= 1, "Invalid occupancy percentage"
    assert -50 <= df['temperature_C'].mean() <= 50, "Suspicious temperature values"
    return True

try:
    raw_df = pd.read_csv(RAW_PATH, parse_dates=['timestamp'])
    print("Successfully loaded organizer's dataset")
except FileNotFoundError:
    print("WARNING: Generating fallback sample data")
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start="2024-01-01", periods=8760, freq='H')
    raw_df = pd.DataFrame({
        'timestamp': dates,
        'energy_kWh': np.clip(np.random.normal(50, 15, 8760), 0, None),
        'temperature_C': np.random.uniform(10, 30, 8760),
        'occupancy_pct': np.random.beta(2, 5, 8760),
        'building_type': np.random.choice(['Office', 'Residential'], 8760)
    })
    raw_df.to_csv(RAW_PATH, index=False)

# Validate data
try:
    validate_data(raw_df)
    print("Data validation passed")
except AssertionError as e:
    print(f"Data validation failed: {e}")
    
# Optimize memory usage
raw_df['building_type'] = raw_df['building_type'].astype('category')

# Cell 2: Feature Engineering
if TEMPORAL_FEATURES:
    # Time-based features
    raw_df['hour'] = raw_df['timestamp'].dt.hour
    raw_df['day_sin'] = np.sin(2 * np.pi * raw_df['timestamp'].dt.dayofyear/365)
    raw_df['day_cos'] = np.cos(2 * np.pi * raw_df['timestamp'].dt.dayofyear/365)
    
    # Handle missing values with time-based fill
    if HANDLE_SPARSE:
        for col in ['occupancy_pct', 'temperature_C']:
            if raw_df[col].isnull().any():
                raw_df[col] = raw_df[col].fillna(
                    raw_df.groupby(raw_df['timestamp'].dt.hour)[col].transform('mean')
                )

# Cell 3: Preprocessing Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Define feature groups
numeric_features = ['energy_kWh', 'temperature_C', 'occupancy_pct']
if TEMPORAL_FEATURES:
    numeric_features.extend(['hour', 'day_sin', 'day_cos'])
categorical_features = ['building_type']

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Load or fit preprocessor
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Loaded existing preprocessor")
except FileNotFoundError:
    print("Fitting new preprocessor")
    preprocessor.fit(raw_df)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

# Transform and save data
processed_data = preprocessor.transform(raw_df)
feature_names = (
    numeric_features +
    [f"{feat}_{val}" for feat, vals in 
     zip(categorical_features, preprocessor.named_transformers_['cat'].categories_) 
     for val in vals]
)

pd.DataFrame(
    processed_data,
    columns=feature_names
).to_csv(PROCESSED_PATH, index=False)

print(f"Saved processed data with {len(feature_names)} features")

# Cell 4: Data Summary
print("\nDataset Summary:")
print(f"Time Range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")
print(f"Total Records: {len(raw_df):,}")
print("\nFeature Statistics:")
print(raw_df[numeric_features].describe())