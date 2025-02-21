# Cell 1: Load and Validate Data
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from config import *

try:
    real = pd.read_csv(RAW_PATH, parse_dates=['timestamp'])
    synth = pd.read_csv(SYNTHETIC_PATH, parse_dates=['timestamp'])
    print("Successfully loaded real and synthetic datasets")
except FileNotFoundError:
    print("CRITICAL: Loading backup validation data")
    real = pd.read_csv(os.path.join(DATA_DIR, 'backups/raw_backup.csv'))
    synth = pd.read_csv(os.path.join(DATA_DIR, 'backups/synth_backup.csv'))

# Validate data alignment
assert set(real.columns) == set(synth.columns), "Column mismatch"
assert len(real) == len(synth), "Length mismatch"

# Cell 2: Distribution Analysis
def analyze_distributions(real_data, synth_data, features):
    results = {}
    for feature in features:
        # Calculate statistical distances
        w_dist = wasserstein_distance(real_data[feature], synth_data[feature])
        ks_stat, p_value = ks_2samp(real_data[feature], synth_data[feature])
        
        # Store results
        results[feature] = {
            'wasserstein': w_dist,
            'ks_pvalue': p_value,
            'real_mean': real_data[feature].mean(),
            'synth_mean': synth_data[feature].mean(),
            'real_std': real_data[feature].std(),
            'synth_std': synth_data[feature].std()
        }
    return results

numerical_features = ['energy_kWh', 'temperature_C', 'occupancy_pct']
dist_analysis = analyze_distributions(real, synth, numerical_features)

# Print analysis results
print("\nDistribution Analysis:")
for feature, metrics in dist_analysis.items():
    print(f"\n{feature}:")
    print(f"  Wasserstein Distance: {metrics['wasserstein']:.4f}")
    print(f"  KS-test p-value: {metrics['ks_pvalue']:.4f}")
    print(f"  Real  μ={metrics['real_mean']:.2f}, σ={metrics['real_std']:.2f}")
    print(f"  Synth μ={metrics['synth_mean']:.2f}, σ={metrics['synth_std']:.2f}")

# Cell 3: Temporal Pattern Validation
import plotly.express as px
import plotly.graph_objects as go

def plot_temporal_comparison(real_data, synth_data, feature):
    fig = go.Figure()
    
    # Add real data
    fig.add_trace(go.Scatter(
        x=real_data['timestamp'],
        y=real_data[feature],
        name='Real Data',
        line=dict(color='firebrick', width=1)
    ))
    
    # Add synthetic data
    fig.add_trace(go.Scatter(
        x=synth_data['timestamp'],
        y=synth_data[feature],
        name='Synthetic Data',
        line=dict(color='royalblue', width=1)
    ))
    
    fig.update_layout(
        title=f'Real vs Synthetic {feature} Patterns',
        xaxis_title='Time',
        yaxis_title=feature,
        showlegend=True
    )
    return fig

# Plot energy patterns
energy_fig = plot_temporal_comparison(real, synth, 'energy_kWh')
energy_fig.show()

# Cell 4: Anomaly Detection
def detect_anomalies(df, column, z_threshold=ANOMALY_ZSCORE):
    """Detect anomalies using Z-score method"""
    mean = df[column].mean()
    std = df[column].std()
    z_scores = abs((df[column] - mean) / std)
    return df[z_scores > z_threshold]

# Find anomalies in synthetic data
energy_anomalies = detect_anomalies(synth, 'energy_kWh')
print(f"\nAnomaly Detection:")
print(f"Found {len(energy_anomalies)} anomalies in synthetic energy data")

if len(energy_anomalies) > 0:
    print("\nTop 5 Anomalies:")
    display(energy_anomalies.nlargest(5, 'energy_kWh')[
        ['timestamp', 'energy_kWh', 'temperature_C', 'occupancy_pct']
    ])

# Cell 5: Privacy Preservation Validation
def validate_privacy(real_data, synth_data, epsilon):
    """Validate privacy preservation"""
    # Check if privacy budget is within bounds
    if epsilon > MAX_EPSILON:
        print(f"WARNING: Privacy budget (ε={epsilon:.2f}) exceeds maximum ({MAX_EPSILON})")
    
    # Check for exact duplicates
    duplicates = pd.merge(real_data, synth_data, how='inner')
    if len(duplicates) > 0:
        print(f"WARNING: Found {len(duplicates)} exact matches between real and synthetic data")
    
    return {
        'privacy_budget': epsilon,
        'exact_matches': len(duplicates),
        'budget_exceeded': epsilon > MAX_EPSILON
    }

privacy_results = validate_privacy(real, synth, synth.epsilon)
print("\nPrivacy Validation:")
print(f"Privacy Budget (ε): {privacy_results['privacy_budget']:.2f}")
print(f"Exact Matches: {privacy_results['exact_matches']}")
print(f"Budget Status: {'EXCEEDED' if privacy_results['budget_exceeded'] else 'Within Limits'}")

# Save validation results
validation_results = {
    'distribution_analysis': dist_analysis,
    'anomalies': len(energy_anomalies),
    'privacy': privacy_results
}

import json
with open(os.path.join(DATA_DIR, 'validation_results.json'), 'w') as f:
    json.dump(validation_results, f, indent=2)
