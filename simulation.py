# Cell 1: Load Dependencies and Data
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from config import *

# Load synthetic data
synth = pd.read_csv(SYNTHETIC_PATH, parse_dates=['timestamp'])
print("Loaded synthetic data:", synth.shape)

# Cell 2: Energy Prediction Model
def create_model(input_shape):
    """Create energy prediction model with regularization"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Cell 3: Data Preparation
# Prepare features
X = synth[['temperature_C', 'occupancy_pct', 'hour']]
y = synth['energy_kWh']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=VALIDATION_SPLIT, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Cell 4: Model Training
# Create and train model
model = create_model((3,))
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=1
)

# Plot training history
fig = px.line(history.history, 
              title='Model Training History',
              labels={'value': 'Metric Value', 'index': 'Epoch'})
fig.show()

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE: {test_mae:.2f} kWh")

# Cell 5: Comfort and Cost Functions
def calculate_comfort_penalty(temp, occ):
    """Calculate comfort penalty based on temperature and occupancy"""
    penalty = 0
    
    # Penalty for low temperature with high occupancy
    if temp < MIN_COMFORT_TEMP and occ > 0.5:
        penalty += 100 * (MIN_COMFORT_TEMP - temp)
    
    # Penalty for high temperature with any occupancy
    if temp > MAX_COMFORT_TEMP and occ > 0.3:
        penalty += 100 * (temp - MAX_COMFORT_TEMP)
    
    return penalty

def calculate_total_cost(energy_kwh, temp, occ):
    """Calculate total cost including energy and comfort penalty"""
    energy_cost = energy_kwh * ENERGY_COST_PER_KWH
    comfort_penalty = calculate_comfort_penalty(temp, occ)
    return energy_cost + comfort_penalty

# Cell 6: Optimization Function
def optimize_energy():
    """Find optimal temperature and occupancy settings"""
    best = {
        'energy': float('inf'),
        'cost': float('inf'),
        'temp': None,
        'occ': None,
        'total_score': float('inf')
    }
    
    # Grid search
    temps = np.linspace(18, 30, 50)
    occs = np.linspace(0.1, 0.9, 50)
    
    results = []
    for temp in temps:
        for occ in occs:
            # Predict energy consumption
            pred = model.predict([[temp, occ, 14]], verbose=0)[0][0]  # 2PM reference
            
            # Calculate costs
            energy_cost = pred * ENERGY_COST_PER_KWH
            comfort_penalty = calculate_comfort_penalty(temp, occ)
            total_score = energy_cost + comfort_penalty
            
            results.append({
                'temp': temp,
                'occ': occ,
                'energy': pred,
                'cost': energy_cost,
                'penalty': comfort_penalty,
                'total_score': total_score
            })
            
            # Update best settings
            if total_score < best['total_score']:
                best.update({
                    'energy': pred,
                    'cost': energy_cost,
                    'temp': temp,
                    'occ': occ,
                    'total_score': total_score
                })
    
    return best, pd.DataFrame(results)

# Cell 7: Run Optimization
optimal, all_results = optimize_energy()

# Print results
print("\nOptimal Settings Found:")
print(f"Temperature: {optimal['temp']:.1f}°C")
print(f"Occupancy: {optimal['occ']:.1%}")
print(f"Expected Energy: {optimal['energy']:.1f} kWh")
print(f"Daily Cost: €{optimal['cost']:.2f}")

# Cell 8: Visualize Results
# Create optimization landscape heatmap
fig = go.Figure(data=go.Heatmap(
    x=sorted(all_results['temp'].unique()),
    y=sorted(all_results['occ'].unique()),
    z=all_results.pivot_table(
        values='total_score',
        index='occ',
        columns='temp'
    ).values,
    colorscale='Viridis',
    colorbar=dict(title='Total Cost (€)')
))

fig.update_layout(
    title='Energy Optimization Landscape',
    xaxis_title='Temperature (°C)',
    yaxis_title='Occupancy (%)',
    width=800,
    height=600
)

# Add optimal point
fig.add_trace(go.Scatter(
    x=[optimal['temp']],
    y=[optimal['occ']],
    mode='markers',
    marker=dict(
        color='red',
        size=10,
        symbol='star'
    ),
    name='Optimal Point'
))

fig.show()

# Cell 9: Calculate and Save Results
def calculate_annual_savings(current_energy, optimal_energy):
    """Calculate annual energy and cost savings"""
    daily_energy_saving = current_energy - optimal_energy
    annual_energy_saving = daily_energy_saving * 365
    
    cost_saving = annual_energy_saving * ENERGY_COST_PER_KWH
    co2_saving = annual_energy_saving * CO2_PER_KWH
    
    return {
        'energy': annual_energy_saving,
        'cost': cost_saving,
        'co2': co2_saving
    }

# Calculate savings
current_avg_energy = synth['energy_kWh'].mean()
savings = calculate_annual_savings(current_avg_energy, optimal['energy'])

print("\nProjected Annual Savings:")
print(f"Energy: {savings['energy']:,.1f} kWh")
print(f"Cost: €{savings['cost']:,.2f}")
print(f"CO2 Reduction: {savings['co2']:,.1f} kg")

# Save results
results_dict = {
    'optimal_settings': optimal,
    'annual_savings': savings,
    'model_performance': {
        'mae': float(test_mae),
        'final_epoch': len(history.history['loss'])
    }
}

import json
with open(os.path.join(DATA_DIR, 'optimization_results.json'), 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\nResults saved to 'optimization_results.json'")
