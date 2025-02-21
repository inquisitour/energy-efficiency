# Cell 1: Privacy-Preserving GAN Setup
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import numpy as np
from config import *

# Cell 2: Privacy Budget Monitoring
class PrivacyBudgetCallback(tf.keras.callbacks.Callback):
    def __init__(self, n_samples, batch_size, noise_multiplier, target_delta):
        super().__init__()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        
    def on_epoch_end(self, epoch, logs=None):
        current_epsilon, _ = compute_dp_sgd_privacy(
            n=self.n_samples,
            batch_size=self.batch_size,
            noise_multiplier=self.noise_multiplier,
            epochs=epoch + 1,
            delta=self.target_delta
        )
        logs = logs or {}
        logs['epsilon'] = current_epsilon
        
        if current_epsilon > MAX_EPSILON:
            print(f"\nPrivacy budget exceeded (ε={current_epsilon:.2f})")
            self.model.stop_training = True

# Cell 3: Custom DP-DoppelGANger
class DPDGANGER(TimeSeriesSynthesizer):
    def __init__(self, model_parameters):
        super().__init__(modelname='doppelganger', model_parameters=model_parameters)
        self.epsilon = None
        self.delta = None
        
    def _build_discriminator(self):
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=MAX_GRAD_NORM,
            noise_multiplier=NOISE_MULTIPLIER,
            num_microbatches=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(1, activation='sigmoid'),
            tf.keras.layers.ClipByValue(-MAX_GRAD_NORM, MAX_GRAD_NORM)
        ])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

# Cell 4: Training Setup
# Load processed data
processed_data = np.loadtxt(PROCESSED_PATH, delimiter=',')

# Define model parameters
gan_args = ModelParameters(
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    latent_dim=LATENT_DIM
)

# Initialize model
synth = DPDGANGER(gan_args)

# Set up callbacks
checkpoint_path = os.path.join(MODEL_DIR, 'checkpoints/dp_doppelganger.ckpt')
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    ),
    PrivacyBudgetCallback(
        n_samples=processed_data.shape[0],
        batch_size=BATCH_SIZE,
        noise_multiplier=NOISE_MULTIPLIER,
        target_delta=DELTA
    )
]

# Cell 5: Model Training
try:
    history = synth.fit(
        processed_data,
        train_args=TrainParameters(
            epochs=EPOCHS,
            sequence_length=SEQ_LENGTH
        ),
        callbacks=callbacks
    )
    
    # Plot training history
    import plotly.express as px
    fig = px.line(history.history, title='Training History')
    fig.show()
    
except Exception as e:
    print(f"Training failed: {str(e)}")
    # Load last successful checkpoint if available
    if os.path.exists(checkpoint_path):
        print("Loading last successful checkpoint")
        synth.load_weights(checkpoint_path)

# Cell 6: Generate & Save Synthetic Data
synth.load_weights(checkpoint_path)  # Ensure best weights are loaded
synthetic = synth.sample(processed_data.shape[0])

# Inverse transform and save
synthetic_df = pd.DataFrame(
    preprocessor.inverse_transform(synthetic),
    columns=raw_df.columns
)
synthetic_df.to_csv(SYNTHETIC_PATH, index=False)

# Cell 7: Privacy Budget Calculation
final_epsilon, final_delta = compute_dp_sgd_privacy(
    n=processed_data.shape[0],
    batch_size=BATCH_SIZE,
    noise_multiplier=NOISE_MULTIPLIER,
    epochs=len(history.history['loss']),
    delta=DELTA
)

synth.epsilon = final_epsilon
synth.delta = final_delta

print("\nPrivacy Guarantees:")
print(f"(ε, δ)-DP: ({final_epsilon:.2f}, {final_delta})")
print(f"Privacy Budget Used: {(final_epsilon/MAX_EPSILON)*100:.1f}%")
