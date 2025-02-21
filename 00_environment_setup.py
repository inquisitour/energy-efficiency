# Cell 1: Install Required Packages
%%capture
!pip install -r requirements.txt

# Cell 2: Create Directory Structure
import os

# Create necessary directories
directories = [
    'data/raw',
    'data/processed',
    'data/synthetic',
    'models/checkpoints'
]

for dir_path in directories:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created/verified directory: {dir_path}")

# Cell 3: Verify Environment Setup
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow_privacy import optimizers as dp_optimizers
from ydata_synthetic.synthesizers import TimeSeriesSynthesizer

# Check TensorFlow and GPU
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU Available:", gpus[0].name)
    print("CUDA Version:", tf.sysconfig.get_build_info()["cuda_version"])
    # Enable memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found. Running on CPU")

# Verify other packages
print("\nKey Package Versions:")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {pd.__version__}")
print(f"tensorflow-privacy: {dp_optimizers.DPGradientDescentGaussianOptimizer.__version__}")

# Cell 4: Environment Test
def test_environment():
    try:
        # Test GPU operations
        if gpus:
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                print("GPU computation test: Passed")
        
        # Test data operations
        df = pd.DataFrame({'test': range(5)})
        print("Pandas operations test: Passed")
        
        # Test file operations
        test_file = os.path.join('data/raw', 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("File operations test: Passed")
        
        return True
    except Exception as e:
        print(f"Environment test failed: {str(e)}")
        return False

environment_ready = test_environment()
print(f"\nEnvironment Setup: {'SUCCESS' if environment_ready else 'FAILED'}")
