# Watt's Up? Energy Efficiency Hackathon Project

## Overview
This project was developed for the "Watt's Up? Hack for Energy Efficiency" hackathon at TU Wien (February 22-23, 2025). It implements a privacy-preserving approach to optimize building energy consumption using synthetic data generation and machine learning techniques.

## Key Features
- 🔒 Privacy-preserving synthetic data generation using DP-DoppelGANger
- 📊 Time-series energy consumption analysis
- 🎯 Multi-objective optimization for energy efficiency
- 🌡️ Comfort-aware temperature and occupancy recommendations
- 📈 Comprehensive validation and visualization suite

## Project Structure
```
hackathon_project/
├── config.py                  # Configuration parameters
├── 00_environment_setup.py # Environment setup and verification
├── 01_data_loading.py     # Data preprocessing pipeline
├── 02_synthetic_generation.py # Privacy-preserving GAN implementation
├── 03_validation.py       # Synthetic data validation
├── 04_simulation.py       # Energy optimization model
├── 05_presentation.py     # Results compilation
├── data/
│   ├── raw/                 # Original building data
│   ├── processed/           # Preprocessed datasets
│   ├── synthetic/          # Generated synthetic data
│   └── backups/            # Fallback datasets
└── models/
    └── checkpoints/        # Model checkpoints
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/watts-up-hackathon.git
cd watts-up-hackathon
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify environment setup:
```bash
jupyter notebook 00_environment_setup.ipynb
```

## Technical Stack
- **Python**: 3.9+
- **Core Libraries**:
  - TensorFlow 2.15.0
  - TensorFlow Privacy 0.9.0
  - scikit-learn 1.3.2
  - pandas 2.1.4
  - plotly 5.18.0
  - YData-Synthetic 1.4.0

## Pipeline Workflow

1. **Environment Setup** (`00_environment_setup.ipynb`)
   - Package installation
   - Directory structure creation
   - GPU compatibility verification

2. **Data Loading** (`01_data_loading.ipynb`)
   - Raw data ingestion
   - Feature engineering
   - Preprocessing pipeline

3. **Synthetic Generation** (`02_synthetic_generation.ipynb`)
   - DP-DoppelGANger implementation
   - Privacy budget monitoring
   - Synthetic data generation

4. **Validation** (`03_validation.ipynb`)
   - Distribution analysis
   - Privacy preservation verification
   - Anomaly detection

5. **Simulation** (`04_simulation.ipynb`)
   - Energy consumption prediction
   - Multi-objective optimization
   - Comfort-aware recommendations

6. **Presentation** (`05_presentation.ipynb`)
   - Results compilation
   - Visualization generation
   - Report creation

## Privacy Features
- Differential Privacy integration (ε-δ guarantees)
- Privacy budget monitoring
- Synthetic data validation
- No direct data exposure

## Performance Metrics
- Privacy Budget (ε): Configurable, default 10.0
- Data Similarity: Wasserstein distance validation
- Energy Savings: Calculated per building
- CO2 Reduction: Tracked in kg/year

## Usage Example
```python
# Load and preprocess data
python 01_data_loading.ipynb

# Generate synthetic data
python 02_synthetic_generation.ipynb

# Get optimization recommendations
python 04_simulation.ipynb
```

## Contributing
This project was developed during a hackathon but we welcome improvements:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License - see LICENSE file for details

## Team
- [Your Name]
- [Team Member 2]
- [Team Member 3]

## Acknowledgments
- TU Wien for hosting the hackathon
- [Other organizations/people to thank]

## Contact
For questions or feedback, please contact [your email]

---
**Note**: This project was developed during a hackathon and is provided as-is. For production use, additional testing and validation would be recommended.
