# MTC-AIC3 EEG Classification Framework

## Overview

This repository contains our solution for the MTC-AIC3 Brain-Computer Interface competition, implementing machine learning techniques for two distinct EEG classification tasks:

1. **Motor Imagery (MI)**: Classification of imagined left vs right hand movements
2. **Steady-State Visual Evoked Potentials (SSVEP)**: Classification of attention direction across 4 classes (Left, Right, Forward, Backward)

The framework provides a unified command-line interface for training, evaluation, and submission generation, designed for both local development and Kaggle competition environments.

**Note**: This repository contains only the source code and framework. The MTC-AIC3 dataset must be obtained separately.

## Dataset Requirements

The framework expects the following dataset structure:
- **EEG Channels**: 8 channels (FZ, C3, CZ, C4, PZ, PO7, OZ, PO8)
- **Sampling Rate**: 250 Hz
- **Participants**: 40 subjects (30 training, 5 validation, 5 test)
- **Trial Duration**: 9 seconds (MI), 7 seconds (SSVEP)
- **Classes**: 2 (MI), 4 (SSVEP)
- **Format**: CSV files organized in the official MTC-AIC3 directory structure

## Installation and Setup

### Clone Repository

```bash
git clone <repository-url>
cd AIC25-MIA-AI
```

### Install Dependencies

Install all required Python packages using the provided requirements file:

```bash
pip install -r requirements.txt
```

#### Core Dependencies

The requirements.txt includes the following key packages:

**Scientific Computing:**
- numpy>=1.21.0 - Numerical computing
- pandas>=1.3.0 - Data manipulation and analysis
- scipy>=1.7.0 - Scientific computing library

**Machine Learning:**
- scikit-learn>=1.0.0 - Classical machine learning algorithms
- xgboost>=1.5.0 - Gradient boosting framework
- lightgbm>=3.3.0 - Light gradient boosting machine
- catboost>=1.0.0 - Categorical boosting algorithm

**Deep Learning:**
- torch>=1.9.0 - PyTorch deep learning framework
- torchvision>=0.10.0 - Computer vision utilities

**EEG Signal Processing:**
- mne>=1.0.0 - Neurophysiological data analysis

**Optimization:**
- optuna>=3.0.0 - Hyperparameter optimization framework

**Utilities:**
- tqdm>=4.62.0 - Progress bars
- matplotlib>=3.5.0 - Plotting library
- seaborn>=0.11.0 - Statistical data visualization
- pickle-mixin>=1.0.2 - Enhanced pickle functionality

### Dataset Setup

1. Obtain the MTC-AIC3 dataset.
2. Extract the dataset to your preferred location
3. Note the dataset path for use in commands (e.g., `/path/to/MTC-AIC3/`)

## Project Structure

```
AIC25-MIA-AI/
├── Code/                           # Main source code directory
│   ├── main.py                     # Main entry point
│   ├── cli/                        # Command-line interface
│   │   ├── main_cli.py             # Main CLI entry point
│   │   ├── train_command.py        # Training commands
│   │   └── predict_command.py      # Evaluation commands
│   ├── configs/                    # Configuration files
│   │   ├── mi_config.py            # Motor Imagery parameters
│   │   └── ssvep_config.py         # SSVEP parameters
│   ├── data/                       # Data loading utilities
│   │   ├── mi_loader.py            # MI data loading
│   │   └── ssvep_loader.py         # SSVEP data loading
│   ├── models/                     # Model architectures
│   │   ├── model_factory.py        # Model creation factory
│   │   ├── weights/                # Directory for trained model weights
│   │   └── zoo/                    # Pre-defined model architectures
│   │       ├── MI/                 # Motor Imagery models
│   │       └── SSVEP/              # SSVEP models
│   └── training/                   # Training pipelines
│       ├── mi_trainer.py           # MI training logic
│       ├── ssvep_trainer.py        # SSVEP training logic
│       └── parallel_trainer.py     # Parallel training utilities
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Usage

### Training Models

The framework supports training individual tasks or both tasks simultaneously:

#### Train Motor Imagery Model

```bash
python Code/main.py train --task MI --data-path /path/to/MTC-AIC3 --output-dir Code/models/weights/
```

#### Train SSVEP Model

```bash
python Code/main.py train --task SSVEP --data-path /path/to/MTC-AIC3 --output-dir Code/models/weights/
```

#### Train Both Models Sequentially

```bash
python Code/main.py train --task BOTH --data-path /path/to/MTC-AIC3 --output-dir Code/models/weights/
```

#### Custom Model Names

```bash
python Code/main.py train --task BOTH --ssvep-model-name custom_ssvep --mi-model-name custom_mi --data-path /path/to/MTC-AIC3 --output-dir Code/models/weights/
```

### Generate Predictions

After training, generate predictions for test data:

```bash
python Code/main.py evaluate --data-path /path/to/MTC-AIC3 --models-dir Code/models/weights/ --output-file submission.csv
```

#### Custom Model Loading

```bash
python Code/main.py evaluate --data-path /path/to/MTC-AIC3 --models-dir Code/models/weights/ --ssvep-model-name custom_ssvep --mi-model-name custom_mi --output-file submission.csv
```

### Command Line Options

#### Training Parameters

- `--task`: Task to train (SSVEP, MI, BOTH)
- `--data-path`: Path to dataset directory
- `--output-dir`: Directory to save trained models
- `--parallel`: Enable parallel training for both tasks
- `--ssvep-model-name`: Custom name for SSVEP model weights
- `--mi-model-name`: Custom name for MI model weights
- `--config-dir`: Directory containing configuration files

#### Evaluation Parameters

- `--data-path`: Path to test dataset
- `--models-dir`: Directory containing trained models
- `--output-file`: Output CSV file path
- `--ssvep-model-name`: Name of SSVEP model to load
- `--mi-model-name`: Name of MI model to load

## Model Architectures

### Motor Imagery (MI)

- **Architecture**: SimpleNet (Convolutional Neural Network)
- **Input Channels**: C3, CZ, C4 (motor cortex regions)
- **Preprocessing**: 
  - Bandpass filtering (8-30 Hz)
  - Common Average Reference (CAR)
  - Independent Component Analysis (ICA)
- **Epoch Window**: 2-4 seconds (movement execution phase)

### SSVEP

- **Architecture**: Ensemble voting classifier
- **Classifiers**: Linear Discriminant Analysis, LightGBM, CatBoost
- **Input Channels**: PZ, PO7, OZ, PO8 (posterior regions)
- **Feature Extraction**:
  - Power Spectral Density (PSD)
  - Canonical Correlation Analysis (CCA)
  - Task-Related Component Analysis (TRCA)
- **Analysis Window**: 3-6 seconds (steady-state period)

## Output Files

### Training Outputs

Trained models are saved in the specified output directory (`Code/models/weights/` by default):
- `{model_name}.pth`: PyTorch model weights (MI)
- `{model_name}.pkl`: Pickled ensemble model (SSVEP)

### Prediction Outputs

The evaluation command generates a CSV file with the following format:
- **Columns**: subject_id, session_id, prediction
- **Values**: Predictions for each test trial
- **Classes**: 
  - MI: 0 (Left), 1 (Right)
  - SSVEP: 0 (Left), 1 (Right), 2 (Forward), 3 (Backward)

## Task Differentiation

### Motor Imagery (MI)
- **Paradigm**: Imagined hand movements
- **Neural Origin**: Sensorimotor cortex mu rhythm (8-13 Hz) and beta rhythm (13-30 Hz)
- **Channels**: Central electrodes (C3, CZ, C4)
- **Classification**: Binary (Left vs Right hand)
- **Method**: Deep learning with temporal convolutions

### SSVEP
- **Paradigm**: Visual attention to flickering stimuli
- **Neural Origin**: Visual cortex steady-state responses
- **Channels**: Posterior electrodes (PZ, PO7, OZ, PO8)
- **Classification**: 4-class directional
- **Method**: Classical machine learning with frequency domain features

## Configuration

Model parameters and preprocessing settings are configurable through files in the `Code/configs/` directory:
- `Code/configs/mi_config.py`: Motor Imagery parameters
- `Code/configs/ssvep_config.py`: SSVEP parameters

Key configurable aspects:
- Filtering parameters
- Epoch extraction windows
- Model hyperparameters
- Feature extraction methods
- Training parameters

## Kaggle Environment

The framework is optimized for Kaggle environments:
- Automatic path detection
- Memory-efficient processing
- Clean exit handling

Default paths for Kaggle:
- Dataset: `/kaggle/input/mtcaic3`
- Output: `/kaggle/working/submission.csv`

## Performance Notes

- **MI Task**: Achieves ~73% accuracy using optimized neural networks
- **SSVEP Task**: Achieves ~74% accuracy using ensemble methods
- **Training Time**: Approximately 30-45 minutes per task (depends on hardware)

## Reproducibility Notes

Due to numerical precision variations inherent in different execution environments (notebook vs. script migration), minor weight deviations may occur during model reproduction. Therfore we provided pre-trained weights (`Code/models/weights/mi_model_SimpleNet.pth`) that represents the exact model state used for the final competition submission and maintains the reported performance metrics.

While these minor parameter variations typically maintain validation set performance consistency, they may exhibit amplified effects on test set generalization due to the inherent signal-to-noise characteristics and inter-subject variability present in motor imagery EEG data. Detailed analysis of these data quality considerations is provided in the accompanying project documentation.

## License

This project is developed for the MTC-AIC3 competition. Please refer to competition guidelines for usage restrictions and requirements.