"""
Steady-State Visual Evoked Potentials (SSVEP) Task Configuration
Contains all parameters for SSVEP data processing, feature extraction, and training
"""

SSVEP_CONFIG = {
    'data': {
        'all_channels': ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8'],  # All channels for initial processing
        'post_channels': ['PZ', 'PO7', 'OZ', 'PO8'],  # Posterior channels for SSVEP
        'sampling_rate': 250,
        'samples_per_trial': 1750,  # 7 seconds * 250 Hz
        'epoch_window': [1.0, 7.0],  # Extract 1-7 seconds (SSVEP stimulation window)
        'ssvep_window': [3.0, 6.0],  # Final SSVEP analysis window (3-6 seconds)
        'task_name': 'SSVEP',
        'target_freqs': [7, 8, 10, 13, 14, 16, 20, 21, 24, 26, 30, 39]
    },
    
    'preprocessing': {
        'filter_low': 6,
        'filter_high': 42,
        'notch_freq': 50,
        'use_ica': True,
        'ica_components': 8,
        'ica_eog_channels': ['FZ', 'CZ', 'C3', 'C4'],
        'ica_threshold': 3
    },
    
    'feature_extraction': {
        'use_psd': True,
        'use_cca': True,
        'use_trca': True,
        'psd_fmin': 1,
        'psd_fmax': 80,
        'cca_harmonics': 2,
        'psd_harmonics': [1, 2]  # Fundamental and first harmonic
    },
    
    'model': {
        'type': 'classical',  # 'classical' for sklearn models, 'neural' for deep learning
        'classifiers': ['lda', 'lgbm', 'cat'],  # Top 3 from notebook
        'voting_type': 'soft',
        'cv_folds': 5
    },
    
    'training': {
        'use_cross_validation': True,
        'cv_folds': 5,
        'random_state': 42,
        'use_stratified': True,
        'scoring': 'f1_weighted'
    },
    
    'labels': {
        # Will be determined dynamically from data
        'mapping': {},
        'inverse_mapping': {}
    },
    
    'seed': 42
}

class SSVEPConfig:
    """Configuration class for SSVEP task"""
    
    def __init__(self):
        self.config = SSVEP_CONFIG
    
    def get(self, key_path, default=None):
        """Get nested configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def update(self, key_path, new_value):
        """Update nested configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = new_value
