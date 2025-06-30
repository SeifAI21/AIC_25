"""
Motor Imagery (MI) Task Configuration
Contains all parameters for MI data processing, model architecture, and training
"""

MI_CONFIG = {
    'data': {
        'channels': ['C3', 'CZ', 'C4'],  # Motor cortex channels for MI
        'all_channels': ['FZ', 'C3', 'CZ', 'C4'], # Included channels for initial processing
        'sampling_rate': 250,
        'samples_per_trial': 2250,  # 9 seconds * 250 Hz
        'epoch_window': [2.0, 4.0],  # Extract 2-4 seconds (movement execution)
        'task_name': 'MI'
    },
    
    'preprocessing': {
        'filter_low': 8,
        'filter_high': 30,
        'notch_freq': 50,
        'highpass_cutoff': 1.0,
        'use_car': True,
        'use_ica': True,
        'ica_components': 3,
        'asr_threshold_std': 5.0,
        'asr_window_sec': 1.0
    },
    
    'model': {
        'architecture': 'SimpleNet',
        'num_classes': 2,
        'dropout_rate': 0.20536865153489237
    },
    
    'training': {
        'batch_size': 16,
        'max_epochs': 200,
        'early_stopping_patience': 30,
        'learning_rate': 0.0015862257353335505,
        'weight_decay': 0.00011841849524796584,
        
        # Optuna hyperparameter optimization
        'use_optuna': False, # Enable Optuna optimization or not
        'optuna_trials': 150,
        'optuna_search_spaces': {
            'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
            'weight_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-3, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
            'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5}
        }
    },
    
    'labels': {
        'mapping': {'Left': 0, 'Right': 1},
        'inverse_mapping': {0: 'Left', 1: 'Right'}
    },
    
    'seed': 42
}

class MIConfig:
    """Configuration class for Motor Imagery task"""
    
    def __init__(self):
        self.config = MI_CONFIG
    
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
