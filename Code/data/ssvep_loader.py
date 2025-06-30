"""
Steady-State Visual Evoked Potentials (SSVEP) Data Loading and Preprocessing
Handles loading, preprocessing, and feature extraction for SSVEP EEG data
"""

import os
import mne
mne.set_log_level('ERROR')
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_decomposition import CCA
from mne.time_frequency import psd_array_welch
from scipy.linalg import eigh

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
mne.set_log_level('ERROR')

class SSVEPDataLoader:
    """Data loader for SSVEP EEG data"""
    
    def __init__(self, config):
        self.config = config
        self.all_channels = config.get('data.all_channels')
        self.post_channels = config.get('data.post_channels')
        self.sfreq = config.get('data.sampling_rate')
        self.samples_per_trial = config.get('data.samples_per_trial')
        self.target_freqs = config.get('data.target_freqs')
        self.ssvep_window = config.get('data.ssvep_window')  # [3, 6] seconds
        
        # For storing TRCA filters and templates
        self.trca_filters = None
        self.templates = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def apply_ica(self, raw, n_components=None):
        """Apply ICA for artifact removal"""
        if n_components is None:
            n_components = self.config.get('preprocessing.ica_components', 8)
        
        ica = mne.preprocessing.ICA(
            n_components=n_components, 
            method='fastica', 
            random_state=42, 
            max_iter='auto', 
            verbose=False
        )
        ica.fit(raw)

        # Try detecting EOG using fronto-central channels
        eog_channels = self.config.get('preprocessing.ica_eog_channels')
        threshold = self.config.get('preprocessing.ica_threshold', 3)
        
        try:
            eog_inds, scores = ica.find_bads_eog(
                raw, 
                ch_name=eog_channels, 
                threshold=threshold, 
                verbose=False
            )
            ica.exclude = eog_inds
        except Exception:
            pass  # Continue without EOG removal if it fails

        cleaned_raw = ica.apply(raw.copy(), verbose=False)
        return cleaned_raw
    
    def preprocess_and_epoch(self, trial_df):
        """Preprocess trial data and extract epoch"""
        # Use all channels for initial processing
        df = trial_df.copy()[self.all_channels]
        
        # Create MNE raw object
        info = mne.create_info(
            ch_names=self.all_channels, 
            sfreq=self.sfreq, 
            ch_types="eeg", 
            verbose=False
        )
        raw = mne.io.RawArray(df.T.values, info, verbose=False)

        # Apply filtering
        filter_low = self.config.get('preprocessing.filter_low')
        filter_high = self.config.get('preprocessing.filter_high')
        raw.filter(filter_low, filter_high, fir_design='firwin', verbose=False)
        
        # Apply notch filter
        notch_freq = self.config.get('preprocessing.notch_freq')
        raw.notch_filter(notch_freq, verbose=False)

        # Apply ICA on full channel set
        if self.config.get('preprocessing.use_ica'):
            raw = self.apply_ica(raw, n_components=len(self.all_channels))

        # Select posterior channels for SSVEP analysis
        raw.pick_channels(self.post_channels)

        # Extract SSVEP window (3s to 6s from the notebook)
        start_time, end_time = self.ssvep_window
        start_idx = int(start_time * self.sfreq)
        stop_idx = int(end_time * self.sfreq)
        
        epoched = raw.get_data(start=start_idx, stop=stop_idx)
        
        # Normalize each channel
        epoched = (epoched - epoched.mean(axis=1, keepdims=True)) / epoched.std(axis=1, keepdims=True)
        
        return epoched
    
    def extract_psd_snr_features(self, epoch):
        """Extract PSD-based SNR features for target frequencies"""
        features = []
        
        for ch in range(epoch.shape[0]):
            sig = epoch[ch, :]
            psd, freqs = psd_array_welch(
                sig.reshape(1, -1), 
                sfreq=self.sfreq, 
                fmin=self.config.get('feature_extraction.psd_fmin'),
                fmax=self.config.get('feature_extraction.psd_fmax'),
                n_fft=2 * self.sfreq, 
                verbose=False
            )
            psd = psd.flatten()
            
            # Extract features for target frequencies and harmonics
            for freq in self.target_freqs:
                for h in self.config.get('feature_extraction.psd_harmonics'):
                    target = freq * h
                    band = (freqs >= target - 0.5) & (freqs <= target + 0.5)
                    power = np.mean(psd[band]) if np.any(band) else 0
                    features.append(np.log10(power + 1e-12))
        
        return features
    
    def generate_cca_references(self, freq, t, n_harmonics=None):
        """Generate CCA reference signals for given frequency"""
        if n_harmonics is None:
            n_harmonics = self.config.get('feature_extraction.cca_harmonics')
        
        refs = []
        for h in range(1, n_harmonics + 1):
            refs.append(np.sin(2 * np.pi * h * freq * t))
            refs.append(np.cos(2 * np.pi * h * freq * t))
        return np.array(refs).T
    
    def extract_cca_features(self, epoch):
        """Extract CCA correlation features"""
        t = np.arange(epoch.shape[1]) / self.sfreq
        eeg = epoch.T
        cca = CCA(n_components=1)
        correlations = []
        
        for freq in self.target_freqs:
            ref = self.generate_cca_references(freq, t)
            try:
                cca.fit(eeg, ref)
                u, v = cca.transform(eeg, ref)
                correlation = np.corrcoef(u.T, v.T)[0, 1]
                correlations.append(correlation if not np.isnan(correlation) else 0.0)
            except Exception:
                correlations.append(0.0)
        
        return correlations
    
    def trca(self, X_class):
        """Task-Related Component Analysis"""
        S = np.zeros((X_class.shape[1], X_class.shape[1]))
        
        for i in range(X_class.shape[0]):
            for j in range(i + 1, X_class.shape[0]):
                x1 = X_class[i] - X_class[i].mean(axis=1, keepdims=True)
                x2 = X_class[j] - X_class[j].mean(axis=1, keepdims=True)
                S += x1 @ x2.T + x2 @ x1.T
        
        Q = np.concatenate(X_class, axis=1)
        Q = Q @ Q.T
        
        try:
            evals, evecs = eigh(S, Q)
            return evecs[:, -1]
        except Exception:
            # Return identity filter if TRCA fails
            return np.zeros(X_class.shape[1])
    
    def extract_trca_features(self, epoch):
        """Extract TRCA correlation features"""
        if self.trca_filters is None or self.templates is None:
            # Return zeros if TRCA not trained yet
            return [0.0] * len(self.target_freqs)
        
        correlations = []
        for i in range(self.trca_filters.shape[0]):
            w = self.trca_filters[i]
            
            if np.all(w == 0):  # Skip if filter is zero
                correlations.append(0.0)
                continue
                
            trial_proj = w @ epoch
            template_proj = w @ self.templates[i]
            
            try:
                correlation = np.corrcoef(template_proj, trial_proj)[0, 1]
                correlations.append(correlation if not np.isnan(correlation) else 0.0)
            except Exception:
                correlations.append(0.0)
        
        return correlations
    
    def extract_features(self, epoch):
        """Extract all features from a single epoch"""
        features = []
        
        if self.config.get('feature_extraction.use_psd'):
            psd_features = self.extract_psd_snr_features(epoch)
            features.extend(psd_features)
        
        if self.config.get('feature_extraction.use_cca'):
            cca_features = self.extract_cca_features(epoch)
            features.extend(cca_features)
        
        if self.config.get('feature_extraction.use_trca'):
            trca_features = self.extract_trca_features(epoch)
            features.extend(trca_features)
        
        return np.array(features)
    
    def build_trca_filters_and_templates(self, X_epochs, y_labels):
        """Build TRCA filters and templates from training data"""
        unique_labels = np.unique(y_labels)
        n_classes = len(unique_labels)
        
        self.trca_filters = np.zeros((n_classes, X_epochs.shape[1]))
        self.templates = np.zeros((n_classes, X_epochs.shape[1], X_epochs.shape[2]))
        
        for i, cls in enumerate(unique_labels):
            class_epochs = X_epochs[y_labels == cls]
            if len(class_epochs) > 0:
                self.trca_filters[i] = self.trca(class_epochs)
                self.templates[i] = np.mean(class_epochs, axis=0)
    
    def load_trial_data(self, row, base_path):
        """Load single trial EEG data from CSV file"""
        # Determine dataset split based on subject_id
        subject_id = int(row['subject_id'].replace('S', ''))
        if subject_id < 31:
            dataset = 'train'
        else:
            dataset = 'validation'
        
        eeg_path = f"{base_path}/SSVEP/{dataset}/S{subject_id}/{row['trial_session']}/EEGdata.csv"
        
        if not os.path.exists(eeg_path):
            raise FileNotFoundError(f"EEG data not found: {eeg_path}")
        
        eeg_data = pd.read_csv(eeg_path)
        trial_num = int(row['trial'])
        
        # Calculate trial boundaries
        start_idx = (trial_num - 1) * self.samples_per_trial
        end_idx = start_idx + self.samples_per_trial
        
        if end_idx > len(eeg_data):
            raise ValueError(f"Trial extends beyond data length")
        
        trial_data = eeg_data.iloc[start_idx:end_idx]
        return trial_data
    
    def load_trial_data_test(self, row, base_path):
        """Load single trial EEG data from CSV file for test set"""
        subject_id = int(row['subject_id'].replace('S', ''))
        
        eeg_path = f"{base_path}/SSVEP/test/S{subject_id}/{row['trial_session']}/EEGdata.csv"
        
        if not os.path.exists(eeg_path):
            raise FileNotFoundError(f"EEG data not found: {eeg_path}")
        
        eeg_data = pd.read_csv(eeg_path)
        trial_num = int(row['trial'])
        
        # Calculate trial boundaries
        start_idx = (trial_num - 1) * self.samples_per_trial
        end_idx = start_idx + self.samples_per_trial
        
        if end_idx > len(eeg_data):
            raise ValueError(f"Trial extends beyond data length")
        
        trial_data = eeg_data.iloc[start_idx:end_idx]
        return trial_data
    
    def extract_features_from_trials(self, index_df, base_path, fit_transforms=False, is_test=False):
        """Extract features from multiple trials"""
        X_epochs = []
        y_labels = []
        failed_count = 0
        
        # First pass: collect epochs for TRCA training
        for idx in tqdm(range(len(index_df)), desc="Loading SSVEP epochs", ncols=100):
            row = index_df.iloc[idx]
            
            # Skip non-SSVEP trials
            if row.get('task') != 'SSVEP':
                continue
            
            # Load trial data
            if is_test:
                trial_df = self.load_trial_data_test(row, base_path)
            else:
                trial_df = self.load_trial_data(row, base_path)
            
            # Preprocess and extract epoch
            epoch = self.preprocess_and_epoch(trial_df)
            X_epochs.append(epoch)
            
            # Extract label if available
            if 'label' in row:
                y_labels.append(row['label'])
                
            
        
        if len(X_epochs) == 0:
            return np.array([]), np.array([])
        
        X_epochs = np.array(X_epochs)
        y_labels = np.array(y_labels) if y_labels else None
        
        # Fit label encoder and build TRCA filters if training
        if fit_transforms and y_labels is not None:
            y_encoded = self.label_encoder.fit_transform(y_labels)
            self.build_trca_filters_and_templates(X_epochs, y_encoded)
        
        # Second pass: extract features
        X_features = []
        for epoch in tqdm(X_epochs, desc="Extracting SSVEP features", ncols=100):
            features = self.extract_features(epoch)
            X_features.append(features)
        
        X_features = np.array(X_features)
        
        # Fit scaler if training
        if fit_transforms:
            X_features = self.scaler.fit_transform(X_features)
        else:
            X_features = self.scaler.transform(X_features)
        
        if failed_count > 0:
            print(f"Failed to process {failed_count} trials")
        
        return X_features, y_labels


def load_ssvep_training_data(train_index, val_index, base_path, config):
    """Load SSVEP training and validation data"""
    loader = SSVEPDataLoader(config)
    
    # Load training data first to fit transforms
    X_train, y_train = loader.extract_features_from_trials(
        train_index, base_path, fit_transforms=True
    )
    
    # Load validation data using fitted transforms
    X_val, y_val = loader.extract_features_from_trials(
        val_index, base_path, fit_transforms=False
    )
    
    return X_train, y_train, X_val, y_val, loader


def load_ssvep_test_data(test_index, base_path, config, trained_loader):
    """Load SSVEP test data for inference"""
    # Use the trained loader with fitted transforms
    X_test, _ = trained_loader.extract_features_from_trials(
        test_index, base_path, fit_transforms=False, is_test=True
    )
    return X_test
