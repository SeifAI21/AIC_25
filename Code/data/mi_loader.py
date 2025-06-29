"""
Motor Imagery (MI) Data Loading and Preprocessing
Handles loading, preprocessing, and feature extraction for MI EEG data
"""

import os
import mne
import numpy as np
import pandas as pd

from tqdm import tqdm
from mne.preprocessing import ICA

# Suppress MNE logging
mne.set_log_level('ERROR')

class MIDataLoader():
    """Data loader for Motor Imagery EEG data"""
    
    def __init__(self, config):
        self.config = config
        self.channels = config.get('data.channels') # Motor cortex channels
        self.sfreq = config.get('data.sampling_rate') # 250 Hz
        self.samples_per_trial = config.get('data.samples_per_trial') # 2250
        self.epoch_window = config.get('data.epoch_window')
        self.label_mapping = config.get('labels.mapping')
    
    def load_trial_data(self, row, base_path):
        """Load single trial EEG data from CSV file"""
        id_num = row['id']
        
        # Determine dataset split
        if id_num <= 4800:
            dataset = 'train'
        elif id_num <= 4900:
            dataset = 'validation'
        else:
            dataset = 'test'
        
        eeg_path = f"{base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"
        
        if not os.path.exists(eeg_path):
            raise FileNotFoundError(f"EEG data not found: {eeg_path}")
        
        eeg_data = pd.read_csv(eeg_path)
        trial_num = int(row['trial'])
        
        # Calculate trial boundaries
        start_idx = (trial_num - 1) * self.samples_per_trial
        end_idx = start_idx + self.samples_per_trial - 1
        
        trial_data = eeg_data.iloc[start_idx:end_idx+1]
        
        # Check data quality
        if trial_data['Validation'].mean() < 0.5:
            return None
        
        return trial_data
    
    def apply_notch_filter(self, raw, line_freq=50):
        """Remove powerline noise at 50 Hz and harmonics"""
        freqs = np.arange(line_freq, self.sfreq / 2, line_freq)
        raw.notch_filter(freqs=freqs, verbose=False)
        return raw
    
    def apply_highpass_filter(self, raw, cutoff=1.0):
        """Apply high-pass FIR filter"""
        raw.filter(l_freq=cutoff, h_freq=None, fir_design='firwin', verbose=False)
        return raw
    
    def apply_asr_like_artifact_removal(self, raw, window_size_sec=1.0, threshold_std=5.0):
        """Simple ASR-like artifact removal by zeroing high-RMS windows"""
        data = raw.get_data()
        n_samples = data.shape[1]
        win_size = int(window_size_sec * self.sfreq)
        
        if n_samples < win_size * 2:
            return raw
        
        n_windows = n_samples // win_size
        
        for ch in range(data.shape[0]):
            sig = data[ch]
            rms_vals = []
            
            for i in range(n_windows):
                seg = sig[i*win_size:(i+1)*win_size]
                rms_vals.append(np.sqrt(np.mean(seg**2)))
            
            rms_vals = np.array(rms_vals)
            mean_rms = rms_vals.mean()
            std_rms = rms_vals.std()
            threshold = mean_rms + threshold_std * std_rms
            
            for i, rms in enumerate(rms_vals):
                if rms > threshold:
                    data[ch, i*win_size:(i+1)*win_size] = 0.0
        
        raw._data = data
        return raw
    
    def preprocess_trial(self, trial_df, ch_names):
        """Complete preprocessing pipeline for single trial"""
        # Convert from microvolts to millivolts
        data = trial_df[ch_names].T.values / 1e3
        
        # Create MNE Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=self.sfreq, ch_types='eeg', verbose=False)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Apply preprocessing steps

        # 1. High-pass filter
        raw = self.apply_highpass_filter(raw, cutoff=self.config.get('preprocessing.highpass_cutoff'))

        # 2. Notch filter for powerline noise
        raw = self.apply_notch_filter(raw, line_freq=self.config.get('preprocessing.notch_freq'))

        # 3. ASR-like artifact removal
        raw = self.apply_asr_like_artifact_removal(
            raw, 
            window_size_sec=self.config.get('preprocessing.asr_window_sec'),
            threshold_std=self.config.get('preprocessing.asr_threshold_std')
        )
        
        # 4. Apply Common Average Reference if enabled
        if self.config.get('preprocessing.use_car'):
            try:
                raw.set_eeg_reference('average', projection=False, verbose=False)
            except Exception:
                pass
        
        # 5. ICA-based artifact removal for longer trials
        if raw.n_times > self.sfreq * 5 and len(raw.ch_names) > 1 and self.config.get('preprocessing.use_ica'):
            n_comps = min(len(ch_names) - 1, self.config.get('preprocessing.ica_components'))
            ica = ICA(n_components=n_comps, method='infomax', max_iter='auto', verbose=False)
            try:
                ica.fit(raw, verbose=False)
                raw = ica.apply(raw, verbose=False)
            except Exception:
                pass
        
        # 6. Band-pass filter for MI task
        low_freq = self.config.get('preprocessing.filter_low')
        high_freq = self.config.get('preprocessing.filter_high')
        try:
            raw.filter(l_freq=low_freq, h_freq=high_freq, verbose=False)
        except Exception:
            arr = raw.get_data()
            arr = mne.filter.filter_data(
                arr, sfreq=self.sfreq, l_freq=low_freq, h_freq=high_freq,
                phase='zero', verbose=False
            )
            raw._data = arr
        
        # 8. Extract epoch window
        t0_s, dur_s = self.epoch_window
        start = int(t0_s * self.sfreq)
        end = start + int(dur_s * self.sfreq)
        
        data_all = raw.get_data()
        if start >= data_all.shape[1]:
            start = 0
        if end > data_all.shape[1]:
            end = data_all.shape[1]
        
        epoch = data_all[:, start:end]
        if epoch.shape[1] == 0:
            raise RuntimeError("Epoch window exceeds data length")
        
        # 9. Baseline correction
        epoch = epoch - epoch.mean(axis=1, keepdims=True)
        
        return pd.DataFrame(epoch.T, columns=ch_names)
    
    def extract_features(self, index_df, base_path):
        """Extract features from multiple trials"""
        X_all = []
        y_all = []
        failed_count = 0
        
        # Use more channels for initial processing
        all_channels = self.config.get('data.all_channels')
        
        for idx in tqdm(range(len(index_df)), desc="Processing MI trials", ncols=100):
            row = index_df.iloc[idx]
            
            # Skip non-MI trials
            if row.get('task') != 'MI':
                continue
            
            try:
                # Load trial data
                trial_df = self.load_trial_data(row, base_path)
                if trial_df is None:
                    failed_count += 1
                    continue
                
                # Preprocess trial
                processed_df = self.preprocess_trial(trial_df[all_channels], all_channels)
                
                # Extract motor cortex channels only
                final_data = processed_df[self.channels]
                
                # Convert to array format (channels, samples)
                trial_array = final_data.values.T
                X_all.append(trial_array)
                
                # Extract label if available
                if 'label' in row:
                    label = self.label_mapping.get(row['label'])
                    if label is not None:
                        y_all.append(label)
                
            except Exception as e:
                failed_count += 1
                continue
        
        if failed_count > 0:
            print(f"Failed to process {failed_count} trials")
        
        return np.array(X_all), np.array(y_all) if y_all else None


def load_mi_training_data(train_index, val_index, base_path, config):
    """Load MI training and validation data"""
    loader = MIDataLoader(config)
    
    X_train, y_train = loader.extract_features(train_index, base_path)
    X_val, y_val = loader.extract_features(val_index, base_path)
    
    return X_train, y_train, X_val, y_val


def load_mi_test_data(test_index, base_path, config):
    """Load MI test data for inference"""
    loader = MIDataLoader(config)
    X_test, _ = loader.extract_features(test_index, base_path)
    return X_test
