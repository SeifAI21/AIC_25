"""
Motor Imagery (MI) Trainer
Handles training, validation, and hyperparameter optimization for MI models
"""

import os
import pickle
import random
import optuna

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from optuna.exceptions import TrialPruned

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from models.model_factory import create_mi_model
from data.mi_loader import load_mi_training_data, load_mi_test_data

class EEGDataset(Dataset):
    """PyTorch Dataset wrapper for EEG data"""
    
    def __init__(self, X, y=None):
        """
        Args:
            X: numpy array of shape (n_samples, channels, time_points)
            y: numpy array of labels (n_samples,) or None for test data
        """
        self.X = torch.FloatTensor(X)
        if y is not None:
            self.y = torch.LongTensor(y)
        else:
            self.y = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is not None:
            return x, self.y[idx]
        else:
            return x

class MITrainer:
    """Trainer for Motor Imagery EEG classification"""
    
    def __init__(self, config, data_path, output_dir, model_name):
        self.config = config
        self.data_path = data_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = config.get('seed', 42)
        self._set_seed()
        
        print(f"MI Trainer initialized - Device: {self.device}")
        print(f"Model will be saved as: {model_name}")
    
    def _set_seed(self):
        """Set random seeds for reproducible results"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _seed_worker(self, worker_id):
        """Seed worker for DataLoader to ensure reproducibility"""
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def _create_data_loaders(self, train_dataset, val_dataset, batch_size):
        """Create training and validation data loaders"""
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True, 
            generator=generator,
            worker_init_fn=self._seed_worker,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, 
            generator=generator,
            worker_init_fn=self._seed_worker,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """Train model for one epoch"""
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        # Add progress bar for training
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            if batch_x.ndim == 3:
                batch_x = batch_x.unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader.dataset)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, f1, acc
    
    def _validate_epoch(self, model, val_loader, criterion):
        """Validate model for one epoch"""
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        # Add progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                if batch_x.ndim == 3:
                    batch_x = batch_x.unsqueeze(1)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_x.size(0)
                all_preds.append(outputs.argmax(dim=1).cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(val_loader.dataset)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, f1, acc, all_preds, all_labels
    
    def _objective(self, trial, train_dataset, val_dataset):
        """Optuna objective function"""
        search_spaces = self.config.get('training.optuna_search_spaces')
        params = {}
        
        for param_name, space_config in search_spaces.items():
            if space_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, space_config['low'], space_config['high'],
                    log=space_config.get('log', False)
                )
            elif space_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, space_config['choices'])
        
        return self._train_single_trial(trial, train_dataset, val_dataset, params)
    
    def _train_single_trial(self, trial, train_dataset, val_dataset, params):
        """Train model with specific parameters and return best validation F1"""
        self._set_seed()
        
        train_loader, val_loader = self._create_data_loaders(
            train_dataset, val_dataset, params['batch_size']
        )
        
        # Create model with selected architecture
        channels, samples = train_dataset.X.shape[1], train_dataset.X.shape[2]
        model = create_mi_model(self.config, channels, samples, dropout_rate=params['dropout_rate'])
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
        
        best_val_f1 = 0.0
        patience_counter = 0
        early_stop_patience = 30
        max_epochs = 100
        
        for epoch in range(max_epochs):
            train_loss, train_f1, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_f1, val_acc, _, _ = self._validate_epoch(model, val_loader, criterion)
            
            scheduler.step(val_f1)
            
            # Report to Optuna for pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise TrialPruned()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                break
        
        return best_val_f1
    
    def _train_final_model(self, train_dataset, val_dataset, best_params):
        """Train final model with best parameters"""
        print("Training final model with optimized parameters...")
        
        self._set_seed()
        train_loader, val_loader = self._create_data_loaders(
            train_dataset, val_dataset, best_params['batch_size']
        )

        # Create model with selected architecture and name
        channels, samples = train_dataset.X.shape[1], train_dataset.X.shape[2]
        model = create_mi_model(self.config, channels, samples, dropout_rate=best_params['dropout_rate'])
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
        
        best_val_f1 = 0.0
        best_val_preds = None
        best_val_labels = None
        patience_counter = 0
        early_stop_patience = self.config.get('training.early_stopping_patience', 30)
        max_epochs = self.config.get('training.max_epochs', 200)
        
        model_arch = self.config.get('model.architecture', 'SimpleNet')

        # Save path with architecture and model name
        model_save_path = os.path.join(self.output_dir, f"{self.model_name}_{model_arch}.pth")
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(max_epochs), desc="Training Progress")
        
        for epoch in epoch_pbar:
            train_loss, train_f1, train_acc = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_f1, val_acc, val_preds, val_labels = self._validate_epoch(model, val_loader, criterion)
            
            scheduler.step(val_f1)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'Val F1': f'{val_f1:.4f}',
                'Best F1': f'{best_val_f1:.4f}',
                'Train F1': f'{train_f1:.4f}'
            })
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_preds = val_preds
                best_val_labels = val_labels
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'architecture': model_arch,
                    'channels': channels,
                    'samples': samples,
                    'best_params': best_params,
                    'val_f1': best_val_f1
                }, model_save_path)
            else:
                patience_counter += 1
            
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}: Val F1 {val_f1:.4f} (best {best_val_f1:.4f})")
            
            if patience_counter >= early_stop_patience:
                break
        
        epoch_pbar.close()
        
        # Plot confusion matrix with best validation results
        if best_val_preds is not None and best_val_labels is not None:
            cm_save_path = os.path.join(self.output_dir, f"{self.model_name}_{model_arch}_confusion_matrix.png")
            self._plot_confusion_matrix(best_val_labels, best_val_preds, cm_save_path)
        
        print(f"Model saved: {model_save_path}")
        print(f"Best validation F1: {best_val_f1:.4f}")
        
        return best_val_f1
    
    def _plot_confusion_matrix(self, y_true, y_pred, save_path):
        """Plot and save confusion matrix"""
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                   yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
        plt.title('Confusion Matrix - Final Validation Results')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {save_path}")
    
    def train(self):
        """Main training function with optional Optuna optimization"""
        print("Loading MI training data...")
        
        # Load data
        train_index = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        val_index = pd.read_csv(os.path.join(self.data_path, 'validation.csv'))
        X_train, y_train, X_val, y_val = load_mi_training_data(
            train_index, val_index, self.data_path, self.config
        )
        
        print(f"Data loaded - Train: {X_train.shape}, Val: {X_val.shape}")
        
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        
        # Check if Optuna optimization is enabled
        use_optuna = self.config.get('training.use_optuna', False)
        
        if use_optuna:
            print("Starting hyperparameter optimization with Optuna...")
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )
            
            n_trials = self.config.get('training.optuna_trials', 50)
            
            # Add progress bar for Optuna trials
            with tqdm(total=n_trials, desc="Optuna Optimization") as pbar:
                def callback(study, trial):
                    pbar.set_postfix({
                        'Best F1': f'{study.best_trial.value:.4f}' if study.best_trial else 'N/A',
                        'Trial': trial.number
                    })
                    pbar.update(1)
                
                study.optimize(
                    lambda trial: self._objective(trial, train_dataset, val_dataset),
                    n_trials=n_trials,
                    callbacks=[callback],
                    show_progress_bar=False
                )
            
            print(f"Optimization completed - Best F1: {study.best_trial.value:.4f}")
            best_params = study.best_trial.params
        else:
            # Use default parameters
            print("Using default parameters...")
            best_params = {
                'lr': self.config.get('training.learning_rate'),
                'weight_decay': self.config.get('training.weight_decay'),
                'batch_size': self.config.get('training.batch_size'),
                'dropout_rate': self.config.get('model.dropout_rate')
            }
        
        # Train final model
        final_f1 = self._train_final_model(train_dataset, val_dataset, best_params)
        print(f"Training completed - Final F1: {final_f1:.4f}")
