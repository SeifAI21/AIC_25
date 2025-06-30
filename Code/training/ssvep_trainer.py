"""
Steady-State Visual Evoked Potentials (SSVEP) Trainer
Handles training and cross-validation for SSVEP classical ML models
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from models.model_factory import create_ssvep_model
from data.ssvep_loader import load_ssvep_training_data, load_ssvep_test_data

class SSVEPTrainer:
    """Trainer for SSVEP EEG classification using classical ML models"""
    
    def __init__(self, config, data_path, output_dir, model_name):
        self.config = config
        self.data_path = data_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.seed = config.get('seed', 42)
        self.label_encoder = LabelEncoder()
        self._set_seed()
        
        print(f"SSVEP Trainer initialized")
        print(f"Model will be saved as: {model_name}")
    
    def _set_seed(self):
        """Set random seeds for reproducible results"""
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def _plot_confusion_matrix(self, y_true, y_pred, class_labels, save_path):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('SSVEP Classification - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {save_path}")
    
    def _cross_validate_model(self, X, y):
        """Perform cross-validation to evaluate model performance"""
        use_cv = self.config.get('training.use_cross_validation', True)
        if not use_cv:
            return None
        
        cv_folds = self.config.get('training.cv_folds', 5)
        scoring = self.config.get('training.scoring', 'f1_weighted')
        use_stratified = self.config.get('training.use_stratified', True)
        
        if use_stratified:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)
        else:
            cv = cv_folds
        
        # Get unique classes and create model
        unique_classes = len(np.unique(y))
        model = create_ssvep_model(self.config, unique_classes)
        
        print(f"Starting {cv_folds}-fold cross-validation...")
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        print(f"Cross-validation {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        print(f"Individual fold scores: {scores}")
        
        return scores.mean()
    
    def _train_final_model(self, X_train, y_train, X_val, y_val):
        """Train final model and evaluate on validation set"""
        print("Training final model...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        unique_classes = len(self.label_encoder.classes_)
        print(f"Number of classes: {unique_classes}")
        print(f"Class labels: {self.label_encoder.classes_}")
        
        # Create and train model
        model = create_ssvep_model(self.config, unique_classes)
        model.fit(X_train, y_train_encoded)
        
        # Evaluate on validation set
        val_predictions = model.predict(X_val)
        val_probabilities = model.predict_proba(X_val)
        
        # Calculate metrics
        val_accuracy = accuracy_score(y_val_encoded, val_predictions)
        val_f1 = f1_score(y_val_encoded, val_predictions, average='weighted')
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        
        # Print detailed classification report
        target_names = self.label_encoder.classes_
        report = classification_report(
            y_val_encoded, val_predictions, 
            target_names=target_names, 
            digits=4
        )
        print("\nDetailed Classification Report:")
        print(report)
        
        # Save model and related objects
        model_save_path = os.path.join(self.output_dir, f"{self.model_name}_ssvep_model.pkl")
        
        model_data = {
            'model': model,
            'label_encoder': self.label_encoder,
            'config': self.config.config,  # Save the config dict
            'val_f1': val_f1,
            'val_accuracy': val_accuracy,
            'class_labels': self.label_encoder.classes_
        }
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved: {model_save_path}")
        
        # Plot confusion matrix
        cm_save_path = os.path.join(self.output_dir, f"{self.model_name}_ssvep_confusion_matrix.png")
        self._plot_confusion_matrix(
            y_val_encoded, val_predictions, 
            self.label_encoder.classes_, cm_save_path
        )
        
        return val_f1
    
    def _train_with_cv_ensemble(self, X_full, y_full):
        """Train model using cross-validation ensemble approach from notebook"""
        print("Training with cross-validation ensemble approach...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_full)
        unique_classes = len(self.label_encoder.classes_)
        
        cv_folds = self.config.get('training.cv_folds', 5)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)
        
        # Store fold results
        fold_scores = []
        models = []
        
        print(f"Training {cv_folds} models using cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_encoded)):
            print(f"\n--- Fold {fold + 1} ---")
            
            X_train_fold, X_val_fold = X_full[train_idx], X_full[val_idx]
            y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
            
            # Create and train model for this fold
            model = create_ssvep_model(self.config, unique_classes)
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate on validation fold
            val_preds = model.predict(X_val_fold)
            val_f1 = f1_score(y_val_fold, val_preds, average='weighted')
            fold_scores.append(val_f1)
            models.append(model)
            
            print(f"Fold {fold + 1} F1 score: {val_f1:.4f}")
        
        # Calculate average CV performance
        mean_cv_score = np.mean(fold_scores)
        std_cv_score = np.std(fold_scores)
        
        print(f"\nCross-validation results:")
        print(f"Mean F1 Score: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")
        print(f"Individual fold scores: {fold_scores}")
        
        # Train final model on full dataset
        print("\nTraining final model on full dataset...")
        final_model = create_ssvep_model(self.config, unique_classes)
        final_model.fit(X_full, y_encoded)
        
        # Save model and related objects
        model_save_path = os.path.join(self.output_dir, f"{self.model_name}_ssvep_model.pkl")
        
        model_data = {
            'model': final_model,
            'cv_models': models,  # Save individual fold models for ensemble if needed
            'label_encoder': self.label_encoder,
            'config': self.config.config,
            'cv_scores': fold_scores,
            'mean_cv_score': mean_cv_score,
            'class_labels': self.label_encoder.classes_
        }
        
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved: {model_save_path}")
        
        return mean_cv_score
    
    def train(self):
        """Main training function"""
        try:
            print("Loading SSVEP training data...")
            
            # Load data
            train_index = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
            val_index = pd.read_csv(os.path.join(self.data_path, 'validation.csv'))
            
            X_train, y_train, X_val, y_val, loader = load_ssvep_training_data(
                train_index, val_index, self.data_path, self.config
            )
            
            if len(X_train) == 0:
                print("No SSVEP data found! Please check the data path and file structure.")
                return False
            
            print(f"Data loaded - Train: {X_train.shape}, Val: {X_val.shape}")
            print(f"Train labels: {len(y_train)} unique classes: {len(np.unique(y_train))}")
            print(f"Val labels: {len(y_val)} unique classes: {len(np.unique(y_val))}")
            
            # Store loader for later use in prediction
            loader_save_path = os.path.join(self.output_dir, f"{self.model_name}_ssvep_loader.pkl")
            with open(loader_save_path, 'wb') as f:
                pickle.dump(loader, f)
            print(f"Data loader saved: {loader_save_path}")
            
            # Combine train and val for CV ensemble
            X_full = np.vstack([X_train, X_val])
            y_full = np.concatenate([y_train, y_val])
            final_score = self._train_with_cv_ensemble(X_full, y_full)
            
            
            print(f"Training completed - Final score: {final_score:.4f}")
            return True
            
        except Exception as e:
            print(f"SSVEP training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
