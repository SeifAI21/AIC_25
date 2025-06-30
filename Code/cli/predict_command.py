"""
Evaluation command implementation
Loads test data, runs inference with trained models, generates combined CSV
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader

def run_evaluation(args):
    """
    Main evaluation function
    Args:
        args: Parsed command line arguments
    """
    print("Starting evaluation and submission generation...")
    print(f"Data path: {args.data_path}")
    print(f"Models directory: {args.models_dir}")
    print(f"Output file: {args.output_file}")
    print(f"SSVEP model name: {args.ssvep_model_name}")
    print(f"MI model name: {args.mi_model_name}")
    
    # Load test data for both tasks
    test_data = load_test_data(args.data_path)
    
    # Load trained models
    models = load_trained_models(args.models_dir, args.ssvep_model_name, args.mi_model_name)
    
    # Generate predictions
    predictions = generate_predictions(test_data, models, args.data_path)
    
    # Create combined submission
    create_submission_csv(predictions, args.output_file)
    
    print(f"Evaluation completed. Submission saved to: {args.output_file}")

def load_test_data(data_path):
    """Load test data metadata"""
    print("Loading test data metadata...")
    
    # Load test metadata
    test_df = pd.read_csv(f"{data_path}/test.csv")
    
    # Separate SSVEP and MI test data
    ssvep_test = test_df[test_df['task'] == 'SSVEP'].copy()
    mi_test = test_df[test_df['task'] == 'MI'].copy()
    
    print(f"Found {len(ssvep_test)} SSVEP test samples")
    print(f"Found {len(mi_test)} MI test samples")
    
    return {
        'ssvep': ssvep_test,
        'mi': mi_test
    }

def load_trained_models(models_dir, ssvep_model_name, mi_model_name):
    """Load trained SSVEP and MI models"""
    
    models = {}
    
    # Load SSVEP model (classical ML model saved as pickle)
    ssvep_model_path = os.path.join(models_dir, f'{ssvep_model_name}_ssvep_model.pkl')
    ssvep_loader_path = os.path.join(models_dir, f'{ssvep_model_name}_ssvep_loader.pkl')
    
    if os.path.exists(ssvep_model_path) and os.path.exists(ssvep_loader_path):
        with open(ssvep_model_path, 'rb') as f:
            ssvep_model_data = pickle.load(f)
        with open(ssvep_loader_path, 'rb') as f:
            ssvep_loader = pickle.load(f)
        
        models['ssvep'] = {
            'model': ssvep_model_data['model'],
            'label_encoder': ssvep_model_data['label_encoder'],
            'loader': ssvep_loader,
            'config': ssvep_model_data['config']
        }
        print(f"SSVEP model loaded from {ssvep_model_name}_ssvep_model.pkl")
    else:
        print(f"Warning: SSVEP model not found at {ssvep_model_path} or {ssvep_loader_path}")
    
    # Load MI model (PyTorch model)
    mi_model_files = [f for f in os.listdir(models_dir) if f.startswith(mi_model_name) and f.endswith('.pth')]
    
    if mi_model_files:
        mi_model_path = os.path.join(models_dir, mi_model_files[0])
        
        # Load model checkpoint
        checkpoint = torch.load(mi_model_path, map_location='cpu', weights_only=False)
        
        # Extract model info from checkpoint
        architecture = checkpoint.get('architecture', 'SimpleNet')
        channels = checkpoint.get('channels', 3)
        samples = checkpoint.get('samples', 500)
        
        from configs.mi_config import MIConfig
        from models.model_factory import create_mi_model
        
        # Load model configuration
        mi_config = MIConfig()
        mi_config.update('model.architecture', architecture)
        
        # Create model architecture
        model = create_mi_model(mi_config, channels, samples)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        models['mi'] = {
            'model': model,
            'config': mi_config,
            'channels': channels,
            'samples': samples
        }
        print(f"MI model loaded from {mi_model_files[0]}")
    else:
        print(f"Warning: MI model not found with prefix {mi_model_name}")
    
    return models

def generate_predictions(test_data, models, data_path):
    """Generate predictions for both tasks"""
    
    predictions = {}
    
    # SSVEP predictions
    if 'ssvep' in models and len(test_data['ssvep']) > 0:
        print("Running SSVEP inference...")
        
        ssvep_model = models['ssvep']['model']
        ssvep_loader = models['ssvep']['loader']
        label_encoder = models['ssvep']['label_encoder']
        
        # Load and preprocess SSVEP test data
        from data.ssvep_loader import load_ssvep_test_data
        X_test_ssvep = load_ssvep_test_data(
            test_data['ssvep'], data_path, ssvep_loader.config, ssvep_loader
        )
        
        if len(X_test_ssvep) > 0:
            # Generate predictions
            ssvep_preds_encoded = ssvep_model.predict(X_test_ssvep)
            # Convert back to original labels
            ssvep_preds = label_encoder.inverse_transform(ssvep_preds_encoded)
            
            predictions['ssvep'] = {
                'ids': test_data['ssvep']['id'].values[:len(ssvep_preds)],
                'predictions': ssvep_preds
            }
            print(f"SSVEP inference completed: {len(ssvep_preds)} predictions")
        else:
            print("Warning: No SSVEP test data could be loaded")
            predictions['ssvep'] = {'ids': [], 'predictions': []}
    
    # MI predictions
    if 'mi' in models and len(test_data['mi']) > 0:
        print("Running MI inference...")
        
        mi_model = models['mi']['model']
        mi_config = models['mi']['config']
        
        # Load and preprocess MI test data
        from data.mi_loader import load_mi_test_data
        X_test_mi = load_mi_test_data(test_data['mi'], data_path, mi_config)
        
        if len(X_test_mi) > 0:
            # Create test dataset and loader
            from training.mi_trainer import EEGDataset
            test_dataset = EEGDataset(X_test_mi)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            mi_preds = []
            mi_model.eval()
            with torch.no_grad():
                for batch_x in test_loader:
                    if batch_x.ndim == 3:
                        batch_x = batch_x.unsqueeze(1)
                    outputs = mi_model(batch_x)
                    preds = outputs.argmax(dim=1)
                    mi_preds.append(preds.cpu().numpy())
            
            mi_preds = np.concatenate(mi_preds) if mi_preds else np.array([])
            
            # Convert to label strings using MI config
            inverse_mapping = mi_config.get('labels.inverse_mapping')
            mi_labels = [inverse_mapping.get(pred, 'Unknown') for pred in mi_preds]
            
            predictions['mi'] = {
                'ids': test_data['mi']['id'].values[:len(mi_labels)],
                'predictions': mi_labels
            }
            print(f"MI inference completed: {len(mi_labels)} predictions")
        else:
            print("Warning: No MI test data could be loaded")
            predictions['mi'] = {'ids': [], 'predictions': []}
    
    return predictions

def create_submission_csv(predictions, output_file):
    """Create combined submission CSV"""
    
    # Combine predictions from both tasks
    all_ids = []
    all_labels = []
    
    # Add SSVEP predictions
    if 'ssvep' in predictions:
        all_ids.extend(predictions['ssvep']['ids'])
        all_labels.extend(predictions['ssvep']['predictions'])
    
    # Add MI predictions
    if 'mi' in predictions:
        all_ids.extend(predictions['mi']['ids'])
        all_labels.extend(predictions['mi']['predictions'])
    
    if not all_ids:
        print("Warning: No predictions generated!")
        # Create empty submission
        submission_df = pd.DataFrame({'id': [], 'label': []})
    else:
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': all_ids,
            'label': all_labels
        })
        
        # Sort by id to match expected format
        submission_df = submission_df.sort_values('id').reset_index(drop=True)
    
    # Save to CSV
    submission_df.to_csv(output_file, index=False)
    
    print(f"Submission contains {len(submission_df)} predictions")
    if 'ssvep' in predictions:
        print(f"    - SSVEP: {len(predictions['ssvep']['ids'])} samples")
    if 'mi' in predictions:
        print(f"    - MI: {len(predictions['mi']['ids'])} samples")