"""
Evaluation command implementation
Loads test data, runs inference with trained models, generates combined CSV
"""
import os
import sys
import pandas as pd
import numpy as np

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
    predictions = generate_predictions(test_data, models)
    
    # Create combined submission
    create_submission_csv(predictions, args.output_file)
    
    print(f"Evaluation completed. Submission saved to: {args.output_file}")

def load_test_data(data_path):
    """Load test data for both SSVEP and MI tasks"""
    print("Loading test data...")
    
    # Load test metadata
    test_df = pd.read_csv(f"{data_path}/test.csv")
    
    # Separate SSVEP and MI test data
    ssvep_test = test_df[test_df['task'] == 'SSVEP'].copy()
    mi_test = test_df[test_df['task'] == 'MI'].copy()
    
    # Load and preprocess actual EEG data
    from data.ssvep_loader import load_ssvep_test_data
    from data.mi_loader import load_mi_test_data
    
    ssvep_data = load_ssvep_test_data(ssvep_test, data_path)
    mi_data = load_mi_test_data(mi_test, data_path)
    
    return {
        'ssvep': {'metadata': ssvep_test, 'data': ssvep_data},
        'mi': {'metadata': mi_test, 'data': mi_data}
    }

def load_trained_models(models_dir, ssvep_model_name, mi_model_name):
    """Load trained SSVEP and MI models"""
    
    models = {}
    
    # Load SSVEP model (classical ML model saved as pickle)
    ssvep_model_path = os.path.join(models_dir, f'{ssvep_model_name}.pkl')
    if os.path.exists(ssvep_model_path):
        import pickle
        with open(ssvep_model_path, 'rb') as f:
            models['ssvep'] = pickle.load(f)
        print(f"SSVEP model loaded from {ssvep_model_name}.pkl")
    else:
        print(f"Warning: SSVEP model not found at {ssvep_model_path}")
    
    # Load MI model (PyTorch model)
    mi_model_path = os.path.join(models_dir, f'{mi_model_name}.pth')
    if os.path.exists(mi_model_path):
        import torch
        from Code.models.model_factory import create_mi_model
        from configs.mi_config import MIConfig
        
        # Load model configuration
        mi_config = MIConfig()
        
        # Create model architecture
        # Note: In production, you should save model metadata with dimensions
        model = create_mi_model(mi_config, channels=3, samples=500)
        model.load_state_dict(torch.load(mi_model_path, map_location='cpu'))
        model.eval()
        models['mi'] = model
        print(f"MI model loaded from {mi_model_name}.pth")
    else:
        print(f"Warning: MI model not found at {mi_model_path}")
    
    return models

def generate_predictions(test_data, models):
    """Generate predictions for both tasks"""
    
    predictions = {}
    
    # SSVEP predictions
    if 'ssvep' in models and 'ssvep' in test_data:
        print("Running SSVEP inference...")
        ssvep_model = models['ssvep']
        ssvep_data = test_data['ssvep']['data']
        
        # Generate SSVEP predictions
        ssvep_preds = ssvep_model.predict(ssvep_data)
        predictions['ssvep'] = {
            'ids': test_data['ssvep']['metadata']['id'].values,
            'predictions': ssvep_preds
        }
        print(f"SSVEP inference completed: {len(ssvep_preds)} predictions")
    
    # MI predictions
    if 'mi' in models and 'mi' in test_data:
        print("Running MI inference...")
        mi_model = models['mi']
        mi_data = test_data['mi']['data']
        
        # Generate MI predictions using PyTorch model
        import torch
        from Code.models.model_factory import EEGDataset
        from torch.utils.data import DataLoader
        
        # Create test dataset and loader
        test_dataset = EEGDataset(mi_data)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        mi_preds = []
        mi_model.eval()
        with torch.no_grad():
            for batch_x in test_loader:
                if isinstance(batch_x, (list, tuple)):
                    batch_x = batch_x[0]
                if batch_x.ndim == 3:
                    batch_x = batch_x.unsqueeze(1)
                outputs = mi_model(batch_x)
                preds = outputs.argmax(dim=1)
                mi_preds.append(preds.cpu().numpy())
        
        mi_preds = np.concatenate(mi_preds) if mi_preds else np.array([])
        
        # Convert to label strings
        label_mapping = {0: 'Left', 1: 'Right'}
        mi_labels = [label_mapping.get(pred, 'Unknown') for pred in mi_preds]
        
        predictions['mi'] = {
            'ids': test_data['mi']['metadata']['id'].values,
            'predictions': mi_labels
        }
        print(f"MI inference completed: {len(mi_labels)} predictions")
    
    return predictions

def create_submission_csv(predictions, output_file):
    """Create combined submission CSV"""
    
    # Combine predictions from both tasks
    all_ids = []
    all_labels = []
    
    # Add SSVEP predictions
    all_ids.extend(predictions['ssvep']['ids'])
    all_labels.extend(predictions['ssvep']['predictions'])
    
    # Add MI predictions
    all_ids.extend(predictions['mi']['ids'])
    all_labels.extend(predictions['mi']['predictions'])
    
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
    print(f"    - SSVEP: {len(predictions['ssvep']['ids'])} samples")
    print(f"    - MI: {len(predictions['mi']['ids'])} samples")