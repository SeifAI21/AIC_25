"""
Evaluation command implementation
Loads test data, runs inference with trained models, generates combined CSV
"""
import os
import sys
import pandas as pd

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
    
    #TODO: Implement actual model loading logic
    print("Loading trained models...")
    models = {}
    
    return models

def generate_predictions(test_data, models):
    """Generate predictions for both tasks"""
    
    #TODO: Implement actual prediction logic and add timing to time inference
    predictions = {}

    # SSVEP predictions
    print("Running SSVEP inference...")

    # MI predictions
    print("Running MI inference...")    
    
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