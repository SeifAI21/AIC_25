#!/usr/bin/env python3
"""
Main CLI for EEG Classification Framework
Supports two modes: train and evaluate
"""
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
def main():
    parser = argparse.ArgumentParser(description='EEG Classification Framework')
    subparsers = parser.add_subparsers(dest='mode', help='Available modes')
    
    # Train mode
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--task', choices=['SSVEP', 'MI', 'BOTH'], required=True,
                             help='Task to train: SSVEP, MI, or BOTH')
    train_parser.add_argument('--data-path', type=str, default='/kaggle/input/mtcaic3',
                             help='Path to dataset')
    train_parser.add_argument('--config-dir', type=str, default='configs/',
                             help='Directory containing config files')
    train_parser.add_argument('--output-dir', type=str, default='models/weights/',
                             help='Directory to save trained models')
    train_parser.add_argument('--parallel', action='store_true',
                             help='Train SSVEP and MI in parallel (when task=BOTH)')
    train_parser.add_argument('--ssvep-model-name', type=str, default='ssvep_model',
                             help='Name for SSVEP model weights (without extension)')
    train_parser.add_argument('--mi-model-name', type=str, default='mi_model',
                             help='Name for MI model weights (without extension)')
    
    # Evaluate mode  
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models and generate submission')
    eval_parser.add_argument('--data-path', type=str, default='/kaggle/input/mtcaic3',
                            help='Path to test dataset')
    eval_parser.add_argument('--models-dir', type=str, default='models/weights/',
                            help='Directory containing trained models')
    eval_parser.add_argument('--output-file', type=str, default='submission.csv',
                            help='Output CSV file name')
    eval_parser.add_argument('--ssvep-model-name', type=str, default='ssvep_model',
                            help='Name of SSVEP model to load (without extension)')
    eval_parser.add_argument('--mi-model-name', type=str, default='mi_model',
                            help='Name of MI model to load (without extension)')
    
    args = parser.parse_args()
    
    # Ensure output directories exist (important for Kaggle)
    if args.mode == 'train':
        os.makedirs(args.output_dir, exist_ok=True)
    elif args.mode == 'evaluate':
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    if args.mode == 'train':
        from cli.train_command import run_training
        run_training(args)
    elif args.mode == 'evaluate':
        from cli.predict_command import run_evaluation
        run_evaluation(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()