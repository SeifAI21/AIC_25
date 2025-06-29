"""
Training command implementation
Handles training for SSVEP, MI, or both tasks
"""
import os
import sys

def run_training(args):
    """
    Main training function
    Args:
        args: Parsed command line arguments
    """
    print(f"Starting training for task: {args.task}")
    print(f"Data path: {args.data_path}")
    print(f"Config directory: {args.config_dir}")
    print(f"Output directory: {args.output_dir}")
    
    if args.task == 'SSVEP' or args.task == 'BOTH':
        print(f"SSVEP model name: {args.ssvep_model_name}")
    if args.task == 'MI' or args.task == 'BOTH':
        print(f"MI model name: {args.mi_model_name}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.task == 'SSVEP':
        train_ssvep(args)
    elif args.task == 'MI':
        train_mi(args)
    elif args.task == 'BOTH':
        if args.parallel:
            train_both_parallel(args)
        else:
            train_both_sequential(args)

    print("Training completed successfully!")

def train_ssvep(args):
    """Train SSVEP model"""
    
    # Load SSVEP config
    from configs.ssvep_config import SSVEPConfig
    config = SSVEPConfig()
    
    # Load SSVEP data and train
    from training.ssvep_trainer import SSVEPTrainer
    model_name = getattr(args, 'ssvep_model_name', 'ssvep_model')
    trainer = SSVEPTrainer(config, args.data_path, args.output_dir, model_name)
    trainer.train()
    
def train_mi(args):
    """Train MI model"""
    
    # Load MI config
    from configs.mi_config import MIConfig
    config = MIConfig()
    
    # Load MI data and train
    from training.mi_trainer import MITrainer
    model_name = getattr(args, 'mi_model_name', 'mi_model')
    trainer = MITrainer(config, args.data_path, args.output_dir, model_name)
    trainer.train()
    
def train_both_sequential(args):
    """Train both models sequentially"""
    train_ssvep(args)
    train_mi(args)
    
# Extra feature for now
def train_both_parallel(args):
    """Train both models in parallel"""
    # from training.parallel_trainer import ParallelTrainer
    # trainer = ParallelTrainer(args.data_path, args.config_dir, args.output_dir)
    # trainer.train_parallel()
    print("Parallel training is not implemented yet. Please train sequentially.")