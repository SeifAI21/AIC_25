#!/usr/bin/env python3
"""
Main entry point for EEG Classification Framework
Usage:
    python main.py train --task SSVEP
    python main.py train --task MI  
    python main.py train --task BOTH --parallel
    python main.py evaluate
"""
import sys
import os

# Add current directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import and run CLI
from cli.main_cli import main

if __name__ == "__main__":
    try:
        main()
        print("Program completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Clean exit for Kaggle notebooks
        if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            print("Kaggle environment detected - forcing clean exit")
            os._exit(0)
        else:
            sys.exit(0)