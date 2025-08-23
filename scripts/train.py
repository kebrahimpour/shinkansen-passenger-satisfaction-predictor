#!/usr/bin/env python3
"""Training script for Shinkansen passenger satisfaction predictor.

This script loads training data from a CSV file, trains a SatisfactionPredictor model,
and saves the trained model to the models directory.

Usage:
    python scripts/train.py [--data-path PATH] [--model-path PATH] [--verbose]
    uv run python scripts/train.py

Example:
    python scripts/train.py --data-path data/synthetic_shinkansen.csv
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import pandas as pd
    from shinkansen_predictor import SatisfactionPredictor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to install dependencies with: uv sync")
    sys.exit(1)


def load_training_data(data_path: str, verbose: bool = False) -> tuple:
    """Load and prepare training data from CSV file.
    
    Args:
        data_path: Path to the CSV file containing training data
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (X_train, y_train) where X_train is list of dicts
        and y_train is list of satisfaction scores
    """
    if verbose:
        print(f"📊 Loading training data from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        if verbose:
            print(f"✅ Loaded {len(df)} training samples")
            print(f"📋 Columns: {list(df.columns)}")
        
        # Validate required columns
        required_features = ['duration', 'service_class', 'on_time_performance', 
                           'weather_condition', 'seat_occupancy']
        target_column = 'satisfaction_score'
        
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")
        
        if target_column not in df.columns:
            raise ValueError(f"Missing target column: {target_column}")
        
        # Prepare features (X) as list of dictionaries
        X_train = []
        for _, row in df.iterrows():
            sample = {
                'duration': float(row['duration']),
                'service_class': str(row['service_class']),
                'on_time_performance': float(row['on_time_performance']),
                'weather_condition': str(row['weather_condition']),
                'seat_occupancy': float(row['seat_occupancy'])
            }
            X_train.append(sample)
        
        # Prepare targets (y)
        y_train = df[target_column].tolist()
        
        if verbose:
            print(f"🎯 Target range: {min(y_train):.2f} - {max(y_train):.2f}")
            print(f"📈 Target mean: {sum(y_train)/len(y_train):.2f}")
        
        return X_train, y_train
        
    except Exception as e:
        raise RuntimeError(f"Error loading training data: {e}")


def train_model(X_train: list, y_train: list, verbose: bool = False) -> SatisfactionPredictor:
    """Train a SatisfactionPredictor model.
    
    Args:
        X_train: List of feature dictionaries
        y_train: List of satisfaction scores
        verbose: Whether to print verbose output
        
    Returns:
        Trained SatisfactionPredictor instance
    """
    if verbose:
        print("🤖 Initializing SatisfactionPredictor...")
    
    predictor = SatisfactionPredictor()
    
    if verbose:
        print(f"🏋️ Training model on {len(X_train)} samples...")
    
    try:
        predictor.fit(X_train, y_train)
        
        if verbose:
            print("✅ Model training completed successfully!")
            
            # Test prediction on first sample
            test_prediction = predictor.predict(X_train[0])
            print(f"🧪 Test prediction on first sample: {test_prediction:.3f}")
            print(f"🎯 Actual satisfaction: {y_train[0]:.3f}")
            
        return predictor
        
    except Exception as e:
        raise RuntimeError(f"Error during model training: {e}")


def save_model(predictor: SatisfactionPredictor, model_path: str, verbose: bool = False):
    """Save the trained model to disk.
    
    Args:
        predictor: Trained SatisfactionPredictor instance
        model_path: Path where to save the model
        verbose: Whether to print verbose output
    """
    if verbose:
        print(f"💾 Saving model to {model_path}...")
    
    # Create models directory if it doesn't exist
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        predictor.save_model(model_path)
        
        if verbose:
            print("✅ Model saved successfully!")
            file_size = os.path.getsize(model_path)
            print(f"📁 Model file size: {file_size:,} bytes")
            
    except Exception as e:
        raise RuntimeError(f"Error saving model: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Shinkansen passenger satisfaction predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py
  python scripts/train.py --data-path data/my_data.csv
  python scripts/train.py --model-path models/custom_model.pkl --verbose
  uv run python scripts/train.py --verbose
"""
    )
    
    parser.add_argument(
        "--data-path",
        default="data/synthetic_shinkansen.csv",
        help="Path to training data CSV file (default: data/synthetic_shinkansen.csv)"
    )
    
    parser.add_argument(
        "--model-path", 
        default="models/model.pkl",
        help="Path to save trained model (default: models/model.pkl)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    print("🚅 Shinkansen Passenger Satisfaction Predictor - Training Script")
    print("=" * 65)
    
    try:
        # Load training data
        X_train, y_train = load_training_data(args.data_path, args.verbose)
        
        # Train model
        predictor = train_model(X_train, y_train, args.verbose)
        
        # Save model
        save_model(predictor, args.model_path, args.verbose)
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Trained on {len(X_train)} samples")
        print(f"💾 Model saved to: {args.model_path}")
        print("\n🚀 You can now use the model with:")
        print("   python -c \"from shinkansen_predictor import SatisfactionPredictor; \\")
        print(f"   print(f'   p = SatisfactionPredictor(); p.load_model(\"{args.model_path}\")\"')")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
