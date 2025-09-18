"""
Main entry point for Math OCR training pipeline
Orchestrates data loading, model creation, and training
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import random
import numpy as np
from omegaconf import OmegaConf
import wandb
import argparse
from datetime import datetime

# Import project modules
from src.data.dataloader import get_dataloaders
from src.models.math_ocr_model import create_model
from src.training.trainer import create_trainer


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_device():
    """Setup and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🍎 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print(f"💻 Using CPU")
    
    return device


def create_experiment_name(cfg):
    """Create a unique experiment name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backbone = cfg.model.encoder.backbone
    batch_size = cfg.training.batch_size
    lr = cfg.training.learning_rate
    
    return f"{cfg.experiment.name}_{backbone}_bs{batch_size}_lr{lr}_{timestamp}"


def main(config_path: str = None):
    """Main training pipeline"""
    
    print("🚀 Math OCR Training Pipeline Starting...")
    print("=" * 50)
    
    # Load configuration
    if config_path is None:
        config_path = "configs/config.yaml"
    
    cfg = OmegaConf.load(config_path)
    print(f"📋 Loaded config from: {config_path}")
    
    # Set random seed
    set_seed(cfg.experiment.seed)
    print(f"🎲 Random seed set to: {cfg.experiment.seed}")
    
    # Setup device
    device = setup_device()
    
    # Create experiment name
    experiment_name = create_experiment_name(cfg)
    print(f"🧪 Experiment: {experiment_name}")
    
    # Initialize wandb (if enabled)
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project,
            name=experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[cfg.model.encoder.backbone, f"bs_{cfg.training.batch_size}"]
        )
        print("📊 Weights & Biases initialized")
    
    try:
        # Load datasets
        print("\n📚 Loading datasets...")
        train_loader, val_loader, test_loader, vocab = get_dataloaders(cfg)
        
        print(f"✅ Datasets loaded:")
        print(f"   📖 Vocabulary size: {len(vocab)}")
        print(f"   🏋️ Train samples: {len(train_loader.dataset)}")
        print(f"   ✅ Validation samples: {len(val_loader.dataset)}")
        print(f"   🧪 Test samples: {len(test_loader.dataset)}")
        
        # Save vocabulary
        vocab_path = Path(cfg.paths.model_dir) / "vocabulary.json"
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save_vocab(str(vocab_path))
        print(f"💾 Vocabulary saved to: {vocab_path}")
        
        # Create model
        print(f"\n🏗️ Creating model...")
        model_config = {
            'backbone': cfg.model.encoder.backbone,
            'hidden_dim': cfg.model.decoder.hidden_dim,
            'num_layers': cfg.model.decoder.num_layers,
            'num_heads': cfg.model.decoder.num_heads,
            'ff_dim': cfg.model.decoder.get('ff_dim', 2048),
            'dropout': cfg.model.decoder.dropout,
            'max_seq_len': cfg.model.max_seq_length,
            'pretrained': cfg.model.encoder.pretrained
        }
        
        model = create_model(len(vocab), model_config)
        print(f"✅ Model created: {model.__class__.__name__}")
        
        # Create trainer
        print(f"\n🏋️ Setting up trainer...")
        trainer = create_trainer(model, train_loader, val_loader, vocab, cfg, device)
        
        # Check if resuming from checkpoint
        checkpoint_path = Path(cfg.paths.model_dir) / "latest_checkpoint.pth"
        if checkpoint_path.exists():
            print(f"📂 Found existing checkpoint: {checkpoint_path}")
            response = input("Resume from checkpoint? (y/n): ").lower()
            if response == 'y':
                trainer.load_checkpoint(str(checkpoint_path))
        
        # Start training
        print(f"\n🚀 Starting training...")
        print(f"   Epochs: {cfg.training.num_epochs}")
        print(f"   Batch size: {cfg.training.batch_size}")
        print(f"   Learning rate: {cfg.training.learning_rate}")
        print(f"   Early stopping patience: {cfg.training.early_stopping_patience}")
        
        best_model_path = trainer.train()
        
        # Evaluate on test set
        print(f"\n🧪 Final evaluation...")
        test_results = trainer.evaluate(test_loader, best_model_path)
        
        # Save final results
        results_path = Path(cfg.paths.output_dir) / "results" / f"{experiment_name}_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_path}")
        
        # Final summary
        print(f"\n🎉 Training Complete!")
        print(f"=" * 50)
        print(f"📊 Final Results:")
        print(f"   🎯 Best Sequence Accuracy: {test_results['sequence_accuracy']:.2f}%")
        print(f"   📝 Token Accuracy: {test_results['token_accuracy']:.2f}%")
        print(f"   💾 Best Model: {best_model_path}")
        print(f"   📈 Results: {results_path}")
        
        if cfg.logging.use_wandb:
            wandb.log({
                'final/test_sequence_accuracy': test_results['sequence_accuracy'],
                'final/test_token_accuracy': test_results['token_accuracy'],
                'final/test_loss': test_results['test_loss']
            })
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        if cfg.logging.use_wandb:
            wandb.finish(exit_code=1)
        
        return False
    
    finally:
        if cfg.logging.use_wandb:
            wandb.finish()
    
    return True


def quick_test():
    """Quick test to verify all components work together"""
    print("🧪 Running Quick Test...")
    
    # Test imports
    try:
        from src.data.vocabulary import MathVocabulary
        from src.data.dataloader import CROHMEDataset
        from src.models.math_ocr_model import create_model
        from src.training.trainer import MathOCRTrainer
        
        print("✅ All imports successful")
        
        # Test vocabulary
        vocab = MathVocabulary()
        print(f"✅ Vocabulary created: {len(vocab)} tokens")
        
        # Test model creation
        model = create_model(len(vocab))
        print(f"✅ Model created successfully")
        
        # Test forward pass
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        tokens = torch.randint(0, len(vocab), (batch_size, 20))
        
        with torch.no_grad():
            output = model(images, tokens)
            print(f"✅ Forward pass: {output.shape}")
        
        print("🎉 Quick test passed! Ready for training.")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Math OCR Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--test", action="store_true",
                       help="Run quick test instead of training")
    
    args = parser.parse_args()
    
    if args.test:
        success = quick_test()
    else:
        success = main(args.config)
    
    exit(0 if success else 1)