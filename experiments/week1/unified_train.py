"""
Unified Math OCR Training Script
- Can start fresh or resume from checkpoint
- Automatically saves progress and retains learned knowledge
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import argparse

# Import our modules
from src.data.vocabulary import MathVocabulary
from src.data.dataloader import CROHMEDataset, collate_fn


class SimpleModel(nn.Module):
    """Simple CNN + LSTM model for Math OCR"""
    
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # LSTM decoder
        self.cnn_proj = nn.Linear(512 * 7 * 7, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, images, target_tokens=None):
        batch_size = images.size(0)
        
        # CNN encoding
        cnn_features = self.cnn(images)
        cnn_features = cnn_features.view(batch_size, -1)
        image_context = self.cnn_proj(cnn_features)
        
        if target_tokens is not None:
            # Training mode
            token_embeds = self.embedding(target_tokens)
            h0 = image_context.unsqueeze(0)
            c0 = torch.zeros_like(h0)
            lstm_out, _ = self.lstm(token_embeds, (h0, c0))
            logits = self.output_proj(lstm_out)
            return logits
        else:
            return None


def setup_device():
    """Setup the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🍎 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using NVIDIA GPU")
    else:
        device = torch.device('cpu')
        print(f"💻 Using CPU")
    return device


def load_dataset(data_path, vocab, num_samples, use_subset=True):
    """Load dataset with optional subset"""
    dataset = CROHMEDataset(data_path, vocab, phase='train', img_size=(224, 224))
    
    if use_subset and num_samples < len(dataset):
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)
    
    return dataset


def train_epoch(model, dataloader, criterion, optimizer, device, vocab):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        images = batch['images'].to(device)
        tokens = batch['tokens'].to(device)
        
        input_tokens = tokens[:, :-1]
        target_tokens = tokens[:, 1:]
        
        optimizer.zero_grad()
        logits = model(images, input_tokens)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(logits, dim=-1)
        mask = (target_tokens != vocab.pad_id)
        correct_tokens += ((predictions == target_tokens) & mask).sum().item()
        total_tokens += mask.sum().item()
        
        total_loss += loss.item()
        
        accuracy = 100 * correct_tokens / total_tokens if total_tokens > 0 else 0
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.1f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = 100 * correct_tokens / total_tokens
    
    return avg_loss, avg_accuracy


def save_checkpoint(model, optimizer, epoch, accuracy, config, checkpoint_path):
    """Save model checkpoint with all training state"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
        'vocab_size': config['vocab_size'],
        'hidden_dim': config['hidden_dim'],
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load checkpoint and restore training state"""
    if not os.path.exists(checkpoint_path):
        return None, 0, 0.0
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore optimizer state (retains learning rate schedule, momentum, etc.)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_accuracy = checkpoint.get('accuracy', 0.0)
    
    print(f"✅ Resumed from epoch {start_epoch}, best accuracy: {best_accuracy:.2f}%")
    return checkpoint, start_epoch, best_accuracy


def find_checkpoint():
    """Find the best available checkpoint"""
    checkpoint_paths = [
        "outputs/models/unified_model.pth",      # Unified script checkpoint
        "outputs/models/week1_best_model.pth"    # Week 1 script checkpoint
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Math OCR Training")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to train on")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--target", type=float, default=90.0, help="Target accuracy")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint (default: True)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore checkpoints)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    
    args = parser.parse_args()
    
    print("🚀 Unified Math OCR Training")
    print("=" * 40)
    print(f"📋 Config: {args.samples} samples, {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"🎯 Target accuracy: {args.target}%")
    
    # Setup device
    if args.device == "auto":
        device = setup_device()
    else:
        device = torch.device(args.device)
        print(f"🔧 Using specified device: {device}")
    
    # Create vocabulary
    vocab = MathVocabulary()
    
    # Model configuration
    config = {
        'vocab_size': len(vocab),
        'hidden_dim': 256,
        'num_samples': args.samples,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'target_accuracy': args.target
    }
    
    # Create model
    model = SimpleModel(config['vocab_size'], config['hidden_dim']).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Handle checkpoint loading
    start_epoch = 0
    best_accuracy = 0.0
    checkpoint = None
    save_path = "outputs/models/unified_model.pth"
    
    if args.fresh:
        print("🆕 Starting fresh training (--fresh flag used)")
    elif args.resume:
        checkpoint_path = find_checkpoint()
        if checkpoint_path:
            print(f"📂 Found checkpoint: {checkpoint_path}")
            checkpoint, start_epoch, best_accuracy = load_checkpoint(checkpoint_path, model, optimizer, device)
        else:
            print("🆕 No checkpoint found, starting fresh")
    else:
        print("🆕 Resume disabled, starting fresh")
    
    if checkpoint is None:
        print("🔧 New training session initialized")
    else:
        print(f"🔄 Continuing training from {best_accuracy:.2f}% accuracy")
    
    # Load dataset
    data_path = Path("data/raw/crohme")
    dataset = load_dataset(data_path, vocab, args.samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"📊 Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    
    # Training loop
    print(f"\n🏋️ Training from epoch {start_epoch + 1}...")
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        current_epoch = epoch + 1
        print(f"\nEpoch {current_epoch}")
        
        # Train one epoch
        loss, accuracy = train_epoch(model, dataloader, criterion, optimizer, device, vocab)
        print(f"📊 Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save if improved
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"🎉 New best accuracy: {accuracy:.2f}%!")
            
            # Save checkpoint (retains ALL learned knowledge)
            save_checkpoint(model, optimizer, current_epoch, accuracy, config, save_path)
            print(f"💾 Checkpoint saved: {save_path}")
        
        # Check if target reached
        if accuracy >= args.target:
            print(f"\n🎯 TARGET ACHIEVED!")
            print(f"   Target: >{args.target}% accuracy")
            print(f"   Achieved: {accuracy:.2f}% accuracy")
            print(f"   Device: {device}")
            break
    
    # Final results
    print(f"\n🏁 Training Complete!")
    print(f"📊 Best Accuracy: {best_accuracy:.2f}%")
    print(f"💾 Model saved with ALL learned knowledge retained")
    print(f"🔄 Run again to continue training automatically")
    
    # Save training history
    history_path = "outputs/results/unified_training_history.json"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    with open(history_path, 'w') as f:
        json.dump({
            'final_accuracy': best_accuracy,
            'total_epochs': current_epoch,
            'config': config,
            'device': str(device),
            'resumed_from': checkpoint_path if checkpoint else None,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"📈 History saved: {history_path}")
    
    if best_accuracy >= args.target:
        print(f"\n✅ Week 1 COMPLETE! Ready for Week 2!")
    else:
        print(f"\n🔄 Run the same command again to continue from {best_accuracy:.2f}%")
        print(f"   python experiments/week1/unified_train.py --samples {args.samples} --epochs {args.epochs}")
        print(f"\n💡 Or try: python experiments/week1/unified_train.py --samples 100 --epochs 5 --target {args.target}")


if __name__ == "__main__":
    main()