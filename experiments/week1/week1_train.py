"""
Week 1 Training Script - Simple Math OCR Training
Goal: Train on 1000 samples, achieve >90% accuracy through overfitting
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Import our modules
from src.data.vocabulary import MathVocabulary
from src.data.dataloader import CROHMEDataset, collate_fn


class SimpleModel(nn.Module):
    """Simplified model for Week 1 - CNN + LSTM instead of full Transformer"""
    
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Simple CNN encoder
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block  
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # Fixed output size
        )
        
        # Project CNN features
        self.cnn_proj = nn.Linear(512 * 7 * 7, hidden_dim)
        
        # LSTM decoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        print(f"🧠 SimpleModel created: {vocab_size} vocab, {hidden_dim} hidden")
        
    def forward(self, images, target_tokens=None):
        batch_size = images.size(0)
        
        # CNN encoding
        cnn_features = self.cnn(images)  # [B, 512, 7, 7]
        cnn_features = cnn_features.view(batch_size, -1)  # [B, 512*7*7]
        image_context = self.cnn_proj(cnn_features)  # [B, hidden_dim]
        
        if target_tokens is not None:
            # Training mode with teacher forcing
            seq_len = target_tokens.size(1)
            
            # Token embeddings
            token_embeds = self.embedding(target_tokens)  # [B, seq_len, hidden_dim]
            
            # Initialize LSTM with image context
            h0 = image_context.unsqueeze(0)  # [1, B, hidden_dim]
            c0 = torch.zeros_like(h0)
            
            # LSTM forward
            lstm_out, _ = self.lstm(token_embeds, (h0, c0))  # [B, seq_len, hidden_dim]
            
            # Project to vocabulary
            logits = self.output_proj(lstm_out)  # [B, seq_len, vocab_size]
            
            return logits
        else:
            # Inference mode (not used in Week 1)
            return None


def load_small_dataset(data_path, vocab, num_samples=1000):
    """Load a small subset of CROHME data for Week 1"""
    print(f"📊 Loading {num_samples} samples from CROHME...")
    
    # Load full dataset
    full_dataset = CROHMEDataset(
        data_path=data_path,
        vocab=vocab,
        phase='train',
        img_size=(224, 224)
    )
    
    # Create small subset
    indices = list(range(min(num_samples, len(full_dataset))))
    small_dataset = Subset(full_dataset, indices)
    
    print(f"✅ Loaded {len(small_dataset)} samples")
    return small_dataset


def train_epoch(model, dataloader, criterion, optimizer, device, vocab):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        # Move to device
        images = batch['images'].to(device)
        tokens = batch['tokens'].to(device)
        
        # Prepare input/target tokens for teacher forcing
        input_tokens = tokens[:, :-1]  # All but last
        target_tokens = tokens[:, 1:]  # All but first
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images, input_tokens)
        
        # Calculate loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        mask = (target_tokens != vocab.pad_id)  # Ignore padding
        correct_tokens += ((predictions == target_tokens) & mask).sum().item()
        total_tokens += mask.sum().item()
        
        total_loss += loss.item()
        
        # Update progress
        accuracy = 100 * correct_tokens / total_tokens if total_tokens > 0 else 0
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.1f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = 100 * correct_tokens / total_tokens
    
    return avg_loss, avg_accuracy


def main():
    """Main training function"""
    print("🚀 Week 1 Math OCR Training")
    print("=" * 40)
    
    # Configuration
    NUM_SAMPLES = 1000  # Small dataset for Week 1
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    HIDDEN_DIM = 256
    
    print(f"📋 Config: {NUM_SAMPLES} samples, {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}")
    
    # Setup device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon GPU
        print(f"🍎 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using NVIDIA GPU")
    else:
        device = torch.device('cpu')
        print(f"💻 Using CPU")

    print(f"🖥️  Device: {device}")
    
    # Create vocabulary
    vocab = MathVocabulary()
    print(f"📚 Vocabulary: {len(vocab)} tokens")
    
    # Load dataset
    data_path = Path("data/raw/crohme")
    if not data_path.exists():
        print(f"❌ Dataset not found at {data_path}")
        print("   Make sure you've downloaded CROHME dataset!")
        return
    
    dataset = load_small_dataset(data_path, vocab, NUM_SAMPLES)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    print(f"🔄 DataLoader: {len(dataloader)} batches")
    
    # Create model
    model = SimpleModel(len(vocab), HIDDEN_DIM).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Model parameters: {total_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n🏋️ Starting training...")
    best_accuracy = 0.0
    history = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train one epoch
        loss, accuracy = train_epoch(model, dataloader, criterion, optimizer, device, vocab)
        
        # Log results
        print(f"📊 Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'loss': loss,
            'accuracy': accuracy
        })
        
        # Check if best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"🎉 New best accuracy: {accuracy:.2f}%!")
            
            # Save best model
            model_path = "outputs/models/week1_best_model.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': len(vocab),
                'hidden_dim': HIDDEN_DIM,
                'accuracy': accuracy,
                'epoch': epoch + 1,
                'config': {
                    'num_samples': NUM_SAMPLES,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE
                }
            }, model_path)
            
            print(f"💾 Model saved: {model_path}")
        
        # Check if we achieved Week 1 goal
        if accuracy > 90.0:
            print(f"\n🎯 WEEK 1 GOAL ACHIEVED!")
            print(f"   Target: >90% accuracy")
            print(f"   Achieved: {accuracy:.2f}% accuracy")
            break
    
    # Final results
    print(f"\n🏁 Training Complete!")
    print(f"=" * 40)
    print(f"📊 Final Results:")
    print(f"   Best Accuracy: {best_accuracy:.2f}%")
    print(f"   Target: >90% (Week 1 goal)")
    print(f"   Status: {'✅ ACHIEVED' if best_accuracy > 90 else '⚠️  Close! Try more epochs'}")
    
    # Save training history
    history_path = "outputs/results/week1_training_history.json"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    
    with open(history_path, 'w') as f:
        json.dump({
            'history': history,
            'best_accuracy': best_accuracy,
            'config': {
                'num_samples': NUM_SAMPLES,
                'num_epochs': NUM_EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'hidden_dim': HIDDEN_DIM
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"📈 History saved: {history_path}")
    
    # Next steps
    print(f"\n🚀 Next Steps:")
    print(f"1. 🎉 Celebrate - you have a working Math OCR model!")
    print(f"2. 📊 Check outputs/models/week1_best_model.pth")
    print(f"3. 📈 Review outputs/results/week1_training_history.json") 
    print(f"4. 🔄 Ready for Week 2: Scale to full dataset!")


def test_setup():
    """Quick test to make sure everything is ready"""
    print("🧪 Testing setup...")
    
    # Check data path
    data_path = Path("data/raw/crohme")
    if not data_path.exists():
        print(f"❌ Dataset not found at {data_path}")
        return False
    
    # Test vocabulary
    vocab = MathVocabulary()
    print(f"✅ Vocabulary: {len(vocab)} tokens")
    
    # Test device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    
    # Count dataset files
    inkml_files = list(data_path.rglob("*.inkml"))
    print(f"✅ Found {len(inkml_files)} InkML files")
    
    if len(inkml_files) == 0:
        print("❌ No InkML files found!")
        return False
    
    print("🎉 Setup test passed! Ready to train.")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Week 1 Math OCR Training")
    parser.add_argument("--test", action="store_true", help="Test setup only")
    
    args = parser.parse_args()
    
    if args.test:
        success = test_setup()
    else:
        success = main()
        
    exit(0 if success else 1)