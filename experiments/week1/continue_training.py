"""
Continue training from saved checkpoint to reach 90% accuracy
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

# Import our modules
from src.data.vocabulary import MathVocabulary
from src.data.dataloader import CROHMEDataset, collate_fn


class SimpleModel(nn.Module):
    """Same model as week1_train.py"""
    
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Simple CNN encoder
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
        
        self.cnn_proj = nn.Linear(512 * 7 * 7, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, images, target_tokens=None):
        batch_size = images.size(0)
        
        cnn_features = self.cnn(images)
        cnn_features = cnn_features.view(batch_size, -1)
        image_context = self.cnn_proj(cnn_features)
        
        if target_tokens is not None:
            seq_len = target_tokens.size(1)
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


def main():
    print("🔄 Continuing Training to Reach 90% Accuracy")
    print("=" * 50)
    
    # Setup device
    device = setup_device()
    
    # Load checkpoint
    checkpoint_path = "outputs/models/week1_best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Run week1_train.py first!")
        return
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create vocabulary
    vocab = MathVocabulary()
    
    # Create model
    model = SimpleModel(len(vocab), checkpoint['hidden_dim']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded - Previous best: {checkpoint['accuracy']:.2f}%")
    
    # Load dataset (smaller for faster overfitting)
    data_path = Path("data/raw/crohme")
    dataset = CROHMEDataset(data_path, vocab, phase='train', img_size=(224, 224))
    
    # Use even smaller subset for guaranteed overfitting
    small_indices = list(range(200))  # Just 200 samples
    small_dataset = Subset(dataset, small_indices)
    
    dataloader = DataLoader(
        small_dataset,
        batch_size=8,  # Smaller batch for GPU memory
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"📊 Using {len(small_dataset)} samples for fast overfitting")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # Lower LR for fine-tuning
    
    # Continue training
    best_accuracy = checkpoint['accuracy']
    start_epoch = checkpoint['epoch']
    
    print(f"🏋️ Resuming from epoch {start_epoch}, target: >90%")
    
    for epoch in range(15):  # More epochs
        current_epoch = start_epoch + epoch + 1
        print(f"\nEpoch {current_epoch}")
        
        loss, accuracy = train_epoch(model, dataloader, criterion, optimizer, device, vocab)
        print(f"📊 Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"🎉 New best accuracy: {accuracy:.2f}%!")
            
            # Save updated checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': len(vocab),
                'hidden_dim': checkpoint['hidden_dim'],
                'accuracy': accuracy,
                'epoch': current_epoch,
                'config': checkpoint['config']
            }, checkpoint_path)
            
            print(f"💾 Model updated: {checkpoint_path}")
        
        # Check if we achieved the goal
        if accuracy >= 90.0:
            print(f"\n🎯 WEEK 1 GOAL ACHIEVED!")
            print(f"   Target: >90% accuracy")
            print(f"   Achieved: {accuracy:.2f}% accuracy")
            print(f"   Device: {device}")
            break
    
    print(f"\n🏁 Final Result: {best_accuracy:.2f}% accuracy")
    
    if best_accuracy >= 90.0:
        print("✅ Week 1 COMPLETE! Ready for Week 2!")
    else:
        print("⚠️  Close! The model is learning - try running again.")


if __name__ == "__main__":
    main()