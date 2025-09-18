"""
Training pipeline for Math OCR model
Handles training loop, validation, checkpointing, and metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple, Optional

#from ..models.math_ocr_model import MathOCRModel


class MathOCRTrainer:
    """Complete training pipeline for Math OCR"""
    
    def __init__(self, model, train_loader, val_loader, vocab, cfg, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.cfg = cfg
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_accuracy = 0.0
        self.patience_counter = 0
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab.pad_id,  # Ignore padding tokens
            label_smoothing=0.1  # Label smoothing for better generalization
        )
        
        # Paths for saving
        self.save_dir = Path(cfg.paths.model_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        print(f"🏋️ Trainer initialized on {device}")
        print(f"   Optimizer: {self.optimizer.__class__.__name__}")
        print(f"   Learning rate: {cfg.training.learning_rate}")
        print(f"   Batch size: {cfg.training.batch_size}")
    
    def _setup_optimizer(self):
        """Setup optimizer based on config"""
        if self.cfg.optimizer.name.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.cfg.training.learning_rate,
                weight_decay=self.cfg.training.weight_decay,
                betas=self.cfg.optimizer.betas
            )
        elif self.cfg.optimizer.name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.training.learning_rate,
                weight_decay=self.cfg.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer.name}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.cfg.scheduler.name.lower() == 'step_lr':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.scheduler.step_size,
                gamma=self.cfg.scheduler.gamma
            )
        elif self.cfg.scheduler.name.lower() == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.num_epochs
            )
        else:
            self.scheduler = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_tokens = 0
        total_tokens = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            
            batch_size, seq_len = tokens.shape
            
            # Prepare inputs and targets for teacher forcing
            input_tokens = tokens[:, :-1]  # All except last token
            target_tokens = tokens[:, 1:]  # All except first token
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(images, input_tokens)  # [B, seq_len-1, vocab_size]
            
            # Calculate loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),  # [B*(seq_len-1), vocab_size]
                target_tokens.reshape(-1)  # [B*(seq_len-1)]
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.cfg.training.grad_clip
                )
            
            # Update weights
            self.optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)  # [B, seq_len-1]
            
            # Mask out padding tokens for accuracy calculation
            mask = (target_tokens != self.vocab.pad_id)
            correct_tokens += ((predictions == target_tokens) & mask).sum().item()
            total_tokens += mask.sum().item()
            
            # Update metrics
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct_tokens/total_tokens:.1f}%',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log to wandb
            if self.cfg.logging.use_wandb and batch_idx % self.cfg.logging.log_every == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_accuracy': 100 * correct_tokens / total_tokens,
                    'train/learning_rate': current_lr,
                    'epoch': self.current_epoch,
                    'step': self.current_epoch * len(self.train_loader) + batch_idx
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = 100 * correct_tokens / total_tokens
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_tokens = 0
        total_tokens = 0
        sequence_accuracies = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                images = batch['images'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                
                batch_size, seq_len = tokens.shape
                
                # Prepare inputs and targets
                input_tokens = tokens[:, :-1]
                target_tokens = tokens[:, 1:]
                
                # Forward pass
                logits = self.model(images, input_tokens)
                
                # Calculate loss
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1)
                )
                
                # Calculate token accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = (target_tokens != self.vocab.pad_id)
                correct_tokens += ((predictions == target_tokens) & mask).sum().item()
                total_tokens += mask.sum().item()
                
                # Calculate sequence accuracy (exact match)
                for i in range(batch_size):
                    pred_seq = predictions[i][mask[i]]
                    target_seq = target_tokens[i][mask[i]]
                    sequence_accuracies.append((pred_seq == target_seq).all().item())
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*correct_tokens/total_tokens:.1f}%'
                })
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        token_accuracy = 100 * correct_tokens / total_tokens
        sequence_accuracy = 100 * np.mean(sequence_accuracies)
        
        return {
            'loss': avg_loss,
            'token_accuracy': token_accuracy,
            'sequence_accuracy': sequence_accuracy
        }
    
    def save_checkpoint(self, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_accuracy': self.best_accuracy,
            'metrics': metrics,
            'vocab_size': self.vocab.get_vocab_size(),
            'config': dict(self.cfg)
        }
        
        # Save latest checkpoint
        checkpoint_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        
        print(f"📂 Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self) -> str:
        """Complete training loop"""
        print(f"🏋️ Starting training for {self.cfg.training.num_epochs} epochs...")
        
        start_time = time.time()
        
        for epoch in range(self.cfg.training.num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_metrics = self.train_epoch()
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_metrics.append(val_metrics)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Print epoch results
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Token Acc: {val_metrics['token_accuracy']:.2f}%, Seq Acc: {val_metrics['sequence_accuracy']:.2f}%")
            
            # Log to wandb
            if self.cfg.logging.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'val/loss': val_metrics['loss'],
                    'val/token_accuracy': val_metrics['token_accuracy'],
                    'val/sequence_accuracy': val_metrics['sequence_accuracy'],
                    'val/learning_rate': train_metrics['learning_rate']
                })
            
            # Check for best model
            is_best = val_metrics['sequence_accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['sequence_accuracy']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.cfg.logging.save_every == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.cfg.training.early_stopping_patience:
                print(f"🛑 Early stopping triggered after {self.patience_counter} epochs without improvement")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Best sequence accuracy: {self.best_accuracy:.2f}%")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        
        return str(self.save_dir / 'best_model.pth')
    
    def evaluate(self, test_loader, model_path=None):
        """Evaluate model on test set"""
        if model_path:
            self.load_checkpoint(model_path)
        
        self.model.eval()
        print("🧪 Evaluating on test set...")
        
        all_predictions = []
        all_targets = []
        all_latex_predictions = []
        all_latex_targets = []
        
        total_loss = 0.0
        total_samples = 0
        correct_tokens = 0
        total_tokens = 0
        sequence_accuracies = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Testing")
            
            for batch in pbar:
                images = batch['images'].to(self.device)
                tokens = batch['tokens'].to(self.device)
                latex_strings = batch['latex']
                
                batch_size = tokens.shape[0]
                
                # Generate predictions
                generated_tokens = self.model.generate(images, max_length=self.cfg.model.max_seq_length)
                
                # Calculate metrics on ground truth (for loss)
                input_tokens = tokens[:, :-1]
                target_tokens = tokens[:, 1:]
                logits = self.model(images, input_tokens)
                
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1)
                )
                
                # Token accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = (target_tokens != self.vocab.pad_id)
                correct_tokens += ((predictions == target_tokens) & mask).sum().item()
                total_tokens += mask.sum().item()
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Convert to LaTeX strings for evaluation
                for i in range(batch_size):
                    # Generated sequence
                    gen_tokens = generated_tokens[i].cpu().tolist()
                    gen_latex = self.vocab.decode_to_string(gen_tokens, remove_special=True)
                    all_latex_predictions.append(gen_latex)
                    
                    # Target sequence
                    target_latex = latex_strings[i]
                    all_latex_targets.append(target_latex)
                    
                    # Sequence accuracy (exact match after decoding)
                    sequence_accuracies.append(gen_latex.strip() == target_latex.strip())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*correct_tokens/total_tokens:.1f}%'
                })
        
        # Calculate final metrics
        test_results = {
            'test_loss': total_loss / total_samples,
            'token_accuracy': 100 * correct_tokens / total_tokens,
            'sequence_accuracy': 100 * np.mean(sequence_accuracies),
            'total_samples': total_samples,
            'predictions': all_latex_predictions[:10],  # Save first 10 for inspection
            'targets': all_latex_targets[:10]
        }
        
        print(f"\n📊 Test Results:")
        print(f"   Loss: {test_results['test_loss']:.4f}")
        print(f"   Token Accuracy: {test_results['token_accuracy']:.2f}%")
        print(f"   Sequence Accuracy: {test_results['sequence_accuracy']:.2f}%")
        print(f"   Total Samples: {test_results['total_samples']}")
        
        # Show some examples
        print(f"\n🔍 Sample Predictions:")
        for i in range(min(3, len(all_latex_predictions))):
            print(f"   Target: {all_latex_targets[i]}")
            print(f"   Pred  : {all_latex_predictions[i]}")
            print(f"   Match : {'✅' if all_latex_predictions[i].strip() == all_latex_targets[i].strip() else '❌'}")
            print()
        
        return test_results


def create_trainer(model, train_loader, val_loader, vocab, cfg, device):
    """Factory function to create trainer"""
    return MathOCRTrainer(model, train_loader, val_loader, vocab, cfg, device)


def test_trainer():
    """Test the trainer with dummy data"""
    print("🧪 Testing Math OCR Trainer...")
    
    # This would normally be called from main.py with real config
    # Just a basic test to check imports and structure
    
    print("✅ Trainer module loaded successfully!")
    print("🏗️ Ready to train with real data and config!")


if __name__ == "__main__":
    test_trainer()