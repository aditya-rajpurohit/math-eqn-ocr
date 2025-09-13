"""
CROHME Dataset Loader
Handles loading and preprocessing of CROHME InkML files
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import json
from .vocabulary import MathVocabulary


class CROHMEDataset(Dataset):
    """Dataset class for CROHME handwritten math expressions"""
    
    def __init__(self, data_path, vocab, phase='train', img_size=(224, 224), max_seq_len=512):
        self.data_path = Path(data_path)
        self.vocab = vocab
        self.phase = phase
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        
        # Load samples
        print(f"📊 Loading CROHME {phase} dataset...")
        self.samples = self._load_samples()
        print(f"✅ Loaded {len(self.samples)} samples")
        
        # Setup image transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup image augmentation transforms"""
        if self.phase == 'train':
            self.transform = A.Compose([
                A.Resize(*self.img_size),
                # Handwriting-specific augmentations
                A.Rotate(limit=5, p=0.3),  # Small rotations only
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def _load_samples(self):
        """Load all InkML samples from the dataset"""
        samples = []
        
        # Find all .inkml files
        inkml_files = list(self.data_path.rglob("*.inkml"))
        print(f"Found {len(inkml_files)} InkML files")
        
        # Process files with progress bar
        for inkml_file in tqdm(inkml_files, desc="Processing InkML files"):
            try:
                sample = self._parse_inkml(inkml_file)
                if sample and sample['strokes'] and sample['latex']:
                    samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed to parse {inkml_file}: {e}")
                continue
        
        return samples
    
    def _parse_inkml(self, inkml_path):
        """Parse a single InkML file"""
        try:
            # Parse XML
            tree = ET.parse(inkml_path)
            root = tree.getroot()
            
            # Extract strokes
            strokes = self._extract_strokes(root)
            if not strokes:
                return None
            
            # Extract LaTeX ground truth
            latex = self._extract_latex_truth(root)
            if not latex:
                return None
            
            return {
                'path': str(inkml_path),
                'strokes': strokes,
                'latex': latex.strip()
            }
            
        except Exception as e:
            print(f"Error parsing {inkml_path}: {e}")
            return None
    
    def _extract_strokes(self, root):
        """Extract stroke data from InkML"""
        strokes = []
        
        # InkML namespace
        ns = {'inkml': 'http://www.w3.org/2003/InkML'}
        
        # Find all trace elements
        for trace in root.findall('.//inkml:trace', ns):
            if trace.text:
                points = []
                # Parse coordinate pairs
                coords = trace.text.strip().replace(',', ' ').split()
                
                # Group coordinates into (x, y) pairs
                for i in range(0, len(coords) - 1, 2):
                    try:
                        x = float(coords[i])
                        y = float(coords[i + 1])
                        points.append([x, y])
                    except (ValueError, IndexError):
                        continue
                
                if len(points) >= 2:  # Need at least 2 points for a stroke
                    strokes.append(points)
        
        return strokes
    
    def _extract_latex_truth(self, root):
        """Extract LaTeX ground truth from InkML"""
        ns = {'inkml': 'http://www.w3.org/2003/InkML'}
        
        # Look for annotation with type="truth"
        for annotation in root.findall('.//inkml:annotation', ns):
            if annotation.get('type') == 'truth':
                return annotation.text
        
        # Fallback: look for any annotation
        for annotation in root.findall('.//inkml:annotation', ns):
            if annotation.text and len(annotation.text.strip()) > 0:
                return annotation.text
        
        return None
    
    def _strokes_to_image(self, strokes):
        """Convert stroke data to PIL image"""
        if not strokes:
            # Return white image
            return Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
        
        # Find bounding box
        all_points = []
        for stroke in strokes:
            all_points.extend(stroke)
        
        if not all_points:
            return Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
        
        all_points = np.array(all_points)
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        
        # Add padding
        padding = 20
        width = int(max_x - min_x + 2 * padding)
        height = int(max_y - min_y + 2 * padding)
        
        # Ensure minimum size
        width = max(width, 100)
        height = max(height, 100)
        
        # Create white canvas
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw strokes
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            
            # Convert to image coordinates
            points = np.array(stroke)
            points[:, 0] = (points[:, 0] - min_x + padding).astype(int)
            points[:, 1] = (points[:, 1] - min_y + padding).astype(int)
            
            # Clip to image bounds
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            
            # Draw stroke
            for i in range(len(points) - 1):
                cv2.line(img, tuple(points[i]), tuple(points[i + 1]), (0, 0, 0), thickness=2)
        
        return Image.fromarray(img)
    
    def _prepare_sequence(self, latex):
        """Prepare LaTeX sequence for training"""
        # Tokenize LaTeX
        tokens = self.vocab.simple_tokenize(latex)
        
        # Add SOS/EOS tokens
        tokens = [self.vocab.SOS_TOKEN] + tokens + [self.vocab.EOS_TOKEN]
        
        # Convert to IDs
        token_ids = self.vocab.encode(tokens)
        
        # Truncate if too long
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len-1] + [self.vocab.eos_id]
        
        return token_ids
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert strokes to image
        image = self._strokes_to_image(sample['strokes'])
        
        # Apply transforms
        transformed = self.transform(image=np.array(image))
        image_tensor = transformed['image']
        
        # Prepare sequence
        token_ids = self._prepare_sequence(sample['latex'])
        
        return {
            'image': image_tensor,
            'tokens': torch.tensor(token_ids, dtype=torch.long),
            'latex': sample['latex'],
            'path': sample['path']
        }


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    
    # Pad sequences to same length
    max_len = max(len(item['tokens']) for item in batch)
    
    padded_tokens = []
    for item in batch:
        tokens = item['tokens']
        # Pad with PAD token (ID 0)
        pad_len = max_len - len(tokens)
        if pad_len > 0:
            padding = torch.zeros(pad_len, dtype=torch.long)
            padded_tokens.append(torch.cat([tokens, padding]))
        else:
            padded_tokens.append(tokens)
    
    tokens = torch.stack(padded_tokens)
    
    return {
        'images': images,
        'tokens': tokens,
        'latex': [item['latex'] for item in batch],
        'paths': [item['path'] for item in batch]
    }


def get_dataloaders(cfg):
    """Create train/val/test dataloaders"""
    # Initialize vocabulary
    vocab = MathVocabulary()
    
    # Dataset path
    data_path = Path(cfg.data.datasets.crohme.path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"CROHME dataset not found at {data_path}")
    
    # Create full dataset
    full_dataset = CROHMEDataset(
        data_path=data_path,
        vocab=vocab,
        phase='train',  # Will split later
        img_size=(cfg.model.get('img_size', [224, 224])),
        max_seq_len=cfg.model.max_seq_length
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(cfg.data.datasets.crohme.train_split * total_size)
    val_size = int(cfg.data.datasets.crohme.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"📊 Dataset splits: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Random split
    generator = torch.Generator().manual_seed(cfg.experiment.seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Set phases for transforms
    for i, dataset in enumerate([train_dataset, val_dataset, test_dataset]):
        phases = ['train', 'val', 'test']
        # Create new transform for each split
        dataset.dataset.phase = phases[i]
        dataset.dataset._setup_transforms()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.get('num_workers', 2),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.get('num_workers', 2),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.get('num_workers', 2),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, vocab


def test_dataloader():
    """Test the CROHME dataloader"""
    print("🧪 Testing CROHME DataLoader...")
    
    # Simple config for testing
    class SimpleConfig:
        def __init__(self):
            self.data = type('', (), {})()
            self.data.datasets = type('', (), {})()
            self.data.datasets.crohme = type('', (), {})()
            self.data.datasets.crohme.path = "data/raw/crohme"
            self.data.datasets.crohme.train_split = 0.8
            self.data.datasets.crohme.val_split = 0.1
            self.data.datasets.crohme.test_split = 0.1
            
            self.training = type('', (), {})()
            self.training.batch_size = 4
            
            self.model = type('', (), {})()
            self.model.max_seq_length = 512
            
            self.experiment = type('', (), {})()
            self.experiment.seed = 42
    
    cfg = SimpleConfig()
    
    try:
        train_loader, val_loader, test_loader, vocab = get_dataloaders(cfg)
        
        print(f"✅ DataLoaders created successfully!")
        print(f"📊 Train: {len(train_loader.dataset)} samples")
        print(f"📊 Val: {len(val_loader.dataset)} samples") 
        print(f"📊 Test: {len(test_loader.dataset)} samples")
        print(f"📚 Vocabulary size: {len(vocab)}")
        
        # Test first batch
        print("\n🔍 Testing first batch...")
        batch = next(iter(train_loader))
        print(f"Images shape: {batch['images'].shape}")
        print(f"Tokens shape: {batch['tokens'].shape}")
        print(f"Sample LaTeX: {batch['latex'][0]}")
        
        print("✅ DataLoader test complete!")
        return True
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False


if __name__ == "__main__":
    test_dataloader()