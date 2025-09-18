"""
Math OCR Model: CNN Encoder + Transformer Decoder
Converts handwritten math expression images to LaTeX sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import timm


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ImageEncoder(nn.Module):
    """CNN encoder for extracting features from math expression images"""
    
    def __init__(self, backbone='resnet34', pretrained=True, hidden_dim=512):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Load pretrained CNN backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='',  # Remove global pooling
        )
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            backbone_out = self.backbone(dummy)
            self.backbone_dim = backbone_out.shape[1]
            self.feature_h = backbone_out.shape[2] 
            self.feature_w = backbone_out.shape[3]
        
        # Project CNN features to transformer dimension
        self.feature_projection = nn.Linear(self.backbone_dim, hidden_dim)
        
        # 2D positional embeddings for spatial features
        max_h, max_w = 32, 32  # Maximum feature map size
        self.pos_embed_h = nn.Embedding(max_h, hidden_dim // 2)
        self.pos_embed_w = nn.Embedding(max_w, hidden_dim // 2)
        
        print(f"🏗️ ImageEncoder: {backbone} -> {self.backbone_dim} -> {hidden_dim}")
        print(f"   Feature map: {self.feature_h}x{self.feature_w}")
    
    def forward(self, images):
        """
        Args:
            images: [batch_size, 3, H, W]
        Returns:
            features: [batch_size, seq_len, hidden_dim]
        """
        batch_size = images.size(0)
        
        # Extract CNN features: [B, C, H, W]
        features = self.backbone(images)
        B, C, H, W = features.shape
        
        # Flatten spatial dimensions: [B, H*W, C]
        features = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Project to transformer dimension: [B, H*W, hidden_dim]
        features = self.feature_projection(features)
        
        # Add 2D positional embeddings
        h_pos = torch.arange(H, device=features.device).unsqueeze(1).repeat(1, W).flatten()
        w_pos = torch.arange(W, device=features.device).unsqueeze(0).repeat(H, 1).flatten()
        
        h_embed = self.pos_embed_h(h_pos)  # [H*W, hidden_dim//2]
        w_embed = self.pos_embed_w(w_pos)  # [H*W, hidden_dim//2]
        
        pos_embed = torch.cat([h_embed, w_embed], dim=-1)  # [H*W, hidden_dim]
        features = features + pos_embed.unsqueeze(0)  # Broadcast to batch
        
        return features


class TransformerDecoder(nn.Module):
    """Transformer decoder for sequence generation"""
    
    def __init__(self, vocab_size, hidden_dim=512, num_layers=6, num_heads=8, 
                 ff_dim=2048, dropout=0.1, max_seq_len=512):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        print(f"🧠 TransformerDecoder: {vocab_size} vocab, {num_layers} layers, {num_heads} heads")
    
    def forward(self, image_features, target_tokens=None, max_length=None):
        """
        Args:
            image_features: [batch_size, spatial_seq_len, hidden_dim]
            target_tokens: [batch_size, seq_len] (for training)
            max_length: int (for inference)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        if target_tokens is not None:
            # Training mode with teacher forcing
            return self._forward_train(image_features, target_tokens)
        else:
            # Inference mode with autoregressive generation
            return self._forward_inference(image_features, max_length or self.max_seq_len)
    
    def _forward_train(self, image_features, target_tokens):
        """Training forward pass with teacher forcing"""
        batch_size, seq_len = target_tokens.shape
        
        # Token embeddings + positional encoding
        token_embeds = self.token_embedding(target_tokens)  # [B, seq_len, hidden_dim]
        token_embeds = self.pos_encoding(token_embeds)
        token_embeds = self.dropout(token_embeds)
        
        # Create causal mask (prevent looking at future tokens)
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(target_tokens.device)
        
        # Transformer decoder
        output = self.transformer(
            tgt=token_embeds,
            memory=image_features,
            tgt_mask=causal_mask
        )  # [B, seq_len, hidden_dim]
        
        # Project to vocabulary
        logits = self.output_projection(output)  # [B, seq_len, vocab_size]
        
        return logits
    
    def _forward_inference(self, image_features, max_length):
        """Inference forward pass with autoregressive generation"""
        batch_size = image_features.size(0)
        device = image_features.device
        
        # Start with SOS token (assuming index 1)
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Get embeddings for current sequence
            token_embeds = self.token_embedding(generated)
            token_embeds = self.pos_encoding(token_embeds)
            token_embeds = self.dropout(token_embeds)
            
            # Create causal mask
            seq_len = generated.size(1)
            causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)
            
            # Transformer forward
            output = self.transformer(
                tgt=token_embeds,
                memory=image_features,
                tgt_mask=causal_mask
            )
            
            # Get logits for last position only
            logits = self.output_projection(output[:, -1:, :])  # [B, 1, vocab_size]
            
            # Sample next token (you can use different strategies here)
            next_token = torch.argmax(logits, dim=-1)  # [B, 1]
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have EOS token (assuming index 2)
            if torch.all(next_token == 2):
                break
        
        return generated
    
    def _generate_square_subsequent_mask(self, sz):
        """Generate causal mask for transformer"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class MathOCRModel(nn.Module):
    """Complete Math OCR model: Image -> LaTeX sequence"""
    
    def __init__(self, vocab_size, backbone='resnet34', hidden_dim=512, 
                 num_layers=6, num_heads=8, ff_dim=2048, dropout=0.1, 
                 max_seq_len=512, pretrained=True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Image encoder (CNN)
        self.encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dim=hidden_dim
        )
        
        # Sequence decoder (Transformer)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔢 Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, images, target_tokens=None, return_attention=False):
        """
        Forward pass through the complete model
        
        Args:
            images: [batch_size, 3, H, W] - Input images
            target_tokens: [batch_size, seq_len] - Target tokens (training)
            return_attention: bool - Return attention weights
        
        Returns:
            logits: [batch_size, seq_len, vocab_size] - Model predictions
            attention_weights: (optional) attention weights
        """
        
        # Encode images to features
        image_features = self.encoder(images)  # [B, spatial_len, hidden_dim]
        
        # Decode to sequences
        if target_tokens is not None:
            # Training mode: use teacher forcing
            logits = self.decoder(image_features, target_tokens)
        else:
            # Inference mode: autoregressive generation
            generated_tokens = self.decoder(image_features, max_length=self.max_seq_len)
            return generated_tokens
        
        if return_attention:
            # TODO: Extract attention weights if needed
            return logits, None
        
        return logits
    
    def generate(self, images, max_length=None, temperature=1.0, top_k=None):
        """
        Generate LaTeX sequences from images
        
        Args:
            images: [batch_size, 3, H, W]
            max_length: Maximum sequence length
            temperature: Sampling temperature (1.0 = greedy)
            top_k: Top-k sampling
        
        Returns:
            generated_tokens: [batch_size, seq_len]
        """
        self.eval()
        
        with torch.no_grad():
            # Get image features
            image_features = self.encoder(images)
            
            # Generate sequences
            batch_size = images.size(0)
            device = images.device
            max_length = max_length or self.max_seq_len
            
            # Start with SOS tokens
            generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            for step in range(max_length - 1):
                # Forward pass
                logits = self.decoder(image_features, generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    # Top-k sampling
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1)
                    next_token = torch.gather(top_k_indices, -1, next_token_idx)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences have EOS
                if torch.all(next_token == 2):  # Assuming EOS token is 2
                    break
            
            return generated


def create_model(vocab_size, config=None):
    """Factory function to create model with configuration"""
    
    if config is None:
        # Default configuration
        config = {
            'backbone': 'resnet34',
            'hidden_dim': 512,
            'num_layers': 6,
            'num_heads': 8,
            'ff_dim': 2048,
            'dropout': 0.1,
            'max_seq_len': 512,
            'pretrained': True
        }
    
    model = MathOCRModel(
        vocab_size=vocab_size,
        backbone=config.get('backbone', 'resnet34'),
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        ff_dim=config.get('ff_dim', 2048),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 512),
        pretrained=config.get('pretrained', True)
    )
    
    return model


def test_model():
    """Test the model architecture"""
    print("🧪 Testing Math OCR Model...")
    
    # Test parameters
    batch_size = 2
    vocab_size = 180
    seq_len = 20
    
    # Create model
    model = create_model(vocab_size)
    model.eval()
    
    # Test data
    images = torch.randn(batch_size, 3, 224, 224)
    target_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"📊 Input shapes:")
    print(f"   Images: {images.shape}")
    print(f"   Tokens: {target_tokens.shape}")
    
    # Test forward pass (training)
    with torch.no_grad():
        logits = model(images, target_tokens)
        print(f"✅ Training forward pass: {logits.shape}")
        
        # Test generation (inference)
        generated = model.generate(images, max_length=30)
        print(f"✅ Generation: {generated.shape}")
    
    print("🎉 Model test complete!")
    return model


if __name__ == "__main__":
    test_model()