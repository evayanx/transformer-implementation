import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Transformer
from data_loader import TextDataset, Vocabulary, create_mask, create_causal_mask
from utils import save_checkpoint, load_checkpoint, plot_training_curves

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Training components
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.lr_step_size, 
            gamma=config.lr_gamma
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            src = batch[:, :-1]  # Source sequence
            tgt = batch[:, 1:]   # Target sequence (shifted right)
            
            # Create masks
            src_mask = create_mask(src, self.config.pad_idx)
            tgt_mask = create_causal_mask(tgt.size(1))
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output, _ = self.model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
            
            # Calculate loss
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt[:, 1:].contiguous().view(-1)
            )
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                src = batch[:, :-1]
                tgt = batch[:, 1:]
                
                src_mask = create_mask(src, self.config.pad_idx)
                tgt_mask = create_causal_mask(tgt.size(1))
                
                output, _ = self.model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])
                
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self):
        print("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{self.config.num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Time: {epoch_time:.2f}s')
            print(f'  LR: {self.learning_rates[-1]:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, self.config.checkpoint_dir, 'best_model.pth')
                
            # Save regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, self.config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        # Plot training curves
        plot_training_curves(self.train_losses, self.val_losses, self.learning_rates, 
                           self.config.result_dir)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ablation_study(config):
    """Run ablation studies by removing different components"""
    ablation_results = {}
    
    # Original model
    print("Training original model...")
    original_model = Transformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff
    )
    print(f"Original model parameters: {count_parameters(original_model):,}")
    
    # Models with different components removed
    ablations = {
        'no_residual': 'Remove residual connections',
        'no_layernorm': 'Remove layer normalization',
        'no_positional': 'Remove positional encoding',
        'single_head': 'Single head attention',
    }
    
    # Implement and train each ablation...
    # This would involve modifying the model architecture accordingly
    
    return ablation_results