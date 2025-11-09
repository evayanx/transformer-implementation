import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def save_checkpoint(state, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, os.path.join(directory, filename))

def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

def plot_training_curves(train_losses, val_losses, learning_rates, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Learning rate
    ax2.plot(epochs, learning_rates, 'g-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_attention_patterns(attention_weights, tokens, layer_idx=0, head_idx=0):
    """Visualize attention patterns for analysis"""
    if len(attention_weights) > layer_idx:
        attn = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(attn, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # Add token labels if available
        if tokens is not None and len(tokens) == attn.shape[0]:
            plt.xticks(range(len(tokens)), tokens, rotation=45)
            plt.yticks(range(len(tokens)), tokens)
        
        plt.tight_layout()
        return plt.gcf()