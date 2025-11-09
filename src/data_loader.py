import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter
import re

class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_length=128):
        self.texts = texts
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Convert text to token indices
        tokens = self.text_to_tokens(text)
        # Add SOS and EOS tokens
        tokens = [self.vocab['<sos>']] + tokens + [self.vocab['<eos>']]
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab['<pad>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
            
        return torch.tensor(tokens, dtype=torch.long)

    def text_to_tokens(self, text):
        # Simple tokenization - split by space and convert to lowercase
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts, min_freq=2):
        counter = Counter()
        for text in texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            counter.update(tokens)
            
        # Add special tokens
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        for token in special_tokens:
            self.word2idx[token] = len(self.word2idx)
            
        # Add words meeting frequency threshold
        for word, freq in counter.items():
            if freq >= min_freq:
                self.word2idx[word] = len(self.word2idx)
                
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
    def save_vocab(self, path):
        import json
        with open(path, 'w') as f:
            json.dump(self.word2idx, f)
            
    def load_vocab(self, path):
        import json
        with open(path, 'r') as f:
            self.word2idx = json.load(f)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

def create_mask(seq, pad_idx):
    """Create padding mask"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_causal_mask(seq_len):
    """Create causal mask for decoder"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(1)