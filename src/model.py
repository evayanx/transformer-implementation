import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    Mathematical formulation:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    MultiHead = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] or [batch_size, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape back to [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(output)
        
        return output, attention_weights

class PositionWiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Using GELU as in modern Transformers
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with:
    - Multi-head self-attention
    - Residual connection + LayerNorm
    - Position-wise FFN
    - Another residual connection + LayerNorm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer with:
    - Masked multi-head self-attention
    - Multi-head cross-attention
    - Position-wise FFN
    - Residual connections + LayerNorm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFFN(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        self_attn_output, self_attention_weights = self.self_attention(x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with encoder output
        cross_attn_output, cross_attention_weights = self.cross_attention(
            x, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attention_weights, cross_attention_weights

class Transformer(nn.Module):
    """
    Complete Transformer model with encoder and decoder
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, 
                 d_ff=2048, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Decoder  
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.d_model**-0.5)
    
    def encode(self, src, src_mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        encoder_output = src_emb
        attention_weights = []
        for layer in self.encoder_layers:
            encoder_output, attn_weights = layer(encoder_output, src_mask)
            attention_weights.append(attn_weights)
            
        return encoder_output, attention_weights
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        decoder_output = tgt_emb
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.decoder_layers:
            decoder_output, self_attn, cross_attn = layer(
                decoder_output, encoder_output, src_mask, tgt_mask
            )
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
            
        return decoder_output, self_attention_weights, cross_attention_weights
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output, enc_attention_weights = self.encode(src, src_mask)
        decoder_output, dec_self_attention_weights, dec_cross_attention_weights = self.decode(
            tgt, encoder_output, src_mask, tgt_mask
        )
        output = self.output_proj(decoder_output)
        return output, {
            'encoder_attention': enc_attention_weights,
            'decoder_self_attention': dec_self_attention_weights,
            'decoder_cross_attention': dec_cross_attention_weights
        }