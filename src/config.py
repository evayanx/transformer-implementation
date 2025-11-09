class Config:
    # Model parameters
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    dropout = 0.1
    max_seq_len = 128
    
    # Training parameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # Learning rate scheduler
    lr_step_size = 10
    lr_gamma = 0.8
    
    # Data parameters
    vocab_size = 10000
    min_freq = 2
    pad_idx = 0
    
    # Paths
    data_dir = '../data'
    checkpoint_dir = '../checkpoints'
    result_dir = '../results'
    
    # Training settings
    save_interval = 10
    seed = 42