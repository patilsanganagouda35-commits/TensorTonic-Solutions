import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model => last column is sin.
    """
    # Create position indices (seq_len, 1)
    positions = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    
    # Create dimension indices for even positions only
    i = np.arange(0, d_model, 2)  # (d_model // 2,)
    
    # Compute the division term: base^(2i/d_model)
    div_term = np.power(base, i / d_model)  # (d_model // 2,)
    
    # Compute angles
    angles = positions / div_term  # (seq_len, d_model // 2)
    
    # Initialize PE matrix
    PE = np.zeros((seq_len, d_model), dtype=float)
    
    # Even indices -> sin, Odd indices -> cos
    PE[:, 0::2] = np.sin(angles)
    PE[:, 1::2] = np.cos(angles[:, :d_model // 2])
    
    return PE