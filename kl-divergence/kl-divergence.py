import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    
    Args:
        p: array-like, first probability distribution, shape (N,)
        q: array-like, second probability distribution, shape (N,)
        eps: float, numerical stability epsilon
    
    Returns:
        float: KL divergence between P and Q
    """
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    
    # Add epsilon for numerical stability (avoid log(0))
    p = p + eps
    q = q + eps
    
    return np.sum(p * np.log(p / q))