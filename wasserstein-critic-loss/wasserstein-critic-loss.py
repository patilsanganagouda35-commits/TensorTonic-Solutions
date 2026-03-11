import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    real_scores = np.array(real_scores, dtype=np.float64)
    fake_scores = np.array(fake_scores, dtype=np.float64)
    
    return float(np.mean(fake_scores) - np.mean(real_scores))