import numpy as np

def adam_step(params, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Convert all inputs to numpy arrays
    params = np.array(params, dtype=float)
    grad = np.array(grad, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)

    # Handle zero gradient case
    if np.all(grad == 0):
        return params, m, v

    # Update biased first and second moment estimates
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad ** 2

    # Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # Update parameters
    param_new = params - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new