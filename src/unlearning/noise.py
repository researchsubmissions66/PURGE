import torch
import torch.nn.functional as F

def apply_gaussian_noise(X, sigma=1.0):
    """
    Applies isotropic Gaussian noise to the features.
    X: Tensor of shape [N, D]
    sigma: Float, standard deviation of the noise
    """
    noise = torch.randn_like(X) * sigma
    return X + noise

def apply_dropout(X, p=0.5):
    """
    Randomly zeroes out feature dimensions with probability p.
    X: Tensor of shape [N, D]
    p: Float, probability of dropping an element
    """
    # Use PyTorch's native dropout for efficiency.
    # Note: F.dropout scales the remaining elements by 1/(1-p) during training
    # to maintain the expected sum. We explicitly want the corruption effect.
    return F.dropout(X, p=p, training=True)
