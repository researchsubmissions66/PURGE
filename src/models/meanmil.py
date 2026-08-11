import torch
import torch.nn as nn

class MeanMIL(nn.Module):
    """
    Mean Pooling MIL. The simplest possible baseline.
    Averages all patch embeddings and passes through a linear classifier.
    """
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [N_patches, D]
        # Global average pool over patches -> [1, D]
        x_mean = x.mean(dim=0, keepdim=True)
        logits = self.classifier(x_mean).squeeze(0) # -> [num_classes]
        
        # Return dummy attention weights for compatibility with train script
        dummy_A = torch.ones(1, x.shape[0], device=x.device) / x.shape[0]
        return logits, dummy_A
