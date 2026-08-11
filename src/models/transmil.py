import torch
import torch.nn as nn

class TransMIL(nn.Module):
    """
    Transformer-based MIL (Simplified version without positional encoding,
    making it permutation invariant like DeepSets/AttentionMIL).
    Applies self-attention across patches before classification.
    """
    def __init__(self, input_dim, num_classes=2, hidden_dim=256, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim*4, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [N_patches, D]
        x = self.proj(x) # [N, hidden_dim]
        
        # Add CLS token
        cls_token = self.cls_token.expand(1, -1, -1) # [1, 1, hidden_dim]
        x = x.unsqueeze(0) # [1, N, hidden_dim]
        x = torch.cat((cls_token, x), dim=1) # [1, N+1, hidden_dim]
        
        # Transformer forward pass
        out = self.transformer(x) # [1, N+1, hidden_dim]
        
        # Classification from CLS token
        cls_out = out[0, 0, :] # [hidden_dim]
        logits = self.classifier(cls_out)
        
        # Return dummy attention for compatibility
        dummy_A = torch.ones(1, x.shape[1]-1, device=x.device) / (x.shape[1]-1)
        return logits, dummy_A
