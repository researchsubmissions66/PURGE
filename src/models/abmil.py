import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    """
    Attention-Based MIL matching the official AMLab-Amsterdam/AttentionDeepMIL GatedAttention architecture.
    Adapted for pre-extracted patch features (no CNN feature extractor).
    """
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        attention_dim=128,
        num_classes=2,
    ):
        super().__init__()

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(attention_dim, 1)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [N_patches, D]
        H = self.feature_proj(x) # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # Kx1
        A = torch.transpose(A, 1, 0)  # 1xK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # 1xM
        
        # Z is 1xM. We squeeze it to pass to classifier
        logits = self.classifier(Z.squeeze(0))

        return logits, A
