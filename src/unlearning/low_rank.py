import math

import torch
import torch.nn as nn


class LowRankEraser(nn.Module):
    """
    Residual low-rank adapter (plan section 5):

        z' = z + alpha * B * A * z = (I + alpha*B*A) z

    WARNING - THIS TRANSFORM CANNOT ERASE INFORMATION.

    I + alpha*B*A is invertible for generic A, B (it fails to be only when
    -1/alpha is exactly an eigenvalue of B*A). An invertible linear map is an
    information-preserving bijection, so a probe *retrained* on z' recovers the
    target concept exactly. It only fools the fixed probe it was trained against
    - precisely the trivial attack that plan section 3 rules out.

    It is retained as a documented negative-control ablation - its test pins the
    original bug so it cannot silently return. For actual erasure use the affine
    SVD projection in src/unlearning/subspace.py, which is rank-deficient by
    construction.

    Run src.unlearning.audit.assert_erases() on any eraser before a sweep.
    """

    def __init__(self, input_dim, rank=16, alpha=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.alpha = alpha

        # A: down-projection [rank, input_dim]
        self.A = nn.Linear(input_dim, rank, bias=False)
        # B: up-projection [input_dim, rank]
        self.B = nn.Linear(rank, input_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        # B starts at zero so the adapter begins as the identity (z' == z).
        nn.init.zeros_(self.B.weight)

    def bottleneck_rank(self):
        """No bottleneck: the map is full-rank, so nothing is erased."""
        return None

    def forward(self, z):
        # z: [..., input_dim]
        delta = self.B(self.A(z))
        return z + self.alpha * delta
