"""
Minimal LoRA for injecting an erasure objective INSIDE an encoder (plan section 19).

`peft` is not installed in this environment, and the need here is narrow: wrap
selected nn.Linear layers so only the adapters train. ~60 lines beats a dependency.

    W x  ->  W x + (alpha / r) * B (A x),   A: [r, in], B: [out, r], B init 0

B starts at zero, so the wrapped model is EXACTLY the original at step 0. That
matters: it makes "did poisoning change anything" a well-posed question, and any
drift is attributable to training rather than to the wrapping.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r
        # Match the base layer, or the adapters land on CPU while the model is on
        # GPU and the first forward dies.
        dev = base.weight.device
        dt = base.weight.dtype
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        delta = self.dropout(x) @ self.A.T @ self.B.T
        return out + self.scaling * delta


def inject_lora(module, target_names=('q_proj', 'v_proj'), last_n_blocks=None,
                block_container=None, r=8, alpha=16.0, verbose=True):
    """
    Replace matching nn.Linear children with LoRALinear, in place.

    target_names   : substrings of attribute names to wrap
    last_n_blocks  : if given with `block_container` (an nn.ModuleList of
                     transformer blocks), restrict to its last N entries -
                     poisoning the final blocks is the cheap, high-leverage
                     placement (plan section 19)
    Returns the list of wrapped parameter names.
    """
    scope = module
    if last_n_blocks is not None and block_container is not None:
        blocks = list(block_container)[-last_n_blocks:]
        scope = nn.ModuleList(blocks)

    wrapped = []

    def _recurse(mod, prefix=''):
        for name, child in list(mod.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and any(t in name for t in target_names):
                setattr(mod, name, LoRALinear(child, r=r, alpha=alpha))
                wrapped.append(full)
            else:
                _recurse(child, full)

    _recurse(scope)
    if verbose:
        print(f"  injected LoRA (r={r}) into {len(wrapped)} layers")
    return wrapped


def lora_parameters(module):
    return [p for n, p in module.named_parameters()
            if p.requires_grad and ('.A' in n or '.B' in n)]


def freeze_base(module):
    """Freeze everything except LoRA adapters."""
    for n, p in module.named_parameters():
        p.requires_grad_(n.endswith('.A') or n.endswith('.B'))
    n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in module.parameters())
    return n_train, n_total
