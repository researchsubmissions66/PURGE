"""
Vendored from EleutherAI/concept-erasure (MIT).

The LLM activation-scrubbing subpackage was removed: it targeted Llama/GPT-NeoX
forward passes, required transformers + datasets, and used absolute
`from concept_erasure import ...` imports that cannot resolve from this vendored
location. Only the estimator classes used by scripts/fit_unlearner.py remain.
"""
from __future__ import annotations
from .groupby import GroupedTensor, groupby
from .leace import ErasureMethod, LeaceEraser, LeaceFitter
from .oracle import OracleEraser, OracleFitter
from .quadratic import QuadraticEditor, QuadraticEraser, QuadraticFitter
from .quantile import QuantileNormalizer, cdf, icdf
from .shrinkage import optimal_linear_shrinkage
from .utils import assert_type

__all__ = [
    "assert_type",
    "cdf",
    "groupby",
    "icdf",
    "optimal_linear_shrinkage",
    "ErasureMethod",
    "GroupedTensor",
    "LeaceEraser",
    "LeaceFitter",
    "OracleEraser",
    "OracleFitter",
    "QuadraticEditor",
    "QuadraticEraser",
    "QuadraticFitter",
    "QuantileNormalizer",
]
