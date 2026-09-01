from .abmil import ABMIL
from .meanmil import MeanMIL
from .transmil import TransMIL

MIL_MODELS = ('abmil', 'meanmil', 'transmil')


def build_mil(model_type, input_dim, num_classes):
    if model_type == 'abmil':
        return ABMIL(input_dim=input_dim, num_classes=num_classes)
    if model_type == 'meanmil':
        return MeanMIL(input_dim=input_dim, num_classes=num_classes)
    if model_type == 'transmil':
        return TransMIL(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model_type '{model_type}'. Expected one of {MIL_MODELS}.")
