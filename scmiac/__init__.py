"""scMIAC Python package."""

from .modeling.scmiac import find_anchors, preprocess, train_model, model_inference
from .utils import set_seed

__all__ = (
    "find_anchors",
    "preprocess",
    "train_model",
    "model_inference",
    "set_seed",
)

__version__ = "0.1.0"