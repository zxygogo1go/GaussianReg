"""Gaussian-native diffeomorphic registration.

Revisions through v9 keep learned correspondence and deformation on Gaussian
primitives. V10 adds an explicit image-pyramid residual SVF refiner after the
Gaussian coarse prior. V11 adds gradient-aware full-resolution refinement for
small longitudinal anatomy.
"""

from .model import GaussianNativeRegistration
from .losses import GaussianNativeObjective

__all__ = ["GaussianNativeRegistration", "GaussianNativeObjective"]
