"""Gaussian-native diffeomorphic registration.

The package deliberately keeps every learned representation and deformation
degree of freedom on Gaussian primitives. Dense tensors are used only to
measure the input volumes and to rasterize/integrate the final velocity.
"""

from .model import GaussianNativeRegistration
from .losses import GaussianNativeObjective

__all__ = ["GaussianNativeRegistration", "GaussianNativeObjective"]
