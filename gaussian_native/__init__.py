"""Gaussian-native diffeomorphic registration.

V12 adds bidirectional partial Gaussian transport. V13 adds
Small-Organ-Adaptive Gaussian Refinement (SAGR): fixed-budget child Gaussian
densification, local correspondence, and residual diffeomorphic composition.
"""

from .model import GaussianNativeRegistration
from .losses import GaussianNativeObjective
from .small_organ_refinement import SmallOrganAdaptiveGaussianRefiner

__all__ = [
    "GaussianNativeObjective",
    "GaussianNativeRegistration",
    "SmallOrganAdaptiveGaussianRefiner",
]
