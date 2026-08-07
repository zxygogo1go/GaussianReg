"""Anatomy-aware plug-in local residual refinement."""

from .adapter import RegistrationWithLocalRefinement
from .functional import INPUT_MODES, build_anatomy_maps, build_feature_tensor
from .network import LocalResidualUNet, count_parameters
from .refiner import AnatomyAwareLocalRefiner
from .types import LocalRefinementConfig, PlugInRegistrationOutput, RefinementInput, RefinementOutput

__all__ = [
    "AnatomyAwareLocalRefiner",
    "INPUT_MODES",
    "LocalRefinementConfig",
    "LocalResidualUNet",
    "PlugInRegistrationOutput",
    "RefinementInput",
    "RefinementOutput",
    "RegistrationWithLocalRefinement",
    "build_anatomy_maps",
    "build_feature_tensor",
    "count_parameters",
]
