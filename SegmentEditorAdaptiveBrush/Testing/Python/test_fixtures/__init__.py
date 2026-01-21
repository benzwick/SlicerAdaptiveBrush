"""Test fixtures and synthetic data generators for SlicerAdaptiveBrush tests."""

from .synthetic_image import (
    create_bimodal_image,
    create_checkerboard_image,
    create_concentric_spheres,
    create_gradient_image,
    create_noisy_sphere,
    create_uniform_image,
)

__all__ = [
    "create_uniform_image",
    "create_bimodal_image",
    "create_gradient_image",
    "create_noisy_sphere",
    "create_concentric_spheres",
    "create_checkerboard_image",
]
