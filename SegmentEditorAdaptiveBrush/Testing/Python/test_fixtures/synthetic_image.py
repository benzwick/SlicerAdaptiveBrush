"""Synthetic image generators for testing adaptive brush algorithms.

These generators create images with known ground truth for validating
segmentation accuracy.
"""

from typing import Tuple

import numpy as np


def create_uniform_image(
    size: Tuple[int, int, int] = (100, 100, 10),
    intensity: float = 100.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Create a uniform intensity image.

    Args:
        size: Image size as (width, height, depth).
        intensity: Uniform intensity value.
        noise_std: Standard deviation of Gaussian noise (0 for no noise).

    Returns:
        numpy array of shape (depth, height, width) with float32 dtype.
    """
    # Note: numpy uses (z, y, x) ordering
    shape = (size[2], size[1], size[0])
    image = np.full(shape, intensity, dtype=np.float32)

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, shape).astype(np.float32)
        image = image + noise

    return image


def create_bimodal_image(
    size: Tuple[int, int, int] = (100, 100, 10),
    mean1: float = 100.0,
    mean2: float = 200.0,
    std1: float = 10.0,
    std2: float = 10.0,
    split_axis: int = 0,
    split_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an image with two distinct intensity regions.

    Args:
        size: Image size as (width, height, depth).
        mean1: Mean intensity of first region.
        mean2: Mean intensity of second region.
        std1: Standard deviation of first region.
        std2: Standard deviation of second region.
        split_axis: Axis to split along (0=x, 1=y, 2=z).
        split_ratio: Ratio of first region (0.5 = half and half).

    Returns:
        Tuple of (image array, ground truth mask).
        Mask has value 1 for region1, 2 for region2.
    """
    shape = (size[2], size[1], size[0])  # (z, y, x)
    image = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)

    # Calculate split point
    axis_size = size[split_axis]
    split_point = int(axis_size * split_ratio)

    # Create slicing for each region
    if split_axis == 0:  # Split along x
        slices1 = (slice(None), slice(None), slice(0, split_point))
        slices2 = (slice(None), slice(None), slice(split_point, None))
        shape1 = (shape[0], shape[1], split_point)
        shape2 = (shape[0], shape[1], shape[2] - split_point)
    elif split_axis == 1:  # Split along y
        slices1 = (slice(None), slice(0, split_point), slice(None))
        slices2 = (slice(None), slice(split_point, None), slice(None))
        shape1 = (shape[0], split_point, shape[2])
        shape2 = (shape[0], shape[1] - split_point, shape[2])
    else:  # Split along z
        slices1 = (slice(0, split_point), slice(None), slice(None))
        slices2 = (slice(split_point, None), slice(None), slice(None))
        shape1 = (split_point, shape[1], shape[2])
        shape2 = (shape[0] - split_point, shape[1], shape[2])

    # Fill regions
    image[slices1] = np.random.normal(mean1, std1, shape1).astype(np.float32)
    image[slices2] = np.random.normal(mean2, std2, shape2).astype(np.float32)
    mask[slices1] = 1
    mask[slices2] = 2

    return image, mask


def create_gradient_image(
    size: Tuple[int, int, int] = (100, 100, 10),
    min_intensity: float = 0.0,
    max_intensity: float = 255.0,
    gradient_axis: int = 0,
) -> np.ndarray:
    """
    Create an image with linear intensity gradient.

    Args:
        size: Image size as (width, height, depth).
        min_intensity: Minimum intensity value.
        max_intensity: Maximum intensity value.
        gradient_axis: Axis along which gradient varies (0=x, 1=y, 2=z).

    Returns:
        numpy array with intensity gradient.
    """
    shape = (size[2], size[1], size[0])
    image = np.zeros(shape, dtype=np.float32)

    axis_size = size[gradient_axis]
    gradient = np.linspace(min_intensity, max_intensity, axis_size, dtype=np.float32)

    if gradient_axis == 0:  # Gradient along x
        image[:, :, :] = gradient[np.newaxis, np.newaxis, :]
    elif gradient_axis == 1:  # Gradient along y
        image[:, :, :] = gradient[np.newaxis, :, np.newaxis]
    else:  # Gradient along z
        image[:, :, :] = gradient[:, np.newaxis, np.newaxis]

    return image


def create_noisy_sphere(
    size: Tuple[int, int, int] = (50, 50, 50),
    center: Tuple[int, int, int] = None,
    radius: float = 15.0,
    inside_mean: float = 200.0,
    outside_mean: float = 50.0,
    inside_std: float = 10.0,
    outside_std: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a spherical region with different intensity than background.

    Args:
        size: Image size as (width, height, depth).
        center: Sphere center (x, y, z). Defaults to image center.
        radius: Sphere radius in voxels.
        inside_mean: Mean intensity inside sphere.
        outside_mean: Mean intensity outside sphere.
        inside_std: Noise standard deviation inside sphere.
        outside_std: Noise standard deviation outside sphere.

    Returns:
        Tuple of (image array, ground truth binary mask).
    """
    if center is None:
        center = (size[0] // 2, size[1] // 2, size[2] // 2)

    shape = (size[2], size[1], size[0])
    image = np.zeros(shape, dtype=np.float32)

    # Create coordinate grids
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]

    # Calculate distance from center
    distance = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    )

    # Create mask
    mask = (distance <= radius).astype(np.uint8)

    # Fill with noisy intensities
    image[mask == 1] = np.random.normal(
        inside_mean, inside_std, np.sum(mask == 1)
    ).astype(np.float32)
    image[mask == 0] = np.random.normal(
        outside_mean, outside_std, np.sum(mask == 0)
    ).astype(np.float32)

    return image, mask


def create_concentric_spheres(
    size: Tuple[int, int, int] = (50, 50, 50),
    center: Tuple[int, int, int] = None,
    radii: Tuple[float, ...] = (5.0, 10.0, 15.0),
    intensities: Tuple[float, ...] = (250.0, 150.0, 50.0),
    noise_std: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create concentric spherical shells with different intensities.

    Useful for testing edge detection at multiple boundaries.

    Args:
        size: Image size as (width, height, depth).
        center: Sphere center. Defaults to image center.
        radii: Radii of spheres (inner to outer).
        intensities: Intensity for each shell (plus background).
        noise_std: Noise standard deviation.

    Returns:
        Tuple of (image array, label mask with different values per shell).
    """
    if center is None:
        center = (size[0] // 2, size[1] // 2, size[2] // 2)

    if len(intensities) != len(radii) + 1:
        raise ValueError("Need one more intensity than radii (for background)")

    shape = (size[2], size[1], size[0])
    image = np.full(shape, intensities[-1], dtype=np.float32)  # Background
    mask = np.zeros(shape, dtype=np.uint8)

    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    distance = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    )

    # Fill from outer to inner
    for i, r in enumerate(reversed(radii)):
        idx = len(radii) - 1 - i
        shell_mask = distance <= r
        image[shell_mask] = intensities[idx]
        mask[shell_mask] = idx + 1

    # Add noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, shape).astype(np.float32)
        image = image + noise

    return image, mask


def create_checkerboard_image(
    size: Tuple[int, int, int] = (100, 100, 10),
    block_size: int = 10,
    intensity1: float = 50.0,
    intensity2: float = 200.0,
    noise_std: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 3D checkerboard pattern.

    Useful for testing edge detection on complex geometry.

    Args:
        size: Image size as (width, height, depth).
        block_size: Size of each checker block.
        intensity1: Intensity of first checker color.
        intensity2: Intensity of second checker color.
        noise_std: Noise standard deviation.

    Returns:
        Tuple of (image array, binary mask of first checker color).
    """
    shape = (size[2], size[1], size[0])
    image = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)

    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                block_x = x // block_size
                block_y = y // block_size
                block_z = z // block_size
                if (block_x + block_y + block_z) % 2 == 0:
                    image[z, y, x] = intensity1
                    mask[z, y, x] = 1
                else:
                    image[z, y, x] = intensity2

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, shape).astype(np.float32)
        image = image + noise

    return image, mask
