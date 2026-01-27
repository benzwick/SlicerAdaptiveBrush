"""Segmentation comparison metrics.

Provides Dice coefficient, Hausdorff distance, and volume measurements
for comparing trial segmentations against gold standards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentationMetrics:
    """Container for segmentation comparison metrics."""

    dice: float
    hausdorff_mm: float | None  # None if spacing not available
    volume_trial_mm3: float | None
    volume_gold_mm3: float | None
    volume_diff_percent: float | None
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    true_negative_count: int
    sensitivity: float  # TP / (TP + FN)
    specificity: float  # TN / (TN + FP)
    precision: float  # TP / (TP + FP)

    @property
    def jaccard(self) -> float:
        """Jaccard index (IoU) from Dice coefficient."""
        if self.dice == 0:
            return 0.0
        return self.dice / (2 - self.dice)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dice": self.dice,
            "hausdorff_mm": self.hausdorff_mm,
            "jaccard": self.jaccard,
            "volume_trial_mm3": self.volume_trial_mm3,
            "volume_gold_mm3": self.volume_gold_mm3,
            "volume_diff_percent": self.volume_diff_percent,
            "true_positive_count": self.true_positive_count,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "true_negative_count": self.true_negative_count,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
        }


class ComparisonMetrics:
    """Calculate segmentation comparison metrics.

    Computes Dice, Hausdorff distance, volume, and confusion matrix
    metrics for comparing trial segmentations to gold standards.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self._last_metrics: SegmentationMetrics | None = None

    def compute(
        self,
        trial_mask: np.ndarray,
        gold_mask: np.ndarray,
        spacing: tuple[float, float, float] | None = None,
    ) -> SegmentationMetrics:
        """Compute all comparison metrics.

        Args:
            trial_mask: Binary trial segmentation mask (3D or 2D).
            gold_mask: Binary gold standard mask (same shape as trial).
            spacing: Voxel spacing in mm (x, y, z). If None, volumes
                     and Hausdorff are computed in voxels.

        Returns:
            SegmentationMetrics with all computed values.
        """
        # Ensure boolean masks
        trial = trial_mask.astype(bool)
        gold = gold_mask.astype(bool)

        # Confusion matrix counts
        tp = np.sum(trial & gold)
        fp = np.sum(trial & ~gold)
        fn = np.sum(~trial & gold)
        tn = np.sum(~trial & ~gold)

        # Dice coefficient
        dice = self._compute_dice(tp, fp, fn)

        # Sensitivity, Specificity, Precision
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Volumes
        if spacing is not None:
            voxel_volume = spacing[0] * spacing[1] * spacing[2]
            volume_trial = float(np.sum(trial)) * voxel_volume
            volume_gold = float(np.sum(gold)) * voxel_volume

            if volume_gold > 0:
                volume_diff = ((volume_trial - volume_gold) / volume_gold) * 100
            else:
                volume_diff = None
        else:
            volume_trial = None
            volume_gold = None
            volume_diff = None

        # Hausdorff distance
        hausdorff = self._compute_hausdorff(trial, gold, spacing)

        metrics = SegmentationMetrics(
            dice=dice,
            hausdorff_mm=hausdorff,
            volume_trial_mm3=volume_trial,
            volume_gold_mm3=volume_gold,
            volume_diff_percent=volume_diff,
            true_positive_count=int(tp),
            false_positive_count=int(fp),
            false_negative_count=int(fn),
            true_negative_count=int(tn),
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
        )

        self._last_metrics = metrics
        return metrics

    def _compute_dice(self, tp: int, fp: int, fn: int) -> float:
        """Compute Dice coefficient from confusion matrix counts.

        Dice = 2*TP / (2*TP + FP + FN)

        Args:
            tp: True positive count.
            fp: False positive count.
            fn: False negative count.

        Returns:
            Dice coefficient (0-1).
        """
        denominator = 2 * tp + fp + fn
        if denominator == 0:
            return 0.0
        return (2 * tp) / denominator

    def _compute_hausdorff(
        self,
        trial: np.ndarray,
        gold: np.ndarray,
        spacing: tuple[float, float, float] | None = None,
    ) -> float | None:
        """Compute Hausdorff distance between surfaces.

        Args:
            trial: Binary trial mask.
            gold: Binary gold mask.
            spacing: Voxel spacing (optional).

        Returns:
            Hausdorff distance in mm, or None if either mask is empty.
        """
        try:
            from scipy import ndimage

            # Check for empty masks
            if not np.any(trial) or not np.any(gold):
                return None

            # Get surface points (using distance transform)
            # Surface = voxels that have at least one non-object neighbor
            trial_surface = self._get_surface(trial)
            gold_surface = self._get_surface(gold)

            if not np.any(trial_surface) or not np.any(gold_surface):
                return None

            # Compute distance transform from each surface
            trial_dist = ndimage.distance_transform_edt(~trial_surface, sampling=spacing)
            gold_dist = ndimage.distance_transform_edt(~gold_surface, sampling=spacing)

            # Hausdorff = max of forward and backward distances
            forward_max = np.max(gold_dist[trial_surface])
            backward_max = np.max(trial_dist[gold_surface])

            return float(max(forward_max, backward_max))

        except Exception as e:
            logger.warning(f"Hausdorff computation failed: {e}")
            return None

    def _get_surface(self, mask: np.ndarray) -> np.ndarray:
        """Extract surface voxels from binary mask.

        Surface voxels are object voxels with at least one
        non-object neighbor (6-connectivity).

        Args:
            mask: Binary mask.

        Returns:
            Binary mask of surface voxels only.
        """
        from scipy import ndimage

        # Erode to find interior
        if mask.ndim == 3:
            struct = ndimage.generate_binary_structure(3, 1)
        else:
            struct = ndimage.generate_binary_structure(2, 1)

        eroded = ndimage.binary_erosion(mask, structure=struct)

        # Surface = mask - interior
        return mask & ~eroded

    def compute_slice_metrics(
        self,
        trial_mask: np.ndarray,
        gold_mask: np.ndarray,
        axis: int = 0,
    ) -> list[dict[str, float]]:
        """Compute per-slice metrics along an axis.

        Args:
            trial_mask: 3D binary trial mask.
            gold_mask: 3D binary gold mask.
            axis: Axis to slice along (0=K, 1=J, 2=I for IJK order).

        Returns:
            List of dicts with slice index and Dice coefficient.
        """
        results = []
        n_slices = trial_mask.shape[axis]

        for i in range(n_slices):
            # Extract slice
            if axis == 0:
                t_slice = trial_mask[i, :, :]
                g_slice = gold_mask[i, :, :]
            elif axis == 1:
                t_slice = trial_mask[:, i, :]
                g_slice = gold_mask[:, i, :]
            else:
                t_slice = trial_mask[:, :, i]
                g_slice = gold_mask[:, :, i]

            # Compute Dice for this slice
            tp = np.sum(t_slice & g_slice)
            fp = np.sum(t_slice & ~g_slice)
            fn = np.sum(~t_slice & g_slice)

            dice = self._compute_dice(tp, fp, fn)

            results.append(
                {
                    "slice_index": i,
                    "dice": dice,
                    "has_trial": bool(np.any(t_slice)),
                    "has_gold": bool(np.any(g_slice)),
                }
            )

        return results

    def format_summary(self, metrics: SegmentationMetrics | None = None) -> str:
        """Format metrics as human-readable summary.

        Args:
            metrics: Metrics to format. Uses last computed if None.

        Returns:
            Formatted string summary.
        """
        m = metrics or self._last_metrics
        if m is None:
            return "No metrics computed"

        lines = [
            f"Dice: {m.dice:.4f}",
            f"Jaccard (IoU): {m.jaccard:.4f}",
        ]

        if m.hausdorff_mm is not None:
            lines.append(f"Hausdorff: {m.hausdorff_mm:.2f}mm")

        if m.volume_trial_mm3 is not None:
            lines.append(f"Volume (trial): {m.volume_trial_mm3:.1f}mm³")
            lines.append(f"Volume (gold): {m.volume_gold_mm3:.1f}mm³")
            if m.volume_diff_percent is not None:
                lines.append(f"Volume diff: {m.volume_diff_percent:+.1f}%")

        lines.extend(
            [
                "",
                f"Sensitivity: {m.sensitivity:.4f}",
                f"Specificity: {m.specificity:.4f}",
                f"Precision: {m.precision:.4f}",
                "",
                f"TP: {m.true_positive_count:,}",
                f"FP: {m.false_positive_count:,}",
                f"FN: {m.false_negative_count:,}",
            ]
        )

        return "\n".join(lines)

    @property
    def last_metrics(self) -> SegmentationMetrics | None:
        """Get last computed metrics."""
        return self._last_metrics


def compute_metrics_from_nodes(
    trial_node: Any,
    gold_node: Any,
    reference_volume: Any = None,
) -> SegmentationMetrics | None:
    """Compute metrics from Slicer segmentation nodes.

    Args:
        trial_node: Trial vtkMRMLSegmentationNode.
        gold_node: Gold standard vtkMRMLSegmentationNode.
        reference_volume: Optional reference volume for geometry.

    Returns:
        SegmentationMetrics or None if computation fails.
    """
    try:
        import slicer
        import vtk

        # Get reference volume if not provided
        if reference_volume is None:
            # Try to find from parameter set
            param_set = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentEditorNode")
            if param_set:
                reference_volume = param_set.GetSourceVolumeNode()

        if reference_volume is None:
            logger.warning("No reference volume available")
            return None

        # Export segmentations to labelmaps aligned with reference volume
        def export_seg_to_array(seg_node):
            labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

            segmentation = seg_node.GetSegmentation()
            segment_id = segmentation.GetNthSegmentID(0)

            segment_ids = vtk.vtkStringArray()
            segment_ids.InsertNextValue(segment_id)

            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                seg_node, segment_ids, labelmap_node, reference_volume
            )

            array = slicer.util.arrayFromVolume(labelmap_node)
            slicer.mrmlScene.RemoveNode(labelmap_node)

            return array

        trial_array = export_seg_to_array(trial_node)
        gold_array = export_seg_to_array(gold_node)

        # Get spacing from reference volume
        spacing = reference_volume.GetSpacing()

        # Compute metrics
        calculator = ComparisonMetrics()
        return calculator.compute(trial_array > 0, gold_array > 0, spacing=tuple(spacing))

    except Exception as e:
        logger.exception(f"Failed to compute metrics from nodes: {e}")
        return None
