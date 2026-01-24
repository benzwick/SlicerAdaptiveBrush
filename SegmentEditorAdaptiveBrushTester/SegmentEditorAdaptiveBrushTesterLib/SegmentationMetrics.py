"""Segmentation quality metrics computation.

Computes Dice coefficient, Hausdorff distance, and other metrics for
comparing test segmentations against reference (gold standard) segmentations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Result of comparing two segmentations."""

    dice: float
    hausdorff_max: float
    hausdorff_avg: float
    hausdorff_95: float
    volume_similarity: float
    false_positive_rate: float
    false_negative_rate: float
    test_voxels: int
    reference_voxels: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dice": self.dice,
            "hausdorff_max": self.hausdorff_max,
            "hausdorff_avg": self.hausdorff_avg,
            "hausdorff_95": self.hausdorff_95,
            "volume_similarity": self.volume_similarity,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "test_voxels": self.test_voxels,
            "reference_voxels": self.reference_voxels,
        }


class SegmentationMetrics:
    """Compute comparison metrics between segmentations.

    Uses SimpleITK for Dice, Hausdorff, and overlap metrics.

    Usage:
        metrics = SegmentationMetrics.compute(
            test_seg_node, test_segment_id,
            ref_seg_node, ref_segment_id,
            volume_node
        )
        print(f"Dice: {metrics.dice:.3f}")
        print(f"Hausdorff 95%: {metrics.hausdorff_95:.1f}mm")
    """

    @staticmethod
    def compute(
        test_seg_node,
        test_segment_id: str,
        reference_seg_node,
        reference_segment_id: str,
        volume_node,
    ) -> MetricsResult:
        """Compute Dice and Hausdorff between test and reference segmentations.

        Args:
            test_seg_node: Test segmentation MRML node.
            test_segment_id: Segment ID within test segmentation.
            reference_seg_node: Reference (gold) segmentation MRML node.
            reference_segment_id: Segment ID within reference segmentation.
            volume_node: Volume node providing geometry (spacing).

        Returns:
            MetricsResult with all computed metrics.
        """
        import SimpleITK as sitk
        import slicer

        # Extract arrays
        test_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
            test_seg_node, test_segment_id, volume_node
        ).astype(np.uint8)

        ref_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
            reference_seg_node, reference_segment_id, volume_node
        ).astype(np.uint8)

        # Ensure binary
        test_arr = (test_arr > 0).astype(np.uint8)
        ref_arr = (ref_arr > 0).astype(np.uint8)

        # Count voxels
        test_voxels = int(np.sum(test_arr))
        ref_voxels = int(np.sum(ref_arr))

        logger.debug(f"Test voxels: {test_voxels}, Reference voxels: {ref_voxels}")

        # Handle edge cases
        if test_voxels == 0 and ref_voxels == 0:
            # Both empty - perfect match
            return MetricsResult(
                dice=1.0,
                hausdorff_max=0.0,
                hausdorff_avg=0.0,
                hausdorff_95=0.0,
                volume_similarity=1.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
                test_voxels=0,
                reference_voxels=0,
            )

        if test_voxels == 0 or ref_voxels == 0:
            # One empty - worst case
            return MetricsResult(
                dice=0.0,
                hausdorff_max=float("inf"),
                hausdorff_avg=float("inf"),
                hausdorff_95=float("inf"),
                volume_similarity=0.0,
                false_positive_rate=1.0 if test_voxels > 0 else 0.0,
                false_negative_rate=1.0 if ref_voxels > 0 else 0.0,
                test_voxels=test_voxels,
                reference_voxels=ref_voxels,
            )

        # Convert to SimpleITK
        test_sitk = sitk.GetImageFromArray(test_arr)
        ref_sitk = sitk.GetImageFromArray(ref_arr)

        # Set spacing for distance metrics (mm)
        spacing = volume_node.GetSpacing()
        # Note: Slicer arrays are KJI order, SimpleITK expects IJK spacing
        test_sitk.SetSpacing((spacing[2], spacing[1], spacing[0]))
        ref_sitk.SetSpacing((spacing[2], spacing[1], spacing[0]))

        # Dice and overlap metrics
        overlap = sitk.LabelOverlapMeasuresImageFilter()
        overlap.Execute(ref_sitk, test_sitk)

        dice = overlap.GetDiceCoefficient()
        volume_sim = overlap.GetVolumeSimilarity()
        fp_rate = overlap.GetFalsePositiveError()
        fn_rate = overlap.GetFalseNegativeError()

        logger.debug(f"Dice: {dice:.4f}, Volume similarity: {volume_sim:.4f}")
        logger.debug(f"FP rate: {fp_rate:.4f}, FN rate: {fn_rate:.4f}")

        # Hausdorff distance
        hausdorff = sitk.HausdorffDistanceImageFilter()
        hausdorff.Execute(ref_sitk, test_sitk)
        hd_max = hausdorff.GetHausdorffDistance()
        hd_avg = hausdorff.GetAverageHausdorffDistance()

        logger.debug(f"Hausdorff max: {hd_max:.2f}mm, avg: {hd_avg:.2f}mm")

        # 95th percentile Hausdorff
        hd_95 = SegmentationMetrics._compute_hd95(
            test_arr, ref_arr, (spacing[2], spacing[1], spacing[0])
        )

        logger.debug(f"Hausdorff 95%: {hd_95:.2f}mm")

        return MetricsResult(
            dice=dice,
            hausdorff_max=hd_max,
            hausdorff_avg=hd_avg,
            hausdorff_95=hd_95,
            volume_similarity=volume_sim,
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            test_voxels=test_voxels,
            reference_voxels=ref_voxels,
        )

    @staticmethod
    def _compute_hd95(
        test_arr: np.ndarray, ref_arr: np.ndarray, spacing: tuple[float, float, float]
    ) -> float:
        """Compute 95th percentile Hausdorff distance.

        Args:
            test_arr: Test binary array (KJI order).
            ref_arr: Reference binary array (KJI order).
            spacing: Voxel spacing in mm (K, J, I order).

        Returns:
            95th percentile Hausdorff distance in mm.
        """
        from scipy.ndimage import distance_transform_edt

        # Compute distance transforms from boundaries
        # distance_transform_edt gives distance to nearest 0 (background)
        test_dt = distance_transform_edt(test_arr == 0, sampling=spacing)
        ref_dt = distance_transform_edt(ref_arr == 0, sampling=spacing)

        # Get surface voxels (boundary of each segmentation)
        # Surface = foreground voxels adjacent to background
        from scipy.ndimage import binary_erosion

        test_eroded = binary_erosion(test_arr)
        ref_eroded = binary_erosion(ref_arr)

        test_surface = test_arr & ~test_eroded
        ref_surface = ref_arr & ~ref_eroded

        # Distances from test surface to reference
        test_to_ref = ref_dt[test_surface > 0]
        # Distances from reference surface to test
        ref_to_test = test_dt[ref_surface > 0]

        if len(test_to_ref) == 0 or len(ref_to_test) == 0:
            return float("inf")

        # Symmetric Hausdorff - take all surface distances
        all_distances = np.concatenate([test_to_ref, ref_to_test])

        return float(np.percentile(all_distances, 95))

    @staticmethod
    def compute_dice_only(
        test_seg_node,
        test_segment_id: str,
        reference_seg_node,
        reference_segment_id: str,
        volume_node,
    ) -> float:
        """Compute Dice coefficient only (faster than full metrics).

        Args:
            test_seg_node: Test segmentation MRML node.
            test_segment_id: Segment ID within test segmentation.
            reference_seg_node: Reference (gold) segmentation MRML node.
            reference_segment_id: Segment ID within reference segmentation.
            volume_node: Volume node providing geometry.

        Returns:
            Dice coefficient (0-1).
        """
        import slicer

        test_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
            test_seg_node, test_segment_id, volume_node
        )
        ref_arr = slicer.util.arrayFromSegmentBinaryLabelmap(
            reference_seg_node, reference_segment_id, volume_node
        )

        test_binary = test_arr > 0
        ref_binary = ref_arr > 0

        intersection: int = int(np.sum(test_binary & ref_binary))
        union: int = int(np.sum(test_binary)) + int(np.sum(ref_binary))

        if union == 0:
            return 1.0  # Both empty

        return float(2.0 * intersection / union)


@dataclass
class StrokeRecord:
    """Record of a single stroke's metrics."""

    stroke: int
    params: dict
    dice: float
    hausdorff_95: float
    voxels: int
    dice_delta: float
    diminishing: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "stroke": self.stroke,
            "params": self.params,
            "dice": self.dice,
            "hausdorff_95": self.hausdorff_95,
            "voxels": self.voxels,
            "dice_delta": self.dice_delta,
            "diminishing": self.diminishing,
        }


class StrokeMetricsTracker:
    """Track how metrics improve with each stroke.

    Usage:
        tracker = StrokeMetricsTracker(gold_seg, gold_id, volume)

        for stroke_params in stroke_list:
            # Apply stroke to test_seg...
            record = tracker.record_stroke(test_seg, test_id, stroke_params)
            print(f"Stroke {record.stroke}: Dice={record.dice:.3f}")

        summary = tracker.get_summary()
        print(f"Strokes to 90% Dice: {summary['strokes_to_90pct']}")
    """

    def __init__(
        self, gold_seg_node, gold_segment_id: str, volume_node, diminishing_threshold: float = 0.01
    ) -> None:
        """Initialize tracker.

        Args:
            gold_seg_node: Gold standard segmentation MRML node.
            gold_segment_id: Segment ID within gold standard.
            volume_node: Volume node providing geometry.
            diminishing_threshold: Dice improvement below this is "diminishing returns".
        """
        self.gold_seg_node = gold_seg_node
        self.gold_segment_id = gold_segment_id
        self.volume_node = volume_node
        self.diminishing_threshold = diminishing_threshold
        self.stroke_history: list[StrokeRecord] = []

    def record_stroke(
        self,
        test_seg_node,
        test_segment_id: str,
        stroke_params: dict,
    ) -> StrokeRecord:
        """Record metrics after a stroke.

        Args:
            test_seg_node: Test segmentation MRML node.
            test_segment_id: Segment ID within test segmentation.
            stroke_params: Parameters used for this stroke.

        Returns:
            StrokeRecord with computed metrics.
        """
        # Compute full metrics
        metrics = SegmentationMetrics.compute(
            test_seg_node,
            test_segment_id,
            self.gold_seg_node,
            self.gold_segment_id,
            self.volume_node,
        )

        stroke_num = len(self.stroke_history) + 1

        # Calculate improvement from previous stroke
        if self.stroke_history:
            prev_dice = self.stroke_history[-1].dice
            dice_delta = metrics.dice - prev_dice
        else:
            dice_delta = metrics.dice  # First stroke

        diminishing = dice_delta < self.diminishing_threshold

        record = StrokeRecord(
            stroke=stroke_num,
            params=stroke_params,
            dice=metrics.dice,
            hausdorff_95=metrics.hausdorff_95,
            voxels=metrics.test_voxels,
            dice_delta=dice_delta,
            diminishing=diminishing,
        )

        self.stroke_history.append(record)

        logger.info(
            f"Stroke {stroke_num}: Dice={metrics.dice:.3f} "
            f"(delta={dice_delta:+.3f}), HD95={metrics.hausdorff_95:.1f}mm, "
            f"voxels={metrics.test_voxels}"
        )

        return record

    def get_summary(self) -> dict:
        """Summarize stroke progression.

        Returns:
            Dictionary with summary statistics.
        """
        if not self.stroke_history:
            return {
                "total_strokes": 0,
                "final_dice": 0.0,
                "best_dice": 0.0,
                "strokes_to_90pct": None,
                "strokes_to_95pct": None,
                "diminishing_returns_at": None,
            }

        return {
            "total_strokes": len(self.stroke_history),
            "final_dice": self.stroke_history[-1].dice,
            "best_dice": max(s.dice for s in self.stroke_history),
            "best_dice_stroke": max(self.stroke_history, key=lambda s: s.dice).stroke,
            "strokes_to_90pct": self._strokes_to_threshold(0.9),
            "strokes_to_95pct": self._strokes_to_threshold(0.95),
            "diminishing_returns_at": self._first_diminishing(),
            "stroke_history": [s.to_dict() for s in self.stroke_history],
        }

    def _strokes_to_threshold(self, threshold: float) -> int | None:
        """Find number of strokes needed to reach a Dice threshold."""
        for s in self.stroke_history:
            if s.dice >= threshold:
                return s.stroke
        return None

    def _first_diminishing(self) -> int | None:
        """Find first stroke with diminishing returns."""
        for s in self.stroke_history:
            if s.diminishing:
                return s.stroke
        return None

    def reset(self) -> None:
        """Clear stroke history for a new trial."""
        self.stroke_history.clear()
