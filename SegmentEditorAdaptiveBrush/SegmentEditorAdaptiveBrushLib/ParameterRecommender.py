"""Parameter Recommender for the Quick Select Parameters Wizard.

This module generates algorithm and parameter recommendations based on
intensity and shape analysis results, with awareness of imaging modality
and target structure type.
"""

import logging
from typing import Optional

from WizardDataStructures import (
    IntensityAnalysisResult,
    ShapeAnalysisResult,
    WizardRecommendation,
)

logger = logging.getLogger(__name__)


class ParameterRecommender:
    """Generates parameter recommendations from analysis results.

    This class implements a decision tree for selecting optimal
    segmentation algorithms and parameters based on image characteristics.
    """

    # Modality-specific hints
    MODALITY_HINTS: dict[str, dict] = {
        "CT": {
            "prefer_threshold": True,
            "hu_aware": True,
            "description": "CT images have calibrated HU values",
        },
        "MRI_T1": {
            "prefer_watershed": True,
            "good_gm_wm": True,
            "description": "T1 MRI has good gray/white matter contrast",
        },
        "MRI_T2": {
            "fluid_bright": True,
            "prefer_watershed": True,
            "description": "T2 MRI shows fluid as bright",
        },
        "Ultrasound": {
            "noisy": True,
            "prefer_region_growing": True,
            "low_confidence": True,
            "description": "Ultrasound is typically noisy",
        },
        "PET": {
            "prefer_threshold": True,
            "description": "PET has high contrast for active tissue",
        },
    }

    # Structure-specific hints
    STRUCTURE_HINTS: dict[str, dict] = {
        "tumor": {
            "irregular_boundary": True,
            "prefer_level_set": True,
            "description": "Tumors often have irregular boundaries",
        },
        "vessel": {
            "tubular": True,
            "prefer_geodesic": True,
            "description": "Vessels are tubular structures",
        },
        "bone": {
            "high_contrast": True,
            "prefer_threshold": True,
            "description": "Bone typically has high contrast in CT",
        },
        "brain_tissue": {
            "subtle_edges": True,
            "prefer_watershed": True,
            "description": "Brain tissue has subtle intensity boundaries",
        },
        "organ": {
            "smooth_boundary": True,
            "prefer_geodesic": True,
            "description": "Organs typically have smooth boundaries",
        },
        "lesion": {
            "irregular_boundary": True,
            "small_structure": True,
            "prefer_level_set": True,
            "description": "Lesions may be small with irregular boundaries",
        },
    }

    # Algorithm characteristics
    ALGORITHM_INFO: dict[str, dict] = {
        "geodesic_distance": {
            "speed": "fast",
            "precision": "high",
            "edge_aware": True,
            "handles_noise": True,
            "description": "Fast and precise, good for most cases",
        },
        "watershed": {
            "speed": "medium",
            "precision": "high",
            "edge_aware": True,
            "handles_noise": True,
            "description": "Good balance of speed and boundary accuracy",
        },
        "random_walker": {
            "speed": "slow",
            "precision": "very_high",
            "edge_aware": True,
            "handles_noise": True,
            "description": "High precision for complex boundaries",
        },
        "level_set_gpu": {
            "speed": "fast",
            "precision": "very_high",
            "edge_aware": True,
            "handles_irregular": True,
            "requires_gpu": True,
            "description": "High precision with GPU acceleration",
        },
        "level_set_cpu": {
            "speed": "slow",
            "precision": "very_high",
            "edge_aware": True,
            "handles_irregular": True,
            "description": "High precision for irregular boundaries",
        },
        "connected_threshold": {
            "speed": "very_fast",
            "precision": "low",
            "intensity_based": True,
            "description": "Very fast intensity-based segmentation",
        },
        "region_growing": {
            "speed": "fast",
            "precision": "medium",
            "intensity_based": True,
            "handles_homogeneous": True,
            "description": "Fast for homogeneous regions",
        },
        "threshold_brush": {
            "speed": "very_fast",
            "precision": "variable",
            "intensity_based": True,
            "simple": True,
            "description": "Simple threshold painting",
        },
    }

    def recommend(
        self,
        intensity_result: IntensityAnalysisResult,
        shape_result: ShapeAnalysisResult,
        modality: Optional[str] = None,
        structure_type: Optional[str] = None,
        priority: str = "balanced",
    ) -> WizardRecommendation:
        """Generate recommendation from analysis results.

        Args:
            intensity_result: Results from intensity analysis.
            shape_result: Results from shape analysis.
            modality: Optional imaging modality (CT, MRI_T1, etc.).
            structure_type: Optional structure type (tumor, bone, etc.).
            priority: User priority - "speed", "precision", or "balanced".

        Returns:
            WizardRecommendation with algorithm and parameter suggestions.
        """
        # Calculate base scores for each algorithm
        scores = self._calculate_algorithm_scores(
            intensity_result, shape_result, modality, structure_type, priority
        )

        # Select best algorithm
        best_algo = max(scores, key=lambda k: scores.get(k, 0.0))
        algorithm_reason = self._get_algorithm_reason(
            best_algo, intensity_result, shape_result, modality, structure_type
        )

        # Calculate edge sensitivity
        edge_sensitivity = self._calculate_edge_sensitivity(intensity_result, shape_result)
        sensitivity_reason = self._get_sensitivity_reason(intensity_result, shape_result)

        # Determine brush radius
        brush_radius = shape_result.suggested_brush_radius_mm
        radius_reason = self._get_radius_reason(shape_result)

        # Calculate confidence
        confidence = self._calculate_confidence(intensity_result, shape_result, modality)

        # Generate warnings
        warnings = self._generate_warnings(intensity_result, shape_result, modality)

        # Generate alternatives
        alternatives = self._get_alternative_algorithms(scores, best_algo)

        return WizardRecommendation(
            algorithm=best_algo,
            algorithm_reason=algorithm_reason,
            brush_radius_mm=brush_radius,
            radius_reason=radius_reason,
            edge_sensitivity=edge_sensitivity,
            sensitivity_reason=sensitivity_reason,
            threshold_lower=intensity_result.suggested_threshold_lower,
            threshold_upper=intensity_result.suggested_threshold_upper,
            threshold_reason="Based on foreground/background intensity analysis",
            confidence=confidence,
            warnings=warnings,
            alternative_algorithms=alternatives,
        )

    def _calculate_algorithm_scores(
        self,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        modality: Optional[str],
        structure_type: Optional[str],
        priority: str,
    ) -> dict[str, float]:
        """Calculate scores for each algorithm based on conditions."""
        scores = dict.fromkeys(self.ALGORITHM_INFO, 50.0)

        # 1. Intensity separation scoring
        if intensity.is_well_separated():
            # High separation - threshold-based works well
            scores["connected_threshold"] += 25
            scores["region_growing"] += 20
            scores["threshold_brush"] += 15
        else:
            # Low separation - need edge-aware algorithms
            scores["watershed"] += 25
            scores["geodesic_distance"] += 20
            scores["level_set_cpu"] += 20
            scores["level_set_gpu"] += 20
            scores["random_walker"] += 15
            scores["connected_threshold"] -= 20
            scores["threshold_brush"] -= 15

        # 2. Boundary characteristics
        if shape.has_smooth_boundary():
            scores["connected_threshold"] += 10
            scores["region_growing"] += 10
            scores["geodesic_distance"] += 5
        else:
            # Rough/irregular boundary
            scores["level_set_cpu"] += 15
            scores["level_set_gpu"] += 15
            scores["watershed"] += 10
            scores["random_walker"] += 10

        # 3. Structure size
        if shape.is_small_structure():
            # Small structures need precision
            scores["level_set_cpu"] += 10
            scores["level_set_gpu"] += 10
            scores["geodesic_distance"] += 5
            scores["connected_threshold"] -= 5
        elif shape.is_large_structure():
            # Large structures - prefer faster algorithms
            scores["connected_threshold"] += 10
            scores["watershed"] += 5
            scores["level_set_cpu"] -= 10

        # 4. Modality-specific adjustments
        if modality:
            modality_hints = self.MODALITY_HINTS.get(modality, {})

            if modality_hints.get("prefer_threshold"):
                scores["connected_threshold"] += 15
                scores["threshold_brush"] += 10

            if modality_hints.get("prefer_watershed"):
                scores["watershed"] += 15

            if modality_hints.get("prefer_region_growing"):
                scores["region_growing"] += 15

            if modality_hints.get("noisy"):
                scores["connected_threshold"] -= 10
                scores["threshold_brush"] -= 10
                scores["watershed"] += 10
                scores["geodesic_distance"] += 10

        # 5. Structure type adjustments
        if structure_type:
            struct_hints = self.STRUCTURE_HINTS.get(structure_type, {})

            if struct_hints.get("prefer_level_set"):
                scores["level_set_cpu"] += 20
                scores["level_set_gpu"] += 20

            if struct_hints.get("prefer_geodesic"):
                scores["geodesic_distance"] += 20

            if struct_hints.get("prefer_threshold"):
                scores["connected_threshold"] += 15
                scores["threshold_brush"] += 10

            if struct_hints.get("prefer_watershed"):
                scores["watershed"] += 15

            if struct_hints.get("irregular_boundary"):
                scores["level_set_cpu"] += 10
                scores["level_set_gpu"] += 10
                scores["random_walker"] += 10

        # 6. Priority adjustments
        if priority == "speed":
            scores["connected_threshold"] += 25
            scores["threshold_brush"] += 20
            scores["region_growing"] += 15
            scores["geodesic_distance"] += 10
            scores["level_set_cpu"] -= 20
            scores["random_walker"] -= 25

        elif priority == "precision":
            scores["level_set_cpu"] += 20
            scores["level_set_gpu"] += 25
            scores["random_walker"] += 20
            scores["watershed"] += 10
            scores["geodesic_distance"] += 10
            scores["connected_threshold"] -= 10
            scores["threshold_brush"] -= 15

        # Ensure geodesic_distance is competitive (it's the recommended default)
        scores["geodesic_distance"] += 10

        return scores

    def _calculate_edge_sensitivity(
        self, intensity: IntensityAnalysisResult, shape: ShapeAnalysisResult
    ) -> int:
        """Calculate recommended edge sensitivity (0-100)."""
        # Base: 50 (middle of range)
        sensitivity = 50

        # Adjust for boundary roughness (rougher = higher sensitivity)
        sensitivity += int(shape.boundary_roughness * 30)

        # Adjust for intensity separation (poor separation = higher sensitivity)
        if intensity.separation_score < 0.5:
            sensitivity += 15
        elif intensity.separation_score > 0.8:
            sensitivity -= 10

        # Adjust for structure size (smaller = higher sensitivity)
        if shape.is_small_structure():
            sensitivity += 10
        elif shape.is_large_structure():
            sensitivity -= 5

        return max(0, min(100, sensitivity))

    def _calculate_confidence(
        self,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        modality: Optional[str],
    ) -> float:
        """Calculate overall confidence in the recommendation."""
        confidence = 0.5  # Start at middle

        # Higher separation = higher confidence
        confidence += intensity.separation_score * 0.3

        # Lower roughness = higher confidence
        confidence += (1.0 - shape.boundary_roughness) * 0.1

        # Known modality increases confidence
        if modality:
            if modality in self.MODALITY_HINTS:
                confidence += 0.1
                # But noisy modalities decrease it
                if self.MODALITY_HINTS[modality].get("low_confidence"):
                    confidence -= 0.15

        # Clamp to valid range
        return float(max(0.0, min(1.0, confidence)))

    def _get_algorithm_reason(
        self,
        algorithm: str,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        modality: Optional[str],
        structure_type: Optional[str],
    ) -> str:
        """Generate explanation for algorithm choice."""
        reasons = []

        algo_info = self.ALGORITHM_INFO.get(algorithm, {})

        # Base description
        if algo_info.get("description"):
            reasons.append(algo_info["description"])

        # Add specific reasons
        if intensity.is_well_separated() and algo_info.get("intensity_based"):
            reasons.append(
                "Well-separated intensity distributions favor threshold-based approaches"
            )
        elif not intensity.is_well_separated() and algo_info.get("edge_aware"):
            reasons.append("Low intensity separation requires edge-aware segmentation")

        if shape.boundary_roughness > 0.5 and algo_info.get("handles_irregular"):
            reasons.append("Handles irregular boundaries well")

        if modality and modality in self.MODALITY_HINTS:
            hint = self.MODALITY_HINTS[modality]
            if hint.get("description"):
                reasons.append(hint["description"])

        if structure_type and structure_type in self.STRUCTURE_HINTS:
            hint = self.STRUCTURE_HINTS[structure_type]
            if hint.get("description"):
                reasons.append(hint["description"])

        return ". ".join(reasons) if reasons else "Recommended based on analysis"

    def _get_sensitivity_reason(
        self, intensity: IntensityAnalysisResult, shape: ShapeAnalysisResult
    ) -> str:
        """Generate explanation for edge sensitivity choice."""
        reasons = []

        if shape.boundary_roughness > 0.5:
            reasons.append("Higher sensitivity to follow irregular boundaries")
        elif shape.boundary_roughness < 0.3:
            reasons.append("Lower sensitivity appropriate for smooth boundaries")

        if intensity.separation_score < 0.5:
            reasons.append("Increased sensitivity due to low intensity contrast")

        if shape.is_small_structure():
            reasons.append("Higher precision for small structure")

        return ". ".join(reasons) if reasons else "Default sensitivity based on analysis"

    def _get_radius_reason(self, shape: ShapeAnalysisResult) -> str:
        """Generate explanation for brush radius choice."""
        return (
            f"Based on estimated structure diameter of {shape.estimated_diameter_mm:.1f}mm "
            f"(approximately 1/4 of structure size)"
        )

    def _generate_warnings(
        self,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        modality: Optional[str],
    ) -> list[str]:
        """Generate warnings about potential issues."""
        warnings = []

        # Low separation warning
        if intensity.separation_score < 0.4:
            warnings.append(
                "Low intensity separation between foreground and background - "
                "boundaries may be difficult to detect automatically"
            )

        # High overlap warning
        if intensity.overlap_percentage > 50:
            warnings.append(
                f"High intensity overlap ({intensity.overlap_percentage:.0f}%) - "
                "consider manual threshold adjustment"
            )

        # Large structure warning
        if shape.is_large_structure():
            warnings.append("Large structure may require multiple brush strokes")

        # Noisy modality warning
        if modality and self.MODALITY_HINTS.get(modality, {}).get("noisy"):
            warnings.append(f"{modality} images are typically noisy - results may vary")

        return warnings

    def _get_alternative_algorithms(
        self, scores: dict[str, float], primary: str
    ) -> list[tuple[str, str]]:
        """Get alternative algorithm suggestions."""
        # Sort by score, excluding primary
        sorted_algos = sorted(
            [(algo, score) for algo, score in scores.items() if algo != primary],
            key=lambda x: x[1],
            reverse=True,
        )

        alternatives = []
        for algo, _score in sorted_algos[:3]:  # Top 3 alternatives
            reason = self._get_alternative_reason(algo, primary)
            alternatives.append((algo, reason))

        return alternatives

    def _get_alternative_reason(self, alternative: str, primary: str) -> str:
        """Generate reason for an alternative algorithm."""
        _ = primary  # May be used for future comparative reasons
        alt_info = self.ALGORITHM_INFO.get(alternative, {})

        if alt_info.get("speed") == "very_fast":
            return "Faster option if speed is prioritized"

        if alt_info.get("precision") == "very_high":
            return "Higher precision option for complex boundaries"

        if alt_info.get("handles_irregular"):
            return "Better for irregular or complex boundaries"

        if alt_info.get("handles_homogeneous"):
            return "Good for homogeneous intensity regions"

        return str(alt_info.get("description", "Alternative approach"))
