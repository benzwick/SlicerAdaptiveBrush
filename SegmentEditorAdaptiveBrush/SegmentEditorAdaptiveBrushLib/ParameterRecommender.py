"""Parameter Recommender for the Quick Select Parameters Wizard.

This module generates algorithm and parameter recommendations based on
intensity and shape analysis results, with awareness of imaging modality
and target structure type.

The recommendation engine uses a multi-factor scoring system that considers:
- Intensity distribution separation
- Boundary characteristics
- Structure size and shape
- Imaging modality characteristics
- Target structure type
- User priority (speed vs precision)
"""

import logging
from typing import Any, Optional

from WizardDataStructures import (
    IntensityAnalysisResult,
    ShapeAnalysisResult,
    WizardRecommendation,
)

logger = logging.getLogger(__name__)


class ParameterRecommender:
    """Generates parameter recommendations from analysis results.

    This class implements a sophisticated decision system for selecting optimal
    segmentation algorithms and parameters based on image characteristics.
    """

    # ========================================================================
    # MODALITY-SPECIFIC CONFIGURATION
    # ========================================================================

    MODALITY_HINTS: dict[str, dict] = {
        "CT": {
            "prefer_threshold": True,
            "hu_aware": True,
            "typical_range": (-1000, 3000),  # HU range
            "noise_level": "low",
            "edge_quality": "high",
            "description": "CT images have calibrated HU values with good edge definition",
            "recommended_sampling": "percentile",
            "recommended_sigma": 0.3,
        },
        "MRI_T1": {
            "prefer_watershed": True,
            "good_gm_wm": True,
            "noise_level": "medium",
            "edge_quality": "medium",
            "description": "T1 MRI has good gray/white matter contrast",
            "recommended_sampling": "mean_std",
            "recommended_sigma": 0.5,
        },
        "MRI_T2": {
            "fluid_bright": True,
            "prefer_watershed": True,
            "noise_level": "medium",
            "edge_quality": "medium",
            "description": "T2 MRI shows fluid as bright, good for edema/lesions",
            "recommended_sampling": "mean_std",
            "recommended_sigma": 0.5,
        },
        "MRI_FLAIR": {
            "csf_suppressed": True,
            "prefer_watershed": True,
            "noise_level": "medium",
            "edge_quality": "high",
            "description": "FLAIR suppresses CSF, excellent for periventricular lesions",
            "recommended_sampling": "mean_std",
            "recommended_sigma": 0.4,
        },
        "MRI_DWI": {
            "diffusion_weighted": True,
            "prefer_threshold": True,
            "noise_level": "high",
            "edge_quality": "low",
            "description": "DWI shows restricted diffusion, often noisy",
            "recommended_sampling": "percentile",
            "recommended_sigma": 0.8,
        },
        "Ultrasound": {
            "noisy": True,
            "speckle": True,
            "prefer_region_growing": True,
            "low_confidence": True,
            "noise_level": "very_high",
            "edge_quality": "low",
            "description": "Ultrasound is noisy with speckle artifacts",
            "recommended_sampling": "percentile",
            "recommended_sigma": 1.0,
        },
        "PET": {
            "prefer_threshold": True,
            "suv_based": True,
            "noise_level": "medium",
            "edge_quality": "low",
            "description": "PET has high contrast for metabolically active tissue",
            "recommended_sampling": "percentile",
            "recommended_sigma": 0.6,
        },
        "PET_CT": {
            "prefer_threshold": True,
            "suv_based": True,
            "noise_level": "medium",
            "edge_quality": "medium",
            "description": "Fused PET/CT combines metabolic and anatomic information",
            "recommended_sampling": "percentile",
            "recommended_sigma": 0.5,
        },
        "CBCT": {
            "prefer_threshold": True,
            "noise_level": "medium",
            "edge_quality": "medium",
            "description": "Cone-beam CT, noisier than conventional CT",
            "recommended_sampling": "percentile",
            "recommended_sigma": 0.5,
        },
        "Microscopy": {
            "prefer_watershed": True,
            "noise_level": "low",
            "edge_quality": "high",
            "description": "Microscopy images typically have clear cell boundaries",
            "recommended_sampling": "mean_std",
            "recommended_sigma": 0.3,
        },
    }

    # ========================================================================
    # STRUCTURE-TYPE CONFIGURATION
    # ========================================================================

    STRUCTURE_HINTS: dict[str, dict] = {
        "tumor": {
            "irregular_boundary": True,
            "prefer_level_set": True,
            "heterogeneous": True,
            "description": "Tumors often have irregular, infiltrating boundaries",
            "recommended_std_mult": 2.5,
            "fill_holes": True,
            "closing_radius": 1,
        },
        "vessel": {
            "tubular": True,
            "prefer_geodesic": True,
            "thin_structure": True,
            "description": "Vessels are thin tubular structures",
            "recommended_std_mult": 1.5,
            "fill_holes": True,
            "closing_radius": 0,
        },
        "bone": {
            "high_contrast": True,
            "prefer_threshold": True,
            "sharp_boundary": True,
            "description": "Bone has high contrast and sharp edges in CT",
            "recommended_std_mult": 1.5,
            "fill_holes": True,
            "closing_radius": 0,
        },
        "brain_tissue": {
            "subtle_edges": True,
            "prefer_watershed": True,
            "description": "Brain tissue has subtle intensity gradients",
            "recommended_std_mult": 2.0,
            "fill_holes": True,
            "closing_radius": 0,
        },
        "organ": {
            "smooth_boundary": True,
            "prefer_geodesic": True,
            "homogeneous": True,
            "description": "Organs typically have smooth boundaries",
            "recommended_std_mult": 2.0,
            "fill_holes": True,
            "closing_radius": 1,
        },
        "lesion": {
            "irregular_boundary": True,
            "small_structure": True,
            "prefer_level_set": True,
            "description": "Lesions may be small with irregular boundaries",
            "recommended_std_mult": 2.5,
            "fill_holes": True,
            "closing_radius": 0,
        },
        "lymph_node": {
            "oval_shape": True,
            "prefer_geodesic": True,
            "description": "Lymph nodes are typically oval with smooth margins",
            "recommended_std_mult": 2.0,
            "fill_holes": True,
            "closing_radius": 0,
        },
        "muscle": {
            "homogeneous": True,
            "prefer_region_growing": True,
            "description": "Muscle has relatively uniform intensity",
            "recommended_std_mult": 2.5,
            "fill_holes": True,
            "closing_radius": 1,
        },
        "fat": {
            "high_contrast": True,
            "prefer_threshold": True,
            "description": "Fat has distinct intensity in most modalities",
            "recommended_std_mult": 2.0,
            "fill_holes": True,
            "closing_radius": 1,
        },
        "airway": {
            "tubular": True,
            "low_intensity": True,
            "prefer_region_growing": True,
            "description": "Airways are tubular low-intensity structures",
            "recommended_std_mult": 2.0,
            "fill_holes": False,  # Preserve branching
            "closing_radius": 0,
        },
        "lung": {
            "low_intensity": True,
            "prefer_threshold": True,
            "description": "Lung parenchyma has very low intensity in CT",
            "recommended_std_mult": 2.0,
            "fill_holes": False,  # Preserve airways
            "closing_radius": 0,
        },
    }

    # ========================================================================
    # ALGORITHM CHARACTERISTICS AND OPTIMAL PARAMETERS
    # ========================================================================

    ALGORITHM_INFO: dict[str, dict] = {
        "geodesic_distance": {
            "speed": "fast",
            "precision": "high",
            "edge_aware": True,
            "handles_noise": True,
            "base_score": 55,  # Slight preference as default
            "description": "Fast and precise, excellent for most cases",
            "optimal_params": {
                "default": {},
            },
        },
        "watershed": {
            "speed": "medium",
            "precision": "high",
            "edge_aware": True,
            "handles_noise": True,
            "base_score": 50,
            "description": "Excellent boundary detection for subtle edges",
            "optimal_params": {
                "default": {"gradient_scale": 1.0, "smoothing": 0.5},
                "noisy": {"gradient_scale": 1.5, "smoothing": 0.8},
                "subtle_edges": {"gradient_scale": 0.8, "smoothing": 0.3},
                "high_contrast": {"gradient_scale": 1.2, "smoothing": 0.4},
            },
        },
        "random_walker": {
            "speed": "slow",
            "precision": "very_high",
            "edge_aware": True,
            "handles_noise": True,
            "handles_irregular": True,
            "base_score": 45,
            "description": "Highest precision for complex boundaries",
            "optimal_params": {
                "default": {"beta": 130},
                "noisy": {"beta": 200},
                "high_contrast": {"beta": 80},
            },
        },
        "level_set_gpu": {
            "speed": "fast",
            "precision": "very_high",
            "edge_aware": True,
            "handles_irregular": True,
            "requires_gpu": True,
            "base_score": 48,
            "description": "High precision with GPU acceleration",
            "optimal_params": {
                "default": {"iterations": 100, "propagation": 1.0, "curvature": 0.5},
                "irregular": {"iterations": 150, "propagation": 0.8, "curvature": 0.3},
                "smooth": {"iterations": 80, "propagation": 1.2, "curvature": 0.7},
            },
        },
        "level_set_cpu": {
            "speed": "slow",
            "precision": "very_high",
            "edge_aware": True,
            "handles_irregular": True,
            "base_score": 45,
            "description": "High precision for irregular boundaries (CPU)",
            "optimal_params": {
                "default": {"iterations": 100, "propagation": 1.0, "curvature": 0.5},
                "irregular": {"iterations": 150, "propagation": 0.8, "curvature": 0.3},
                "smooth": {"iterations": 80, "propagation": 1.2, "curvature": 0.7},
            },
        },
        "connected_threshold": {
            "speed": "very_fast",
            "precision": "low",
            "intensity_based": True,
            "base_score": 40,
            "description": "Very fast intensity-based segmentation",
            "optimal_params": {
                "default": {},
            },
        },
        "region_growing": {
            "speed": "fast",
            "precision": "medium",
            "intensity_based": True,
            "handles_homogeneous": True,
            "base_score": 45,
            "description": "Fast and reliable for homogeneous regions",
            "optimal_params": {
                "default": {"multiplier": 2.5},
                "homogeneous": {"multiplier": 3.0},
                "heterogeneous": {"multiplier": 2.0},
            },
        },
        "threshold_brush": {
            "speed": "very_fast",
            "precision": "variable",
            "intensity_based": True,
            "simple": True,
            "base_score": 35,
            "description": "Simple threshold painting for quick work",
            "optimal_params": {
                "default": {},
            },
        },
    }

    # ========================================================================
    # PRESET MAPPING (links analysis results to presets)
    # ========================================================================

    PRESET_MAPPINGS: list[dict] = [
        # CT Presets
        {
            "preset_id": "ct_bone",
            "modality": "CT",
            "structure": "bone",
            "separation_min": 0.7,
        },
        {
            "preset_id": "ct_soft_tissue",
            "modality": "CT",
            "structure": "organ",
            "separation_min": 0.4,
        },
        {
            "preset_id": "ct_lung",
            "modality": "CT",
            "structure": "lung",
            "separation_min": 0.8,
        },
        {
            "preset_id": "ct_vessel_contrast",
            "modality": "CT",
            "structure": "vessel",
            "separation_min": 0.6,
        },
        # MRI Presets
        {
            "preset_id": "mri_t1_brain",
            "modality": "MRI_T1",
            "structure": "brain_tissue",
            "separation_min": 0.5,
        },
        {
            "preset_id": "mri_t1_fat",
            "modality": "MRI_T1",
            "structure": "fat",
            "separation_min": 0.6,
        },
        {
            "preset_id": "mri_t1gd_tumor",
            "modality": "MRI_T1",
            "structure": "tumor",
            "separation_min": 0.4,
        },
        {
            "preset_id": "mri_t2_lesion",
            "modality": "MRI_T2",
            "structure": "lesion",
            "separation_min": 0.4,
        },
        # Generic fallbacks
        {
            "preset_id": "generic_tumor",
            "modality": None,
            "structure": "tumor",
            "separation_min": 0.3,
        },
        {
            "preset_id": "generic_vessel",
            "modality": None,
            "structure": "vessel",
            "separation_min": 0.5,
        },
    ]

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

        # Calculate all parameters
        edge_sensitivity = self._calculate_edge_sensitivity(
            intensity_result, shape_result, modality
        )
        sensitivity_reason = self._get_sensitivity_reason(intensity_result, shape_result)

        threshold_zone = self._calculate_threshold_zone(intensity_result, shape_result, modality)

        sampling_method = self._determine_sampling_method(
            intensity_result, modality, structure_type
        )

        gaussian_sigma = self._calculate_gaussian_sigma(intensity_result, shape_result, modality)

        std_multiplier = self._calculate_std_multiplier(
            intensity_result, shape_result, structure_type
        )

        fill_holes, closing_radius = self._determine_morphology_params(shape_result, structure_type)

        # Determine brush radius
        brush_radius = shape_result.suggested_brush_radius_mm
        radius_reason = self._get_radius_reason(shape_result)

        # Get algorithm-specific parameters
        algorithm_params = self._get_algorithm_params(
            best_algo, intensity_result, shape_result, modality, structure_type
        )

        # Calculate confidence
        confidence = self._calculate_confidence(intensity_result, shape_result, modality)

        # Generate warnings
        warnings = self._generate_warnings(intensity_result, shape_result, modality)

        # Generate alternatives
        alternatives = self._get_alternative_algorithms(scores, best_algo)

        # Find best matching preset
        recommended_preset = self._find_matching_preset(intensity_result, modality, structure_type)

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
            algorithm_params=algorithm_params,
            threshold_zone=threshold_zone,
            sampling_method=sampling_method,
            gaussian_sigma=gaussian_sigma,
            std_multiplier=std_multiplier,
            fill_holes=fill_holes,
            closing_radius=closing_radius,
            recommended_preset=recommended_preset,
        )

    def _calculate_algorithm_scores(
        self,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        modality: Optional[str],
        structure_type: Optional[str],
        priority: str,
    ) -> dict[str, float]:
        """Calculate scores for each algorithm based on multiple factors."""
        # Start with base scores from algorithm info
        scores = {algo: info.get("base_score", 50.0) for algo, info in self.ALGORITHM_INFO.items()}

        # ====================================================================
        # Factor 1: Intensity Separation (weight: high)
        # ====================================================================
        sep = intensity.separation_score

        if sep >= 0.8:  # Excellent separation
            scores["connected_threshold"] += 30
            scores["region_growing"] += 25
            scores["threshold_brush"] += 20
        elif sep >= 0.6:  # Good separation
            scores["connected_threshold"] += 15
            scores["region_growing"] += 15
            scores["geodesic_distance"] += 10
            scores["watershed"] += 10
        elif sep >= 0.4:  # Moderate separation
            scores["watershed"] += 20
            scores["geodesic_distance"] += 20
            scores["level_set_cpu"] += 15
            scores["level_set_gpu"] += 15
            scores["random_walker"] += 10
            scores["connected_threshold"] -= 10
        else:  # Poor separation - need edge-aware
            scores["watershed"] += 30
            scores["geodesic_distance"] += 25
            scores["level_set_cpu"] += 25
            scores["level_set_gpu"] += 25
            scores["random_walker"] += 20
            scores["connected_threshold"] -= 25
            scores["threshold_brush"] -= 20

        # ====================================================================
        # Factor 2: Boundary Characteristics (weight: medium-high)
        # ====================================================================
        roughness = shape.boundary_roughness

        if roughness >= 0.7:  # Very irregular boundary
            scores["level_set_cpu"] += 25
            scores["level_set_gpu"] += 25
            scores["random_walker"] += 20
            scores["watershed"] += 10
            scores["connected_threshold"] -= 15
            scores["region_growing"] -= 10
        elif roughness >= 0.4:  # Moderate roughness
            scores["level_set_cpu"] += 10
            scores["level_set_gpu"] += 10
            scores["watershed"] += 15
            scores["geodesic_distance"] += 10
        else:  # Smooth boundary
            scores["geodesic_distance"] += 15
            scores["region_growing"] += 10
            scores["connected_threshold"] += 10

        # ====================================================================
        # Factor 3: Structure Size (weight: medium)
        # ====================================================================
        if shape.is_small_structure(8.0):  # Very small (<8mm)
            scores["level_set_cpu"] += 15
            scores["level_set_gpu"] += 15
            scores["random_walker"] += 10
            scores["geodesic_distance"] += 5
            scores["connected_threshold"] -= 10
        elif shape.is_small_structure(15.0):  # Small (<15mm)
            scores["level_set_cpu"] += 5
            scores["level_set_gpu"] += 5
            scores["geodesic_distance"] += 5
        elif shape.is_large_structure(80.0):  # Very large (>80mm)
            scores["connected_threshold"] += 15
            scores["region_growing"] += 10
            scores["threshold_brush"] += 10
            scores["level_set_cpu"] -= 15
            scores["random_walker"] -= 20
        elif shape.is_large_structure(50.0):  # Large (>50mm)
            scores["connected_threshold"] += 5
            scores["watershed"] += 5
            scores["level_set_cpu"] -= 5

        # ====================================================================
        # Factor 4: Modality-Specific Adjustments (weight: medium)
        # ====================================================================
        if modality:
            hints = self.MODALITY_HINTS.get(modality, {})

            if hints.get("prefer_threshold"):
                scores["connected_threshold"] += 20
                scores["threshold_brush"] += 15

            if hints.get("prefer_watershed"):
                scores["watershed"] += 20
                scores["geodesic_distance"] += 10

            if hints.get("prefer_region_growing"):
                scores["region_growing"] += 20

            # Noise handling
            noise = hints.get("noise_level", "medium")
            if noise == "very_high":
                scores["connected_threshold"] -= 20
                scores["threshold_brush"] -= 20
                scores["watershed"] += 15
                scores["geodesic_distance"] += 15
            elif noise == "high":
                scores["connected_threshold"] -= 10
                scores["watershed"] += 10
                scores["geodesic_distance"] += 10
            elif noise == "low":
                scores["connected_threshold"] += 10

            # Edge quality
            edge_q = hints.get("edge_quality", "medium")
            if edge_q == "high":
                scores["watershed"] += 10
                scores["geodesic_distance"] += 5
            elif edge_q == "low":
                scores["level_set_cpu"] += 10
                scores["level_set_gpu"] += 10
                scores["random_walker"] += 10

        # ====================================================================
        # Factor 5: Structure Type Adjustments (weight: medium)
        # ====================================================================
        if structure_type:
            struct = self.STRUCTURE_HINTS.get(structure_type, {})

            if struct.get("prefer_level_set"):
                scores["level_set_cpu"] += 25
                scores["level_set_gpu"] += 25

            if struct.get("prefer_geodesic"):
                scores["geodesic_distance"] += 25

            if struct.get("prefer_threshold"):
                scores["connected_threshold"] += 20
                scores["threshold_brush"] += 15

            if struct.get("prefer_watershed"):
                scores["watershed"] += 20

            if struct.get("prefer_region_growing"):
                scores["region_growing"] += 20

            if struct.get("irregular_boundary"):
                scores["level_set_cpu"] += 10
                scores["level_set_gpu"] += 10
                scores["random_walker"] += 10

            if struct.get("tubular"):
                scores["geodesic_distance"] += 15
                scores["region_growing"] -= 10

            if struct.get("homogeneous"):
                scores["region_growing"] += 15
                scores["connected_threshold"] += 10

        # ====================================================================
        # Factor 6: Priority Adjustments (weight: high)
        # ====================================================================
        if priority == "speed":
            scores["connected_threshold"] += 30
            scores["threshold_brush"] += 25
            scores["region_growing"] += 20
            scores["geodesic_distance"] += 15
            scores["watershed"] -= 5
            scores["level_set_cpu"] -= 25
            scores["random_walker"] -= 35
        elif priority == "precision":
            scores["level_set_cpu"] += 25
            scores["level_set_gpu"] += 30
            scores["random_walker"] += 25
            scores["watershed"] += 15
            scores["geodesic_distance"] += 10
            scores["connected_threshold"] -= 15
            scores["threshold_brush"] -= 20

        return scores

    def _calculate_edge_sensitivity(
        self,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        modality: Optional[str],
    ) -> int:
        """Calculate recommended edge sensitivity (0-100)."""
        sensitivity = 50  # Base value

        # Boundary roughness adjustment (larger impact)
        sensitivity += int(shape.boundary_roughness * 35)

        # Intensity separation adjustment
        if intensity.separation_score < 0.4:
            sensitivity += 20
        elif intensity.separation_score < 0.6:
            sensitivity += 10
        elif intensity.separation_score > 0.85:
            sensitivity -= 15

        # Structure size adjustment
        if shape.is_small_structure(10.0):
            sensitivity += 15
        elif shape.is_small_structure(20.0):
            sensitivity += 5
        elif shape.is_large_structure(60.0):
            sensitivity -= 10

        # Modality adjustment
        if modality:
            hints = self.MODALITY_HINTS.get(modality, {})
            edge_q = hints.get("edge_quality", "medium")
            if edge_q == "high":
                sensitivity -= 5
            elif edge_q == "low":
                sensitivity += 10

        return max(10, min(90, sensitivity))

    def _calculate_threshold_zone(
        self,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        modality: Optional[str],
    ) -> int:
        """Calculate recommended threshold zone (0-100)."""
        zone = 50  # Base value

        # Adjust based on overlap
        if intensity.overlap_percentage > 40:
            zone += 15  # Wider zone for overlapping distributions
        elif intensity.overlap_percentage < 10:
            zone -= 15  # Narrow zone for well-separated

        # Adjust for boundary roughness
        if shape.boundary_roughness > 0.6:
            zone += 10  # More tolerance for rough boundaries

        # Modality-specific adjustments
        if modality == "CT":
            zone -= 10  # CT has calibrated values, can be more precise
        elif modality == "Ultrasound":
            zone += 15  # Need more tolerance for noisy images

        return max(20, min(80, zone))

    def _determine_sampling_method(
        self,
        intensity: IntensityAnalysisResult,
        modality: Optional[str],
        structure_type: Optional[str],
    ) -> str:
        """Determine optimal sampling method."""
        _ = structure_type  # May be used in future
        # Check modality preference
        if modality:
            hints = self.MODALITY_HINTS.get(modality, {})
            recommended = hints.get("recommended_sampling")
            if recommended and isinstance(recommended, str):
                return str(recommended)

        # Default logic based on intensity distribution
        if intensity.separation_score > 0.7:
            return "percentile"  # Works well with clear separation
        elif intensity.separation_score < 0.4:
            return "mean_std"  # More robust for overlapping
        else:
            return "mean_std"  # Default

    def _calculate_gaussian_sigma(
        self,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        modality: Optional[str],
    ) -> float:
        """Calculate recommended Gaussian smoothing sigma."""
        sigma = 0.5  # Default

        # Check modality preference
        if modality:
            hints = self.MODALITY_HINTS.get(modality, {})
            if "recommended_sigma" in hints:
                sigma = hints["recommended_sigma"]

        # Adjust for structure size (smaller structures need less smoothing)
        if shape.is_small_structure(10.0):
            sigma = max(0.2, sigma - 0.2)

        # Adjust for boundary roughness (rougher needs more smoothing)
        if shape.boundary_roughness > 0.6:
            sigma = min(1.2, sigma + 0.2)

        return round(sigma, 2)

    def _calculate_std_multiplier(
        self,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        structure_type: Optional[str],
    ) -> float:
        """Calculate recommended standard deviation multiplier."""
        mult = 2.0  # Default

        # Check structure type preference
        if structure_type:
            hints = self.STRUCTURE_HINTS.get(structure_type, {})
            if "recommended_std_mult" in hints:
                mult = hints["recommended_std_mult"]

        # Adjust for intensity separation
        if intensity.separation_score > 0.8:
            mult = min(3.0, mult + 0.5)  # Can be more permissive
        elif intensity.separation_score < 0.4:
            mult = max(1.5, mult - 0.5)  # Need tighter threshold

        return round(mult, 1)

    def _determine_morphology_params(
        self,
        shape: ShapeAnalysisResult,
        structure_type: Optional[str],
    ) -> tuple[bool, int]:
        """Determine fill_holes and closing_radius parameters."""
        fill_holes = True
        closing_radius = 0

        # Check structure type preference
        if structure_type:
            hints = self.STRUCTURE_HINTS.get(structure_type, {})
            if "fill_holes" in hints:
                fill_holes = hints["fill_holes"]
            if "closing_radius" in hints:
                closing_radius = hints["closing_radius"]

        # Adjust for boundary roughness
        if shape.boundary_roughness > 0.6 and closing_radius == 0:
            closing_radius = 1  # Help smooth rough boundaries

        return fill_holes, closing_radius

    def _get_algorithm_params(
        self,
        algorithm: str,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        modality: Optional[str],
        structure_type: Optional[str],
    ) -> dict[str, Any]:
        """Get algorithm-specific parameter overrides."""
        algo_info = self.ALGORITHM_INFO.get(algorithm, {})
        optimal = algo_info.get("optimal_params", {})

        # Determine which preset to use
        preset_key = "default"

        # Check for noise
        if modality:
            hints = self.MODALITY_HINTS.get(modality, {})
            if hints.get("noise_level") in ("high", "very_high"):
                if "noisy" in optimal:
                    preset_key = "noisy"

        # Check for irregular boundary
        if shape.boundary_roughness > 0.6:
            if "irregular" in optimal:
                preset_key = "irregular"

        # Check for high contrast
        if intensity.separation_score > 0.8:
            if "high_contrast" in optimal:
                preset_key = "high_contrast"

        # Check for subtle edges
        if intensity.separation_score < 0.5 and shape.boundary_roughness < 0.3:
            if "subtle_edges" in optimal:
                preset_key = "subtle_edges"

        # Check for homogeneous
        if structure_type:
            struct = self.STRUCTURE_HINTS.get(structure_type, {})
            if struct.get("homogeneous"):
                if "homogeneous" in optimal:
                    preset_key = "homogeneous"

        # Check for smooth boundary
        if shape.boundary_roughness < 0.2:
            if "smooth" in optimal:
                preset_key = "smooth"

        result = optimal.get(preset_key, {})
        return dict(result) if result else {}

    def _calculate_confidence(
        self,
        intensity: IntensityAnalysisResult,
        shape: ShapeAnalysisResult,
        modality: Optional[str],
    ) -> float:
        """Calculate overall confidence in the recommendation."""
        confidence = 0.5  # Start at middle

        # Higher separation = higher confidence (strong factor)
        confidence += intensity.separation_score * 0.35

        # Lower roughness = higher confidence (moderate factor)
        confidence += (1.0 - shape.boundary_roughness) * 0.1

        # Lower overlap = higher confidence
        confidence -= (intensity.overlap_percentage / 100.0) * 0.1

        # Known modality increases confidence
        if modality:
            if modality in self.MODALITY_HINTS:
                confidence += 0.1
                hints = self.MODALITY_HINTS[modality]
                if hints.get("low_confidence"):
                    confidence -= 0.15
                if hints.get("noise_level") == "very_high":
                    confidence -= 0.1

        # 3D structure increases confidence (more data)
        if shape.is_3d_structure:
            confidence += 0.05

        return float(max(0.1, min(0.95, confidence)))

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
            reasons.append("Well-separated intensity distributions favor this approach")
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
        elif intensity.separation_score > 0.8:
            reasons.append("Good intensity contrast allows lower sensitivity")

        if shape.is_small_structure():
            reasons.append("Higher precision for small structure")

        return ". ".join(reasons) if reasons else "Default sensitivity based on analysis"

    def _get_radius_reason(self, shape: ShapeAnalysisResult) -> str:
        """Generate explanation for brush radius choice."""
        return (
            f"Based on estimated structure diameter of {shape.estimated_diameter_mm:.1f}mm "
            f"(approximately 1/4 of structure size for optimal control)"
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
                "boundaries may require manual refinement"
            )

        # High overlap warning
        if intensity.overlap_percentage > 50:
            warnings.append(
                f"High intensity overlap ({intensity.overlap_percentage:.0f}%) - "
                "consider adjusting threshold manually after initial segmentation"
            )

        # Very large structure warning
        if shape.is_large_structure(80.0):
            warnings.append(
                "Large structure may benefit from multiple brush strokes "
                "or a larger brush radius"
            )

        # Very small structure warning
        if shape.is_small_structure(5.0):
            warnings.append(
                "Very small structure - consider using higher precision algorithm "
                "and smaller brush radius"
            )

        # Noisy modality warning
        if modality and self.MODALITY_HINTS.get(modality, {}).get("noise_level") in (
            "high",
            "very_high",
        ):
            warnings.append(f"{modality} images may be noisy - results may need refinement")

        # High roughness warning
        if shape.boundary_roughness > 0.7:
            warnings.append(
                "Highly irregular boundary detected - consider level set or "
                "random walker for best results"
            )

        return warnings

    def _get_alternative_algorithms(
        self, scores: dict[str, float], primary: str
    ) -> list[tuple[str, str]]:
        """Get alternative algorithm suggestions."""
        sorted_algos = sorted(
            [(algo, score) for algo, score in scores.items() if algo != primary],
            key=lambda x: x[1],
            reverse=True,
        )

        alternatives = []
        for algo, score in sorted_algos[:3]:
            # Only suggest if score is reasonably close to primary
            primary_score = scores.get(primary, 0)
            if score >= primary_score * 0.7:  # Within 30% of primary
                reason = self._get_alternative_reason(algo, primary)
                alternatives.append((algo, reason))

        return alternatives

    def _get_alternative_reason(self, alternative: str, primary: str) -> str:
        """Generate reason for an alternative algorithm."""
        _ = primary
        alt_info = self.ALGORITHM_INFO.get(alternative, {})

        if alt_info.get("speed") == "very_fast":
            return "Much faster option if speed is critical"

        if alt_info.get("precision") == "very_high":
            return "Higher precision for challenging boundaries"

        if alt_info.get("handles_irregular"):
            return "Better for irregular or complex boundaries"

        if alt_info.get("handles_homogeneous"):
            return "Good for uniform intensity regions"

        if alt_info.get("handles_noise"):
            return "More robust to image noise"

        return str(alt_info.get("description", "Alternative approach"))

    def _find_matching_preset(
        self,
        intensity: IntensityAnalysisResult,
        modality: Optional[str],
        structure_type: Optional[str],
    ) -> Optional[str]:
        """Find the best matching preset for the analysis results."""
        best_match = None
        best_score = 0

        for mapping in self.PRESET_MAPPINGS:
            score = 0

            # Check modality match
            if mapping.get("modality"):
                if modality == mapping["modality"]:
                    score += 2
                else:
                    continue  # Skip if modality specified but doesn't match

            # Check structure match
            if mapping.get("structure"):
                if structure_type == mapping["structure"]:
                    score += 2
                elif structure_type is None:
                    score += 1  # Partial match if no structure specified
                else:
                    continue  # Skip if structure specified but doesn't match

            # Check separation threshold
            if mapping.get("separation_min"):
                if intensity.separation_score >= mapping["separation_min"]:
                    score += 1

            if score > best_score:
                best_score = score
                best_match = mapping["preset_id"]

        return best_match
