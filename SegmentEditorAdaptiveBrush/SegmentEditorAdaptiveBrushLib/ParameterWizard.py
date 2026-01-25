"""Parameter Wizard - Main coordinator for the Quick Select Parameters Wizard.

This module coordinates the wizard workflow, connecting UI, sampling,
analysis, and recommendation components.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from ParameterRecommender import ParameterRecommender
from WizardAnalyzer import WizardAnalyzer
from WizardDataStructures import WizardRecommendation, WizardSamples
from WizardSampler import WizardSampler

if TYPE_CHECKING:
    from WizardUI import WizardPanel

logger = logging.getLogger(__name__)


class ParameterWizard:
    """Coordinates the Quick Select Parameters wizard workflow.

    This class manages the interaction between:
    - The wizard UI (WizardPanel)
    - Interactive sampling (WizardSampler)
    - Analysis (WizardAnalyzer)
    - Recommendations (ParameterRecommender)
    """

    def __init__(self, effect: Any):
        """Initialize the wizard.

        Args:
            effect: The SegmentEditorEffect instance.
        """
        self.effect = effect
        self.samples = WizardSamples()
        self.analyzer = WizardAnalyzer()
        self.recommender = ParameterRecommender()

        self._wizard_panel: Optional[WizardPanel] = None
        self._fg_sampler: Optional[WizardSampler] = None
        self._bg_sampler: Optional[WizardSampler] = None
        self._boundary_sampler: Optional[WizardSampler] = None
        self._current_sampler: Optional[WizardSampler] = None
        self._current_page = 0

    def start(self) -> None:
        """Launch the wizard dialog."""
        try:
            from WizardUI import WizardPanel

            # Get source volume
            volume_node = self._get_source_volume()
            if not volume_node:
                self._show_error("No source volume found. Please select a volume first.")
                return

            self.samples.volume_node = volume_node

            # Create samplers
            self._fg_sampler = WizardSampler(volume_node, self._on_foreground_sampled)
            self._bg_sampler = WizardSampler(volume_node, self._on_background_sampled)
            self._boundary_sampler = WizardSampler(volume_node, self._on_boundary_traced)

            # Create and show wizard panel
            self._wizard_panel = WizardPanel()
            self._connect_wizard_signals()
            self._wizard_panel.show()

            # Start foreground sampling
            self._activate_foreground_sampling()

        except Exception as e:
            logger.exception("Failed to start wizard")
            self._show_error(f"Failed to start wizard: {e}")

    def _get_source_volume(self) -> Any:
        """Get the source volume node from the effect."""
        try:
            parameterSetNode = self.effect.scriptedEffect.parameterSetNode()
            if parameterSetNode:
                return parameterSetNode.GetSourceVolumeNode()
        except Exception as e:
            logger.warning(f"Failed to get source volume: {e}")
        return None

    def _connect_wizard_signals(self) -> None:
        """Connect wizard panel signals."""
        if not self._wizard_panel:
            return

        try:
            self._wizard_panel.currentIdChanged.connect(self._on_page_changed)
            self._wizard_panel.finished.connect(self._on_wizard_finished)
        except Exception as e:
            logger.warning(f"Failed to connect wizard signals: {e}")

    def _on_page_changed(self, page_id: int) -> None:
        """Handle wizard page changes."""
        self._current_page = page_id

        # Deactivate current sampler
        if self._current_sampler:
            self._current_sampler.deactivate()
            self._current_sampler = None

        # Activate appropriate sampler based on page
        if page_id == 0:
            self._activate_foreground_sampling()
        elif page_id == 1:
            self._activate_background_sampling()
        elif page_id == 2:
            self._activate_boundary_sampling()
        elif page_id == 3:
            # Questions page - no sampling
            pass
        elif page_id == 4:
            # Results page - generate recommendation
            self._generate_and_display_recommendation()

    def _activate_foreground_sampling(self) -> None:
        """Activate foreground sampling mode."""
        if not self._fg_sampler:
            return

        view_widget = self._get_active_slice_widget()
        if view_widget:
            self._fg_sampler.activate(view_widget)
            self._current_sampler = self._fg_sampler
            logger.debug("Foreground sampling activated")

    def _activate_background_sampling(self) -> None:
        """Activate background sampling mode."""
        if not self._bg_sampler:
            return

        view_widget = self._get_active_slice_widget()
        if view_widget:
            self._bg_sampler.activate(view_widget)
            self._current_sampler = self._bg_sampler
            logger.debug("Background sampling activated")

    def _activate_boundary_sampling(self) -> None:
        """Activate boundary tracing mode."""
        if not self._boundary_sampler:
            return

        view_widget = self._get_active_slice_widget()
        if view_widget:
            self._boundary_sampler.activate(view_widget)
            self._current_sampler = self._boundary_sampler
            logger.debug("Boundary sampling activated")

    def _get_active_slice_widget(self) -> Any:
        """Get the currently active slice view widget."""
        try:
            import slicer

            layoutManager = slicer.app.layoutManager()
            if layoutManager:
                # Try to get the Red slice view (most common)
                sliceWidget = layoutManager.sliceWidget("Red")
                if sliceWidget:
                    return sliceWidget

                # Fall back to any visible slice view
                for name in ["Yellow", "Green"]:
                    sliceWidget = layoutManager.sliceWidget(name)
                    if sliceWidget:
                        return sliceWidget
        except Exception as e:
            logger.warning(f"Failed to get slice widget: {e}")

        return None

    def on_foreground_sampled(self, points: list, intensities: np.ndarray) -> None:
        """Called when foreground sampling completes a stroke.

        Args:
            points: List of (i, j, k) coordinates.
            intensities: Array of intensity values.
        """
        # Append to existing samples
        self.samples.foreground_points.extend(points)
        if self.samples.foreground_intensities is None:
            self.samples.foreground_intensities = intensities
        else:
            self.samples.foreground_intensities = np.concatenate(
                [self.samples.foreground_intensities, intensities]
            )

        # Update UI
        if self._wizard_panel:
            self._wizard_panel.foreground_page.update_sample_count(
                len(self.samples.foreground_points)
            )

        logger.debug(f"Foreground samples: {len(self.samples.foreground_points)}")

    def _on_foreground_sampled(self, points: list, intensities: np.ndarray) -> None:
        """Internal callback wrapper."""
        self.on_foreground_sampled(points, intensities)

    def on_background_sampled(self, points: list, intensities: np.ndarray) -> None:
        """Called when background sampling completes a stroke.

        Args:
            points: List of (i, j, k) coordinates.
            intensities: Array of intensity values.
        """
        self.samples.background_points.extend(points)
        if self.samples.background_intensities is None:
            self.samples.background_intensities = intensities
        else:
            self.samples.background_intensities = np.concatenate(
                [self.samples.background_intensities, intensities]
            )

        if self._wizard_panel:
            self._wizard_panel.background_page.update_sample_count(
                len(self.samples.background_points)
            )

        logger.debug(f"Background samples: {len(self.samples.background_points)}")

    def _on_background_sampled(self, points: list, intensities: np.ndarray) -> None:
        """Internal callback wrapper."""
        self.on_background_sampled(points, intensities)

    def on_boundary_traced(self, points: list) -> None:
        """Called when boundary tracing completes a stroke.

        Args:
            points: List of (i, j, k) coordinates along boundary.
        """
        self.samples.boundary_points.extend(points)

        if self._wizard_panel:
            self._wizard_panel.boundary_page.update_sample_count(len(self.samples.boundary_points))

        logger.debug(f"Boundary points: {len(self.samples.boundary_points)}")

    def _on_boundary_traced(self, points: list, intensities: np.ndarray) -> None:
        """Internal callback wrapper (ignores intensities for boundary)."""
        self.on_boundary_traced(points)

    def on_questions_answered(self, answers: dict) -> None:
        """Called when user answers modality/structure questions.

        Args:
            answers: Dictionary with 'modality', 'structure_type', 'priority'.
        """
        # Answers are retrieved from the questions page when generating recommendation
        pass

    def generate_recommendation(self) -> WizardRecommendation:
        """Generate final recommendation from all inputs.

        Returns:
            WizardRecommendation with algorithm and parameter suggestions.
        """
        # Analyze intensity distributions
        intensity_result = self.analyzer.analyze_intensities(self.samples)

        # Analyze shape characteristics
        spacing = self._get_volume_spacing()
        shape_result = self.analyzer.analyze_shape(self.samples, spacing_mm=spacing)

        # Get user answers
        answers = {}
        if self._wizard_panel:
            answers = self._wizard_panel.get_questions_answers()

        # Generate recommendation
        recommendation = self.recommender.recommend(
            intensity_result=intensity_result,
            shape_result=shape_result,
            modality=answers.get("modality"),
            structure_type=answers.get("structure_type"),
            priority=answers.get("priority", "balanced"),
        )

        return recommendation

    def _generate_and_display_recommendation(self) -> None:
        """Generate recommendation and display in results page."""
        try:
            recommendation = self.generate_recommendation()

            if self._wizard_panel:
                self._wizard_panel.set_recommendation(recommendation)

        except Exception as e:
            logger.exception("Failed to generate recommendation")
            self._show_error(f"Failed to generate recommendation: {e}")

    def _get_volume_spacing(self) -> tuple[float, float, float]:
        """Get voxel spacing from the source volume."""
        try:
            if self.samples.volume_node:
                spacing = self.samples.volume_node.GetSpacing()
                return (spacing[0], spacing[1], spacing[2])
        except Exception:
            pass
        return (1.0, 1.0, 1.0)

    def apply_recommendation(self, recommendation: WizardRecommendation) -> None:
        """Apply recommended parameters to the effect.

        Args:
            recommendation: The recommendation to apply.
        """
        try:
            # Apply algorithm
            self.effect.setAlgorithm(recommendation.algorithm)

            # Apply brush radius
            self.effect.setRadiusMm(recommendation.brush_radius_mm)

            # Apply edge sensitivity
            self.effect.setEdgeSensitivity(recommendation.edge_sensitivity)

            # Apply thresholds if available and applicable
            if recommendation.has_threshold_suggestion():
                if hasattr(self.effect, "setThresholdRange"):
                    self.effect.setThresholdRange(
                        recommendation.threshold_lower,
                        recommendation.threshold_upper,
                    )

            logger.info(
                f"Applied wizard recommendation: algorithm={recommendation.algorithm}, "
                f"radius={recommendation.brush_radius_mm}mm, "
                f"sensitivity={recommendation.edge_sensitivity}%"
            )

        except Exception as e:
            logger.exception("Failed to apply recommendation")
            self._show_error(f"Failed to apply parameters: {e}")

    def _on_wizard_finished(self, result: int) -> None:
        """Handle wizard dialog finished.

        Args:
            result: Dialog result (1 = accepted, 0 = rejected).
        """
        # Clean up samplers
        if self._fg_sampler:
            self._fg_sampler.deactivate()
        if self._bg_sampler:
            self._bg_sampler.deactivate()
        if self._boundary_sampler:
            self._boundary_sampler.deactivate()

        self._current_sampler = None

        # Apply if accepted
        if result == 1 and self._wizard_panel is not None:  # QDialog.Accepted
            try:
                recommendation = self._wizard_panel.results_page.get_recommendation()
                if recommendation:
                    self.apply_recommendation(recommendation)
            except Exception:
                logger.exception("Failed to apply wizard results")

        self._wizard_panel = None

    def _show_error(self, message: str) -> None:
        """Show error message to user."""
        try:
            import slicer

            slicer.util.errorDisplay(message, windowTitle="Wizard Error")
        except Exception:
            logger.error(message)

    def clear_samples(self) -> None:
        """Clear all collected samples."""
        self.samples.clear_all()

        if self._fg_sampler:
            self._fg_sampler.clear()
        if self._bg_sampler:
            self._bg_sampler.clear()
        if self._boundary_sampler:
            self._boundary_sampler.clear()
