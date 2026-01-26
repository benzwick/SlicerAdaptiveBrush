"""Parameter Wizard - Main coordinator for the Quick Select Parameters Wizard.

This module coordinates the wizard workflow, connecting UI, sampling,
analysis, and recommendation components.
"""

import logging
from typing import Any, Optional

import numpy as np
from EmbeddedWizardUI import EmbeddedWizardPanel
from ParameterRecommender import ParameterRecommender
from WizardAnalyzer import WizardAnalyzer
from WizardDataStructures import WizardRecommendation, WizardSamples
from WizardSampler import WizardSampler

logger = logging.getLogger(__name__)


class ParameterWizard:
    """Coordinates the Quick Select Parameters wizard workflow.

    This class manages the interaction between:
    - The embedded wizard UI (EmbeddedWizardPanel)
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

        self._embedded_panel: Optional[EmbeddedWizardPanel] = None
        self._fg_sampler: Optional[WizardSampler] = None
        self._bg_sampler: Optional[WizardSampler] = None
        self._boundary_sampler: Optional[WizardSampler] = None
        self._current_sampler: Optional[WizardSampler] = None
        self._hidden_widgets: list = []
        self._left_button_down: bool = False  # Track button state internally

    def start(self) -> None:
        """Launch the wizard by embedding it in the options frame."""
        try:
            # Get source volume
            volume_node = self._get_source_volume()
            if not volume_node:
                self._show_error("No source volume found. Please select a volume first.")
                return

            self.samples = WizardSamples()
            self.samples.volume_node = volume_node

            # Set wizard active flag and store reference for event forwarding
            self.effect._wizardActive = True
            self.effect._activeWizard = self
            logger.info("Wizard started - main effect event handling disabled")

            # Create samplers
            self._fg_sampler = WizardSampler(volume_node, self._on_foreground_sampled)
            self._bg_sampler = WizardSampler(volume_node, self._on_background_sampled)
            self._boundary_sampler = WizardSampler(volume_node, self._on_boundary_traced)

            # Hide the normal effect controls
            self._hide_normal_controls()

            # Create and show embedded wizard panel
            self._embedded_panel = EmbeddedWizardPanel(
                on_page_changed=self._on_page_changed,
                on_finished=self._on_wizard_finished,
            )
            wizard_widget = self._embedded_panel.build_ui()

            if wizard_widget:
                # Add wizard panel to the options frame
                self.effect.scriptedEffect.addOptionsWidget(wizard_widget)
                logger.info("Wizard panel embedded in options frame")

            # Start foreground sampling
            self._activate_foreground_sampling()

        except Exception as e:
            logger.exception("Failed to start wizard")
            self._cleanup()
            self._show_error(f"Failed to start wizard: {e}")

    def _hide_normal_controls(self) -> None:
        """Hide the normal effect controls while wizard is active."""
        try:
            # Get all child widgets of the options frame and hide them
            # We'll store references to restore them later
            self._hidden_widgets = []

            # Hide collapsible sections by storing their visibility
            widgets_to_hide = [
                getattr(self.effect, "brushCollapsible", None),
                getattr(self.effect, "algorithmParamsCollapsible", None),
                getattr(self.effect, "samplingCollapsible", None),
                getattr(self.effect, "displayCollapsible", None),
            ]

            for widget in widgets_to_hide:
                if widget is not None:
                    self._hidden_widgets.append((widget, widget.visible))
                    widget.hide()

            logger.debug(f"Hidden {len(self._hidden_widgets)} control sections")

        except Exception as e:
            logger.warning(f"Error hiding normal controls: {e}")

    def _restore_normal_controls(self) -> None:
        """Restore the normal effect controls."""
        try:
            for widget, was_visible in self._hidden_widgets:
                if was_visible:
                    widget.show()
            self._hidden_widgets = []
            logger.debug("Restored normal controls")
        except Exception as e:
            logger.warning(f"Error restoring controls: {e}")

    def _get_source_volume(self) -> Any:
        """Get the source volume node from the effect."""
        try:
            parameterSetNode = self.effect.scriptedEffect.parameterSetNode()
            if parameterSetNode:
                return parameterSetNode.GetSourceVolumeNode()
        except Exception as e:
            logger.warning(f"Failed to get source volume: {e}")
        return None

    def _on_page_changed(self, page_id: int) -> None:
        """Handle wizard page changes."""
        # Deactivate current sampler
        if self._current_sampler:
            self._current_sampler.deactivate()
            self._current_sampler = None

        # Activate appropriate sampler based on page
        if page_id == EmbeddedWizardPanel.STEP_FOREGROUND:
            self._activate_foreground_sampling()
        elif page_id == EmbeddedWizardPanel.STEP_BACKGROUND:
            self._activate_background_sampling()
        elif page_id == EmbeddedWizardPanel.STEP_BOUNDARY:
            self._activate_boundary_sampling()
        elif page_id == EmbeddedWizardPanel.STEP_QUESTIONS:
            # Questions page - no sampling
            pass
        elif page_id == EmbeddedWizardPanel.STEP_RESULTS:
            # Results page - generate recommendation
            self._generate_and_display_recommendation()

    def _activate_foreground_sampling(self) -> None:
        """Activate foreground sampling mode."""
        if not self._fg_sampler:
            return

        view_widget = self._get_active_slice_widget()
        if view_widget:
            self._fg_sampler.activate(view_widget, sample_type="foreground")
            self._current_sampler = self._fg_sampler
            logger.debug("Foreground sampling activated")

    def _activate_background_sampling(self) -> None:
        """Activate background sampling mode."""
        if not self._bg_sampler:
            return

        view_widget = self._get_active_slice_widget()
        if view_widget:
            self._bg_sampler.activate(view_widget, sample_type="background")
            self._current_sampler = self._bg_sampler
            logger.debug("Background sampling activated")

    def _activate_boundary_sampling(self) -> None:
        """Activate boundary tracing mode."""
        if not self._boundary_sampler:
            return

        view_widget = self._get_active_slice_widget()
        if view_widget:
            self._boundary_sampler.activate(view_widget, sample_type="boundary")
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

    def handle_interaction_event(
        self, callerInteractor: Any, eventId: int, viewWidget: Any
    ) -> bool:
        """Handle mouse interaction events forwarded from the main effect.

        Args:
            callerInteractor: VTK interactor that triggered the event.
            eventId: VTK event ID.
            viewWidget: The view widget where the event occurred.

        Returns:
            True if the event was handled (consumed), False otherwise.
        """
        import vtk

        if not self._current_sampler or not self._current_sampler.is_active:
            return False

        # Only handle events in slice views
        if viewWidget.className() != "qMRMLSliceWidget":
            return False

        # Only handle left-button events - let everything else (zoom, pan) pass through
        if eventId == vtk.vtkCommand.LeftButtonPressEvent:
            self._left_button_down = True
            self._current_sampler._view_widget = viewWidget
            self._current_sampler.process_event(callerInteractor, "LeftButtonPressEvent")
            return True  # Consume to prevent default paint behavior

        elif eventId == vtk.vtkCommand.LeftButtonReleaseEvent:
            self._left_button_down = False
            self._current_sampler._view_widget = viewWidget
            self._current_sampler.process_event(callerInteractor, "LeftButtonReleaseEvent")
            return True

        elif eventId == vtk.vtkCommand.MouseMoveEvent:
            # Only consume mouse move if we're actively sampling (left button down)
            if self._left_button_down:
                self._current_sampler._view_widget = viewWidget
                self._current_sampler.process_event(callerInteractor, "MouseMoveEvent")
                return True
            # Let mouse move pass through when not sampling (for hover effects, etc.)
            return False

        # Let all other events (wheel, right-click, etc.) pass through for zoom/pan
        return False

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
        if self._embedded_panel:
            self._embedded_panel.update_sample_count(
                "foreground", len(self.samples.foreground_points)
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

        if self._embedded_panel:
            self._embedded_panel.update_sample_count(
                "background", len(self.samples.background_points)
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

        if self._embedded_panel:
            self._embedded_panel.update_sample_count("boundary", len(self.samples.boundary_points))

        logger.debug(f"Boundary points: {len(self.samples.boundary_points)}")

    def _on_boundary_traced(self, points: list, intensities: np.ndarray) -> None:
        """Internal callback wrapper (ignores intensities for boundary)."""
        self.on_boundary_traced(points)

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
        if self._embedded_panel:
            answers = self._embedded_panel.get_answers()

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

            if self._embedded_panel:
                self._embedded_panel.set_recommendation(recommendation)

            logger.info(
                f"Generated recommendation: algorithm={recommendation.algorithm}, "
                f"confidence={recommendation.confidence:.2f}"
            )

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

    def _on_wizard_finished(self, accepted: bool) -> None:
        """Handle wizard completion.

        Args:
            accepted: True if user clicked Apply, False if cancelled.
        """
        logger.info(f"Wizard finished: accepted={accepted}")

        # Apply recommendation if accepted
        if accepted and self._embedded_panel:
            recommendation = self._embedded_panel.get_recommendation()
            if recommendation:
                self.apply_recommendation(recommendation)

        # Clean up
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up wizard state and restore normal controls."""
        # Deactivate samplers
        if self._fg_sampler:
            self._fg_sampler.deactivate()
        if self._bg_sampler:
            self._bg_sampler.deactivate()
        if self._boundary_sampler:
            self._boundary_sampler.deactivate()

        self._current_sampler = None

        # Hide and delete wizard panel
        if self._embedded_panel:
            self._embedded_panel.hide()
            if self._embedded_panel.container:
                self._embedded_panel.container.deleteLater()
            self._embedded_panel = None

        # Restore normal controls
        self._restore_normal_controls()

        # Re-enable main effect event handling
        self.effect._wizardActive = False
        self.effect._activeWizard = None
        logger.info("Wizard closed - main effect event handling re-enabled")

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
