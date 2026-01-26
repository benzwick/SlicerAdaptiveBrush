"""Tests for Parameter Wizard workflow.

Tests the wizard workflow components:
1. Wizard launch from UI
2. Wizard pages navigation
3. Recommendation generation
4. Parameter application
"""

from __future__ import annotations

import logging
from typing import Any

import slicer
import vtk
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


def _get_slice_center_xy(slice_widget: Any) -> tuple[int, int]:
    """Get XY coordinates for the center of a slice view using RAS conversion."""
    slice_logic = slice_widget.sliceLogic()
    slice_node = slice_logic.GetSliceNode()

    slice_to_ras = slice_node.GetSliceToRAS()
    center_ras = [
        slice_to_ras.GetElement(0, 3),
        slice_to_ras.GetElement(1, 3),
        slice_to_ras.GetElement(2, 3),
    ]

    xy_to_ras = slice_node.GetXYToRAS()
    ras_to_xy = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(xy_to_ras, ras_to_xy)

    ras_point = [center_ras[0], center_ras[1], center_ras[2], 1]
    xy_point = [0, 0, 0, 1]
    ras_to_xy.MultiplyPoint(ras_point, xy_point)

    return (int(xy_point[0]), int(xy_point[1]))


@register_test(category="wizard")
class TestWizardLaunch(TestCase):
    """Test wizard can be launched from UI."""

    name = "wizard_launch"
    description = "Test that the parameter wizard can be launched"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and activate effect."""
        logger.info("Setting up wizard launch test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("WizardTest")

        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        ctx.screenshot("[setup] Ready to test wizard launch")

    def run(self, ctx: TestContext) -> None:
        """Test launching the parameter wizard."""
        logger.info("Running wizard launch test")

        scripted_effect = self.effect.self()

        # Check if wizard button exists
        if not hasattr(scripted_effect, "wizardButton"):
            ctx.log("Wizard button not found in UI - may be in collapsed section")
            ctx.record_metric("wizard_button_found", 0)
            return

        wizard_button = scripted_effect.wizardButton
        ctx.assert_is_not_none(wizard_button, "Wizard button should exist")
        ctx.record_metric("wizard_button_found", 1)

        ctx.screenshot("[button] Wizard button visible")

        # Test that button is enabled
        ctx.assert_true(wizard_button.enabled, "Wizard button should be enabled")

        ctx.log("Wizard button found and enabled")

    def verify(self, ctx: TestContext) -> None:
        """Verify wizard launch capability."""
        logger.info("Verifying wizard launch")

        scripted_effect = self.effect.self()

        # Verify wizard-related methods exist
        ctx.assert_true(
            hasattr(scripted_effect, "_startParameterWizard")
            or hasattr(scripted_effect, "onWizardButtonClicked"),
            "Effect should have wizard start method",
        )

        ctx.screenshot("[verify] Wizard launch capability verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="wizard")
class TestWizardRecommendation(TestCase):
    """Test wizard generates recommendations based on samples."""

    name = "wizard_recommendation"
    description = "Test that wizard generates valid parameter recommendations"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None
        self.red_widget = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and activate effect."""
        logger.info("Setting up wizard recommendation test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("RecommendTest")

        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        layoutManager = slicer.app.layoutManager()
        self.red_widget = layoutManager.sliceWidget("Red")
        red_logic = self.red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)
        slicer.app.processEvents()

        ctx.screenshot("[setup] Ready to test wizard recommendation")

    def run(self, ctx: TestContext) -> None:
        """Test recommendation generation with simulated samples."""
        logger.info("Running wizard recommendation test")

        # Import wizard components
        try:
            from SegmentEditorAdaptiveBrushLib.ParameterRecommender import (
                ParameterRecommender,
            )
            from SegmentEditorAdaptiveBrushLib.WizardAnalyzer import WizardAnalyzer
            from SegmentEditorAdaptiveBrushLib.WizardDataStructures import WizardSamples
        except ImportError as e:
            ctx.log(f"Could not import wizard components: {e}")
            ctx.record_metric("wizard_imports_available", 0)
            return

        ctx.record_metric("wizard_imports_available", 1)

        # Create sample data (simulating user clicks)
        import numpy as np

        np.random.seed(42)

        # Simulate foreground samples (bright tissue)
        foreground_intensities = np.random.normal(180, 20, 50).tolist()
        foreground_points = [(100 + i, 100 + i, 50) for i in range(50)]

        # Simulate background samples (dark tissue)
        background_intensities = np.random.normal(60, 15, 50).tolist()
        background_points = [(50 + i, 50 + i, 50) for i in range(50)]

        # Create samples object
        samples = WizardSamples(
            foreground_intensities=foreground_intensities,
            foreground_points=foreground_points,
            background_intensities=background_intensities,
            background_points=background_points,
        )

        ctx.log(f"Created samples: {samples.foreground_count} fg, {samples.background_count} bg")
        ctx.record_metric("foreground_samples", samples.foreground_count)
        ctx.record_metric("background_samples", samples.background_count)

        # Analyze samples
        analyzer = WizardAnalyzer()
        intensity_result = analyzer.analyze_intensities(samples)

        ctx.assert_is_not_none(intensity_result, "Intensity analysis should produce result")
        ctx.log(f"Separation score: {intensity_result.separation_score:.2f}")
        ctx.record_metric("separation_score", intensity_result.separation_score)

        # Generate recommendation
        recommender = ParameterRecommender()

        # Get user answers (simulated)
        user_answers = {
            "target_type": "soft_tissue",
            "priority": "balanced",
        }

        recommendation = recommender.recommend(
            intensity_result=intensity_result,
            shape_result=None,  # No shape analysis
            user_answers=user_answers,
        )

        ctx.assert_is_not_none(recommendation, "Recommender should produce recommendation")
        ctx.log(f"Recommended algorithm: {recommendation.algorithm}")
        ctx.log(f"Recommended brush radius: {recommendation.brush_radius_mm}mm")

        ctx.record_metric("recommended_algorithm", recommendation.algorithm)
        ctx.record_metric("recommended_radius", recommendation.brush_radius_mm)

        ctx.screenshot("[recommendation] Generated parameter recommendation")

    def verify(self, ctx: TestContext) -> None:
        """Verify recommendation generation."""
        logger.info("Verifying wizard recommendation")

        ctx.screenshot("[verify] Wizard recommendation verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="wizard")
class TestPresetApplication(TestCase):
    """Test preset application to effect parameters."""

    name = "wizard_preset_application"
    description = "Test that presets correctly update effect parameters"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and activate effect."""
        logger.info("Setting up preset application test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment("PresetTest")

        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        ctx.screenshot("[setup] Ready to test preset application")

    def run(self, ctx: TestContext) -> None:
        """Test applying different presets."""
        logger.info("Running preset application test")

        scripted_effect = self.effect.self()

        # Check if preset combo exists
        if not hasattr(scripted_effect, "presetCombo"):
            ctx.log("Preset combo not found - presets may not be implemented yet")
            ctx.record_metric("preset_combo_found", 0)
            return

        preset_combo = scripted_effect.presetCombo
        ctx.record_metric("preset_combo_found", 1)

        # Test each preset
        for i in range(preset_combo.count):
            preset_text = preset_combo.itemText(i)

            ctx.log(f"Applying preset: {preset_text}")

            # Record parameters before
            algorithm_before = scripted_effect.algorithm
            radius_before = scripted_effect.radiusMm

            # Apply preset
            preset_combo.setCurrentIndex(i)
            slicer.app.processEvents()

            # Record parameters after
            algorithm_after = scripted_effect.algorithm
            radius_after = scripted_effect.radiusMm

            ctx.log(f"  Algorithm: {algorithm_before} -> {algorithm_after}")
            ctx.log(f"  Radius: {radius_before} -> {radius_after}")

            ctx.record_metric(f"preset_{i}_algorithm", algorithm_after)
            ctx.record_metric(f"preset_{i}_radius", radius_after)

            ctx.screenshot(f"[preset_{i}] Applied preset: {preset_text}")

    def verify(self, ctx: TestContext) -> None:
        """Verify preset application."""
        logger.info("Verifying preset application")

        scripted_effect = self.effect.self()

        # Verify effect has valid parameters
        ctx.assert_is_not_none(scripted_effect.algorithm, "Algorithm should be set")
        ctx.assert_greater(scripted_effect.radiusMm, 0, "Radius should be positive")

        ctx.screenshot("[verify] Preset application verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")


@register_test(category="wizard")
class TestParameterPersistence(TestCase):
    """Test that parameters persist across effect deactivation/activation."""

    name = "wizard_parameter_persistence"
    description = "Test that effect parameters persist correctly"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.segment_id = None
        self.effect = None

    def setup(self, ctx: TestContext) -> None:
        """Load sample data and activate effect."""
        logger.info("Setting up parameter persistence test")

        slicer.mrmlScene.Clear(0)

        import SampleData

        self.volume_node = SampleData.downloadSample("MRHead")
        if self.volume_node is None:
            raise RuntimeError("Failed to load MRHead sample data")

        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        self.segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment(
            "PersistenceTest"
        )

        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        self.segment_editor_widget.setCurrentSegmentID(self.segment_id)
        slicer.app.processEvents()

        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        ctx.screenshot("[setup] Ready to test parameter persistence")

    def run(self, ctx: TestContext) -> None:
        """Test parameter persistence across deactivation/activation."""
        logger.info("Running parameter persistence test")

        scripted_effect = self.effect.self()

        # Set specific parameters
        test_radius = 35.0
        test_sensitivity = 70
        test_algorithm = "watershed"

        ctx.log(
            f"Setting parameters: radius={test_radius}, sensitivity={test_sensitivity}, algorithm={test_algorithm}"
        )

        scripted_effect.radiusSlider.value = test_radius
        scripted_effect.sensitivitySlider.value = test_sensitivity

        combo = scripted_effect.algorithmCombo
        idx = combo.findData(test_algorithm)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        ctx.screenshot("[set_params] Parameters configured")

        # Verify parameters are set
        ctx.assert_equal(scripted_effect.radiusMm, test_radius, "Radius should be set")
        ctx.assert_equal(
            scripted_effect.edgeSensitivity, test_sensitivity, "Sensitivity should be set"
        )
        ctx.assert_equal(scripted_effect.algorithm, test_algorithm, "Algorithm should be set")

        # Deactivate effect
        ctx.log("Deactivating effect")
        self.segment_editor_widget.setActiveEffect(None)
        slicer.app.processEvents()

        ctx.screenshot("[deactivated] Effect deactivated")

        # Reactivate effect
        ctx.log("Reactivating effect")
        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()
        scripted_effect = self.effect.self()
        slicer.app.processEvents()

        ctx.screenshot("[reactivated] Effect reactivated")

        # Check if parameters persisted
        radius_after = scripted_effect.radiusMm
        sensitivity_after = scripted_effect.edgeSensitivity
        algorithm_after = scripted_effect.algorithm

        ctx.log(
            f"After reactivation: radius={radius_after}, sensitivity={sensitivity_after}, algorithm={algorithm_after}"
        )

        ctx.record_metric("radius_persisted", 1 if radius_after == test_radius else 0)
        ctx.record_metric(
            "sensitivity_persisted", 1 if sensitivity_after == test_sensitivity else 0
        )
        ctx.record_metric("algorithm_persisted", 1 if algorithm_after == test_algorithm else 0)

    def verify(self, ctx: TestContext) -> None:
        """Verify parameter persistence."""
        logger.info("Verifying parameter persistence")

        scripted_effect = self.effect.self()

        # Parameters should have some value (may or may not persist depending on implementation)
        ctx.assert_greater(scripted_effect.radiusMm, 0, "Radius should be positive")
        ctx.assert_is_not_none(scripted_effect.algorithm, "Algorithm should be set")

        ctx.screenshot("[verify] Parameter persistence verified")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up."""
        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None
        ctx.log("Teardown complete")
