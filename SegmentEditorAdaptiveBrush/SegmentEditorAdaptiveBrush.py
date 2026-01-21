"""SegmentEditorAdaptiveBrush - Adaptive brush segment editor effect for 3D Slicer.

This module provides an adaptive brush effect that automatically segments
regions based on image intensity similarity, adapting to image features
(edges, boundaries) rather than using a fixed geometric shape.

License: Apache 2.0
"""

import logging
import os

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


class SegmentEditorAdaptiveBrush(ScriptedLoadableModule):
    """Main module class for SegmentEditorAdaptiveBrush.

    This module registers the Adaptive Brush effect with the Segment Editor.
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Segment Editor Adaptive Brush")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = ["Segmentations"]
        self.parent.contributors = ["SlicerAdaptiveBrush Contributors"]
        self.parent.helpText = _(
            """
Adaptive brush segment editor effect that automatically segments regions
based on image intensity similarity.

The brush adapts to image features (edges, boundaries) rather than using
a fixed geometric shape, similar to tools in ITK-SNAP and ImFusion.

For more information, see the <a href="https://github.com/your-org/SlicerAdaptiveBrush">
project documentation</a>.
"""
        )
        self.parent.acknowledgementText = _(
            """
This extension was developed as an open-source implementation of adaptive
brush segmentation tools. Inspired by ITK-SNAP and ImFusion Labels.
"""
        )

        # Don't show this module in the module selector - it just registers the effect
        self.parent.hidden = True

        # Register effect on startup
        slicer.app.connect("startupCompleted()", self.registerEffect)

    def registerEffect(self):
        """Register the Adaptive Brush effect with Segment Editor."""
        try:
            import qSlicerSegmentationsEditorEffectsPythonQt as effects

            effectPath = os.path.join(
                os.path.dirname(__file__),
                "SegmentEditorAdaptiveBrushLib",
                "SegmentEditorEffect.py",
            )

            scriptedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
            scriptedEffect.setPythonSource(effectPath)

            # Register with the effect factory
            effectFactory = effects.qSlicerSegmentEditorEffectFactory.instance()
            effectFactory.registerEffect(scriptedEffect)

            logging.info("Adaptive Brush effect registered successfully")

        except Exception as e:
            logging.error(f"Failed to register Adaptive Brush effect: {e}")
            import traceback

            traceback.print_exc()


class SegmentEditorAdaptiveBrushWidget(ScriptedLoadableModuleWidget):
    """Widget for the module (minimal - effect has its own UI)."""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Informational text
        infoLabel = slicer.qMRMLWidget()
        infoText = _(
            """
<h3>Adaptive Brush Effect</h3>
<p>This module provides an Adaptive Brush effect for the Segment Editor.</p>
<p>To use it:</p>
<ol>
<li>Open the <b>Segment Editor</b> module</li>
<li>Select the <b>Adaptive Brush</b> effect from the effects toolbar</li>
<li>Click and drag on the image to paint with the adaptive brush</li>
</ol>
<p>The brush automatically segments regions based on intensity similarity,
stopping at edges and boundaries.</p>
"""
        )

        textWidget = qt.QLabel(infoText)
        textWidget.setWordWrap(True)
        self.layout.addWidget(textWidget)

        # Add spacer
        self.layout.addStretch(1)


class SegmentEditorAdaptiveBrushLogic(ScriptedLoadableModuleLogic):
    """Logic class (minimal - main logic is in the effect)."""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)


class SegmentEditorAdaptiveBrushTest(ScriptedLoadableModuleTest):
    """Test case for the module."""

    def setUp(self):
        """Reset the state before each test."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run all tests."""
        self.setUp()
        self.test_EffectRegistration()
        self.test_BasicSegmentation()

    def test_EffectRegistration(self):
        """Test that the effect is registered with Segment Editor."""
        self.delayDisplay("Testing effect registration...")

        # Check if effect is available
        import qSlicerSegmentationsEditorEffectsPythonQt as effects

        effectFactory = effects.qSlicerSegmentEditorEffectFactory.instance()
        registeredEffects = effectFactory.registeredEffects()

        self.assertIn(
            "Adaptive Brush",
            registeredEffects,
            "Adaptive Brush effect should be registered",
        )

        self.delayDisplay("Effect registration test passed!")

    def test_BasicSegmentation(self):
        """Test basic segmentation functionality."""
        self.delayDisplay("Testing basic segmentation...")

        # Load sample data
        import SampleData

        volumeNode = SampleData.downloadSample("MRHead")
        self.assertIsNotNone(volumeNode, "Failed to load sample data")

        # Create segmentation
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.CreateDefaultDisplayNodes()
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)

        # Add a segment
        segmentId = segmentationNode.GetSegmentation().AddEmptySegment("TestSegment")

        # Get segment editor widget
        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        segmentEditorWidget.setMRMLSegmentEditorNode(
            slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        )
        segmentEditorWidget.setSegmentationNode(segmentationNode)
        segmentEditorWidget.setSourceVolumeNode(volumeNode)
        segmentEditorWidget.setCurrentSegmentID(segmentId)

        # Select Adaptive Brush effect
        segmentEditorWidget.setActiveEffectByName("Adaptive Brush")
        effect = segmentEditorWidget.activeEffect()

        self.assertIsNotNone(effect, "Failed to activate Adaptive Brush effect")

        self.delayDisplay("Basic segmentation test passed!")


# Import qt here to avoid issues if module is imported before Slicer is fully initialized
try:
    import qt
except ImportError:
    pass
