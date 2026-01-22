"""Adaptive brush segment editor effect for 3D Slicer.

This module provides an adaptive brush effect that automatically segments
regions based on image intensity similarity, adapting to image features
(edges, boundaries) rather than using a fixed geometric shape.

This is a scripted module that registers a segment editor effect.
See https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
"""

import os

import qt
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
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
        self.parent.contributors = ["Ben Zwick"]
        self.parent.helpText = _(
            """
Adaptive brush segment editor effect that automatically segments regions
based on image intensity similarity.

The brush adapts to image features (edges, boundaries) rather than using
a fixed geometric shape, similar to tools in ITK-SNAP and ImFusion.

For more information, see the <a href="https://github.com/benzwick/SlicerAdaptiveBrush">
project documentation</a>.
"""
        )
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = _(
            """
This extension was developed as an open-source implementation of adaptive
brush segmentation tools. Inspired by ITK-SNAP and ImFusion Labels.
"""
        )

        # Show module in selector so users can find usage instructions
        self.parent.hidden = False

        # Register effect on startup
        slicer.app.connect("startupCompleted()", self.registerEditorEffect)

    def registerEditorEffect(self):
        """Register the Adaptive Brush effect with Segment Editor."""
        import qSlicerSegmentationsEditorEffectsPythonQt as qSlicerSegmentationsEditorEffects

        effectFilename = os.path.join(
            os.path.dirname(__file__), self.__class__.__name__ + "Lib/SegmentEditorEffect.py"
        )
        clonedEffect = qSlicerSegmentationsEditorEffects.qSlicerSegmentEditorScriptedEffect(None)
        clonedEffect.setPythonSource(effectFilename.replace("\\", "/"))
        clonedEffect.self().register()


class SegmentEditorAdaptiveBrushWidget(ScriptedLoadableModuleWidget):
    """Widget for the module (minimal - effect has its own UI)."""

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Get the icon path for display
        iconPath = os.path.join(
            os.path.dirname(__file__),
            self.moduleName + "Lib",
            "SegmentEditorEffect.png",
        )

        # Informational text with icon
        infoText = _(
            """
<h3>Adaptive Brush Effect</h3>
<p>This module provides an <b>Adaptive Brush</b> effect for the Segment Editor.</p>

<h4>How to Use</h4>
<ol>
<li>Open the <b>Segment Editor</b> module</li>
<li>Look for the Adaptive Brush icon in the effects toolbar:<br/>
    <img src="{icon_path}" width="32" height="32" style="vertical-align: middle;"/>
    (paintbrush with adaptive boundary)</li>
<li>Click and drag on the image to paint with the adaptive brush</li>
</ol>

<p>The brush automatically segments regions based on intensity similarity,
stopping at edges and boundaries.</p>

<h4>Effect Not Visible?</h4>
<p>If you don't see the Adaptive Brush effect in the Segment Editor:</p>
<ul>
<li>Restart 3D Slicer after installing the extension</li>
<li>Check that the extension is enabled in <b>Edit → Application Settings → Modules</b></li>
<li>Look in the "Additional effects" dropdown if your toolbar is narrow</li>
<li>Verify installation in <b>Extension Manager</b></li>
</ul>

<h4>Links</h4>
<ul>
<li><a href="https://github.com/benzwick/SlicerAdaptiveBrush">GitHub Repository</a>
    - Source code, issues, and contributions</li>
<li><a href="https://github.com/benzwick/SlicerAdaptiveBrush#readme">Documentation</a>
    - Usage guide and examples</li>
</ul>
"""
        ).format(icon_path=iconPath.replace("\\", "/"))

        textWidget = qt.QLabel(infoText)
        textWidget.setWordWrap(True)
        textWidget.setOpenExternalLinks(True)
        self.layout.addWidget(textWidget)

        # Add spacer
        self.layout.addStretch(1)


class SegmentEditorAdaptiveBrushTest(ScriptedLoadableModuleTest):
    """Test case for the module."""

    def setUp(self):
        """Reset the state before each test."""
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run all tests."""
        self.setUp()
        self.test_EffectRegistration()
        self.test_BasicSegmentation()

    def test_EffectRegistration(self):
        """Test that the effect is registered with Segment Editor."""
        self.delayDisplay("Testing effect registration...")

        # Check if effect is available
        import qSlicerSegmentationsEditorEffectsPythonQt as qSlicerSegmentationsEditorEffects

        effectFactory = (
            qSlicerSegmentationsEditorEffects.qSlicerSegmentEditorEffectFactory.instance()
        )
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
