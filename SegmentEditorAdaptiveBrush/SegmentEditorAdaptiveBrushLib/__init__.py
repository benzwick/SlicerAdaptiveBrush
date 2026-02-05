"""SegmentEditorAdaptiveBrushLib - Adaptive brush segment editor effect.

This library provides an adaptive brush effect for 3D Slicer's Segment Editor
that automatically segments regions based on image intensity similarity.
"""

# Note: SegmentEditorEffect is loaded directly by Slicer via setPythonSource,
# not imported as a package. This __init__.py is mainly for documentation
# and potential future use as a proper Python package.

__all__ = [
    "SegmentEditorEffect",
    "IntensityAnalyzer",
    "PerformanceCache",
    "DependencyManager",
    "dependency_manager",
    "WizardDataStructures",
    "WizardAnalyzer",
    "ParameterRecommender",
    "WizardSampler",
    "EmbeddedWizardUI",
    "ParameterWizard",
]
