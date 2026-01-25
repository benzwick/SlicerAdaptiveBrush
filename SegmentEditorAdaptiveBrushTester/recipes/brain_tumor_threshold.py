"""
Recipe: brain_tumor_threshold

Segment brain tumor using Threshold effect only.
Demonstrates using standard Slicer effects in recipes.
"""

sample_data = "MRBrainTumor1"
segment_name = "Tumor_Threshold"


def run(effect):
    """Segment tumor using threshold-based approach."""
    import slicer

    # Get the segment editor widget
    editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor

    # Use Threshold effect
    editor.setActiveEffectByName("Threshold")
    threshold = editor.activeEffect()

    # Set threshold range for enhancing tumor (bright in T1+Gd)
    threshold.setParameter("MinimumThreshold", "100")
    threshold.setParameter("MaximumThreshold", "255")
    threshold.self().onApply()

    # Clean up with Islands - keep largest connected region
    editor.setActiveEffectByName("Islands")
    islands = editor.activeEffect()
    islands.setParameter("Operation", "KEEP_LARGEST_ISLAND")
    islands.self().onApply()

    # Smooth the result
    editor.setActiveEffectByName("Smoothing")
    smoothing = editor.activeEffect()
    smoothing.setParameter("SmoothingMethod", "MEDIAN")
    smoothing.setParameter("KernelSizeMm", "2.0")
    smoothing.self().onApply()
