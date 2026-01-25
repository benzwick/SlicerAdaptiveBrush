"""
Recipe: brain_tumor_combined

Segment brain tumor using Adaptive Brush + cleanup with other effects.
Demonstrates combining multiple Slicer effects in one recipe.
"""

sample_data = "MRBrainTumor1"
segment_name = "Tumor_Combined"


def run(effect):
    """Segment tumor with adaptive brush then clean up."""
    import slicer

    # Step 1: Initial segmentation with Adaptive Brush
    effect.applyPreset("mri_t1gd_tumor")
    effect.brushRadiusMm = 20.0
    effect.paintAt(-5.31, 34.77, 20.83)
    effect.paintAt(-5.31, 25.12, 35.97)
    effect.paintAt(-5.31, 20.70, 22.17)

    # Step 2: Clean up with other effects
    editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor

    # Keep only largest island (remove small disconnected regions)
    editor.setActiveEffectByName("Islands")
    islands = editor.activeEffect()
    if islands:
        islands.setParameter("Operation", "KEEP_LARGEST_ISLAND")
        islands.self().onApply()

    # Smooth the boundaries
    editor.setActiveEffectByName("Smoothing")
    smoothing = editor.activeEffect()
    if smoothing:
        smoothing.setParameter("SmoothingMethod", "MEDIAN")
        smoothing.setParameter("KernelSizeMm", "2.0")
        smoothing.self().onApply()

    # Fill any internal holes
    editor.setActiveEffectByName("Smoothing")
    smoothing = editor.activeEffect()
    if smoothing:
        smoothing.setParameter("SmoothingMethod", "MORPHOLOGICAL_CLOSING")
        smoothing.setParameter("KernelSizeMm", "3.0")
        smoothing.self().onApply()
