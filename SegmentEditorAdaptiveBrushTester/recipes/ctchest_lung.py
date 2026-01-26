"""
Recipe: ctchest_lung

Segmentation of lung parenchyma in CTChest sample data.
CTChest is a thoracic CT scan.
Lung tissue appears very dark (air-filled) with intensity around -800 to -900 HU.
"""

# Metadata for the test runner
sample_data = "CTChest"
segment_name = "Lung"
gold_standard = "CTChest_lung"


def run(effect):
    """Execute the segmentation recipe."""
    # Use preset for lung parenchyma
    effect.applyPreset("ct_lung")

    # Large brush for lung regions
    effect.brushRadiusMm = 25.0

    # Paint at lung seed locations (from exploration script)
    # Right lung, various axial slices
    effect.paintAt(97.5, -23.30, -225.25)  # intensity: -785
    effect.paintAt(97.5, -23.30, -200.25)  # intensity: -833
    effect.paintAt(97.5, -23.30, -175.25)  # intensity: -902
    effect.paintAt(97.5, -23.30, -150.25)  # intensity: -909
    effect.paintAt(97.5, -23.30, -125.25)  # intensity: -869
