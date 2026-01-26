"""
Recipe: mrhead_white_matter

Segmentation of white matter in MRHead sample data.
MRHead is a T1-weighted MRI of a healthy brain.
White matter appears bright in T1 MRI.
"""

# Metadata for the test runner
sample_data = "MRHead"
segment_name = "WhiteMatter"
gold_standard = "MRHead_white_matter"


def run(effect):
    """Execute the segmentation recipe."""
    # Use preset for T1 brain tissue
    effect.applyPreset("mri_t1_brain")

    # Medium brush for white matter regions
    effect.brushRadiusMm = 15.0

    # Paint at white matter seed locations (from exploration script)
    # Central white matter region, mid-axial slice
    effect.paintAt(-2.15, 25.93, 8.79)  # intensity: 78
    effect.paintAt(-2.15, 15.93, 8.79)  # intensity: 104
    effect.paintAt(-2.15, 5.93, 8.79)  # intensity: 81
    effect.paintAt(-2.15, -4.07, 8.79)  # intensity: 62
    effect.paintAt(-2.15, 25.93, -1.21)  # intensity: 59
