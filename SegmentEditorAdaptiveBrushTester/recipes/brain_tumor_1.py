"""
Recipe: brain_tumor_1

5-click segmentation of brain tumor in MRBrainTumor1.
This is a T1+Gd contrast-enhanced MRI with a ring-enhancing tumor.
"""

# Metadata for the test runner
sample_data = "MRBrainTumor1"
segment_name = "Tumor"
gold_standard = "MRBrainTumor1_tumor"  # For regression testing


def run(effect):
    """Execute the segmentation recipe."""
    # Use preset for contrast-enhanced tumor in T1+Gd
    effect.applyPreset("mri_t1gd_tumor")

    # Explicitly set watershed algorithm (preset doesn't specify algorithm)
    effect.algorithm = "watershed"

    # Brush sized for this ~3cm tumor
    effect.brushRadiusMm = 20.0

    # Segment the tumor with 5 clicks
    effect.paintAt(-5.31, 34.77, 20.83)
    effect.paintAt(-5.31, 25.12, 35.97)
    effect.paintAt(-5.31, 20.70, 22.17)
    effect.paintAt(-6.16, 38.28, 30.61)
    effect.paintAt(-1.35, 28.65, 18.90)
