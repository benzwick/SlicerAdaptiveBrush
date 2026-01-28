"""
Recipe: brain_tumor_1_alt

Alternative click pattern for brain tumor in MRBrainTumor1.
Uses different seed points than brain_tumor_1.py to test algorithm robustness
to click placement variation.

Based on the original tumor location but with shifted click positions.
"""

# Metadata for the test runner
sample_data = "MRBrainTumor1"
segment_name = "Tumor"
gold_standard = "MRBrainTumor1_tumor"


def run(effect):
    """Execute the segmentation recipe with alternative clicks."""
    # Use preset for contrast-enhanced tumor in T1+Gd
    effect.applyPreset("mri_t1gd_tumor")

    # Brush sized for this ~3cm tumor
    effect.brushRadiusMm = 20.0

    # Alternative click pattern - shifted from original positions
    # Original tumor center approximately at (-4, 30, 25)
    # These clicks approach from different angles
    effect.paintAt(-3.0, 32.0, 25.0)  # Center-ish
    effect.paintAt(-7.0, 28.0, 18.0)  # Lower left
    effect.paintAt(-2.0, 22.0, 30.0)  # Upper right
    effect.paintAt(-8.0, 35.0, 35.0)  # Top back
    effect.paintAt(-1.0, 38.0, 22.0)  # Front
    effect.paintAt(-5.0, 30.0, 28.0)  # Additional center
