"""
Recipe: mrhead_ventricles

Segmentation of lateral ventricles in MRHead sample data.
MRHead is a T1-weighted MRI of a healthy brain.
Ventricles (CSF) appear dark in T1 MRI.

NOTE: Seeds estimated from anatomy - may need refinement.
"""

# Metadata for the test runner
sample_data = "MRHead"
segment_name = "Ventricles"
gold_standard = "MRHead_ventricles"


def run(effect):
    """Execute the segmentation recipe."""
    # Use preset for T1 brain tissue
    effect.applyPreset("mri_t1_brain")

    # Small brush for ventricle bodies
    effect.brushRadiusMm = 8.0

    # Paint at estimated ventricle locations
    # Lateral ventricles are central, ~15-25mm lateral to midline
    # Body of lateral ventricles
    effect.paintAt(0.0, 10.0, 25.0)  # Central, body
    effect.paintAt(15.0, 5.0, 20.0)  # Right lateral ventricle
    effect.paintAt(-15.0, 5.0, 20.0)  # Left lateral ventricle
