"""
Recipe: mrbrain_tumor2

Segmentation of tumor in MRBrainTumor2 sample data.
MRBrainTumor2 is a brain MRI with a tumor.

NOTE: Seeds estimated - may need refinement based on actual tumor location.
"""

# Metadata for the test runner
sample_data = "MRBrainTumor2"
segment_name = "Tumor"
gold_standard = "MRBrainTumor2_tumor"


def run(effect):
    """Execute the segmentation recipe."""
    # Use preset for tumor
    effect.applyPreset("generic_tumor")

    # Medium brush for tumor
    effect.brushRadiusMm = 15.0

    # Paint at estimated tumor locations
    # MRBrainTumor2 origin: [119.53, 119.53, -77.4]
    # Tumor location needs to be determined from the actual data
    effect.paintAt(0.0, 0.0, 0.0)  # Center - placeholder
    effect.paintAt(10.0, 10.0, 0.0)  # Nearby - placeholder
    effect.paintAt(-10.0, -10.0, 0.0)  # Nearby - placeholder
