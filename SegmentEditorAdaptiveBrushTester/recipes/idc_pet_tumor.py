"""
Recipe: idc_pet_tumor

Auto-generated recipe for pet_tumor parameter optimization.
Uses IDC data from NCI Imaging Data Commons.
"""

# Metadata
dicom_source = "idc_data/pet_tumor"
segment_name = "Pet Tumor"
gold_standard = "idc_pet_tumor"


def run(effect):
    """Execute the segmentation recipe."""
    effect.applyPreset("pet_tumor")
    effect.brushRadiusMm = 15.0

    effect.paintAt(35.53, 112.10, -315.46)
    effect.paintAt(79.28, -46.50, -371.05)
    effect.paintAt(-62.91, -8.22, -299.11)
    effect.paintAt(-79.32, 13.66, -331.81)
    effect.paintAt(-134.01, 30.06, -423.37)
