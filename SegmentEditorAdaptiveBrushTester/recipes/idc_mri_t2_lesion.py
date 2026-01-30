"""
Recipe: idc_mri_t2_lesion

Auto-generated recipe for mri_t2_lesion parameter optimization.
Uses IDC data from NCI Imaging Data Commons.
"""

# Metadata
dicom_source = "idc_data/mri_t2_lesion"
segment_name = "Mri T2 Lesion"
gold_standard = "idc_mri_t2_lesion"


def run(effect):
    """Execute the segmentation recipe."""
    effect.applyPreset("mri_t2_lesion")
    effect.brushRadiusMm = 15.0

    effect.paintAt(42.52, -29.98, 54.22)
    effect.paintAt(40.64, -29.98, 48.22)
    effect.paintAt(49.10, -17.80, 54.22)
    effect.paintAt(42.53, -23.42, 51.22)
    effect.paintAt(44.39, -29.05, 42.22)
