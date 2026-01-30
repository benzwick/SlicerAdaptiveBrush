"""
Recipe: idc_mri_t1gd_tumor

Auto-generated recipe for mri_t1gd_tumor parameter optimization.
Uses IDC data from NCI Imaging Data Commons.
"""

# Metadata
dicom_source = "idc_data/mri_t1gd_tumor"
segment_name = "Mri T1Gd Tumor"
gold_standard = "idc_mri_t1gd_tumor"


def run(effect):
    """Execute the segmentation recipe."""
    effect.applyPreset("mri_t1gd_tumor")
    effect.brushRadiusMm = 15.0

    effect.paintAt(56.75, 14.63, 15.87)
    effect.paintAt(43.13, 43.95, 20.87)
    effect.paintAt(48.95, 22.46, 23.87)
    effect.paintAt(48.91, 2.92, 9.87)
    effect.paintAt(43.07, 13.68, 18.87)
