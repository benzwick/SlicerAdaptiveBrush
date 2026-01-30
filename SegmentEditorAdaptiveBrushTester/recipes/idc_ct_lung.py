"""
Recipe: idc_ct_lung

Auto-generated recipe for ct_lung parameter optimization.
Uses IDC data from NCI Imaging Data Commons.
"""

# Metadata
dicom_source = "idc_data/ct_lung"
segment_name = "Ct Lung"
gold_standard = "idc_ct_lung"


def run(effect):
    """Execute the segmentation recipe."""
    effect.applyPreset("ct_lung")
    effect.brushRadiusMm = 15.0

    effect.paintAt(-11.82, -0.44, -120.60)
    effect.paintAt(51.07, 7.08, -205.60)
    effect.paintAt(43.55, -46.92, -95.60)
    effect.paintAt(29.88, 66.55, -140.60)
    effect.paintAt(-4.30, -0.44, -293.10)
