"""
Recipe: idc_ct_soft_tissue

Auto-generated recipe for ct_soft_tissue parameter optimization.
Uses IDC data from NCI Imaging Data Commons.
"""

# Metadata
dicom_source = "idc_data/ct_soft_tissue"
segment_name = "Ct Soft Tissue"
gold_standard = "idc_ct_soft_tissue"


def run(effect):
    """Execute the segmentation recipe."""
    effect.applyPreset("ct_soft_tissue")
    effect.brushRadiusMm = 15.0

    effect.paintAt(21.88, -155.00, -279.80)
    effect.paintAt(28.12, -113.59, -320.80)
    effect.paintAt(-13.28, -161.25, -273.80)
    effect.paintAt(3.12, -147.19, -288.80)
    effect.paintAt(10.16, -171.41, -210.80)
