"""
Recipe: idc_ct_bone

Auto-generated recipe for ct_bone parameter optimization.
Uses IDC data from NCI Imaging Data Commons.
"""

# Metadata
dicom_source = "idc_data/ct_bone"
segment_name = "Ct Bone"
gold_standard = "idc_ct_bone"


def run(effect):
    """Execute the segmentation recipe."""
    effect.applyPreset("ct_bone")
    effect.brushRadiusMm = 15.0

    effect.paintAt(-6.01, 181.43, 65.70)
    effect.paintAt(-12.06, 152.01, 122.70)
    effect.paintAt(-6.87, 190.08, 35.70)
    effect.paintAt(13.89, 164.13, 44.70)
    effect.paintAt(-18.99, 124.33, 212.70)
