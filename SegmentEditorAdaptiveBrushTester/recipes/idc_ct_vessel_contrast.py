"""
Recipe: idc_ct_vessel_contrast

Auto-generated recipe for ct_vessel_contrast parameter optimization.
Uses IDC data from NCI Imaging Data Commons.
"""

# Metadata
dicom_source = "idc_data/ct_vessel_contrast"
segment_name = "Ct Vessel Contrast"
gold_standard = "idc_ct_vessel_contrast"


def run(effect):
    """Execute the segmentation recipe."""
    effect.applyPreset("ct_vessel_contrast")
    effect.brushRadiusMm = 15.0

    effect.paintAt(59.76, 50.01, -275.00)
    effect.paintAt(7.39, 48.37, -250.00)
    effect.paintAt(23.75, 64.74, -260.00)
    effect.paintAt(64.67, -17.91, -290.00)
    effect.paintAt(82.68, 9.09, -310.00)
