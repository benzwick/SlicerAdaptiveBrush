"""
Recipe: ctacardio_left_ventricle

Segmentation of left ventricle in CTACardio sample data.
CTACardio is a contrast-enhanced cardiac CT (CTA).
The left ventricle blood pool is enhanced with contrast agent.
"""

# Metadata for the test runner
sample_data = "CTACardio"
segment_name = "LeftVentricle"
gold_standard = "CTACardio_left_ventricle"


def run(effect):
    """Execute the segmentation recipe."""
    # Use preset for contrast-enhanced vessels/chambers
    effect.applyPreset("ct_vessel_contrast")

    # Medium brush for cardiac chamber
    effect.brushRadiusMm = 15.0

    # Paint at left ventricle seed locations (from exploration script)
    # Various levels through the LV chamber
    effect.paintAt(-48.08, -0.47, -12.50)  # intensity: 103
    effect.paintAt(-48.08, -0.47, -6.25)  # intensity: 209
    effect.paintAt(-48.08, -0.47, 0.0)  # intensity: 99
    effect.paintAt(-48.08, -0.47, 6.25)  # intensity: 104
    effect.paintAt(-48.08, -0.47, 12.50)  # intensity: 134
