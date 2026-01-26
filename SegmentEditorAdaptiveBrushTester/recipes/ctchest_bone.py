"""
Recipe: ctchest_bone

Segmentation of vertebral bone in CTChest sample data.
CTChest is a thoracic CT scan.
Bone appears bright on CT with intensity > 200 HU.

NOTE: Seeds estimated from anatomy - may need refinement.
"""

# Metadata for the test runner
sample_data = "CTChest"
segment_name = "Bone"
gold_standard = "CTChest_bone"


def run(effect):
    """Execute the segmentation recipe."""
    # Use preset for bone
    effect.applyPreset("ct_bone")

    # Medium brush for vertebral bodies
    effect.brushRadiusMm = 10.0

    # Paint at estimated vertebral body locations
    # Vertebrae are posterior and central (A is negative in posterior)
    # CTChest origin: [195, 171.7, -347.75]
    effect.paintAt(0.0, 100.0, -200.0)  # Mid-thoracic vertebra
    effect.paintAt(0.0, 100.0, -175.0)  # Upper thoracic
    effect.paintAt(0.0, 100.0, -225.0)  # Lower thoracic
