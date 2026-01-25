"""
Recipe: TEMPLATE

Copy this file and customize for your segmentation.
Recipes are Python scripts with full access to Slicer's API.
"""

# Metadata for the test runner
sample_data = "MRHead"  # Slicer SampleData name
segment_name = "Segment_1"


def run(effect):
    """Execute the segmentation recipe.

    Args:
        effect: The Adaptive Brush scripted effect instance.

    Presets (by modality - set common parameters, NOT algorithm):
        CT:
            effect.applyPreset("ct_bone")           # High contrast bone
            effect.applyPreset("ct_soft_tissue")    # Liver, kidney, spleen
            effect.applyPreset("ct_lung")           # Lung parenchyma
            effect.applyPreset("ct_vessel_contrast") # CTA vessels

        MRI T1:
            effect.applyPreset("mri_t1_brain")      # Gray/white matter
            effect.applyPreset("mri_t1_fat")        # Adipose tissue

        MRI T1+Gd:
            effect.applyPreset("mri_t1gd_tumor")    # Enhancing tumor

        MRI T2/FLAIR:
            effect.applyPreset("mri_t2_lesion")     # Hyperintense lesions

        Generic:
            effect.applyPreset("default")           # Balanced defaults
            effect.applyPreset("generic_tumor")     # Any tumor/mass
            effect.applyPreset("generic_vessel")    # Any vessel

    Properties:
        effect.brushRadiusMm = 20.0          # Brush size in mm
        effect.edgeSensitivityValue = 50     # Edge sensitivity 0-100

    Paint:
        effect.paintAt(r, a, s)              # Paint at RAS coords
        effect.paintAt(r, a, s, erase=True)  # Erase at RAS coords

    Algorithm Selection (separate from presets):
        # Algorithm is selected via UI or Slicer parameter system
        # See user documentation for algorithm recommendations

    Other Slicer effects (example):
        import slicer
        editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
        editor.setActiveEffectByName("Islands")
        effect = editor.activeEffect()
        effect.setParameter("Operation", "KEEP_LARGEST_ISLAND")
        effect.self().onApply()
    """
    # Example:
    # effect.applyPreset("ct_soft_tissue")
    # effect.brushRadiusMm = 15.0
    # effect.paintAt(0.0, 0.0, 0.0)
    pass
