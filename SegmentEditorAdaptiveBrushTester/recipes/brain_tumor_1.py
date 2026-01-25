"""
Segmentation recipe: brain_tumor_1

Description: 5-click watershed segmentation of brain tumor in MRBrainTumor1
Created: 2026-01-25
Sample Data: MRBrainTumor1
Gold Standard: GoldStandards/MRBrainTumor1_tumor/gold.seg.nrrd

This recipe captures a typical workflow for segmenting a ring-enhancing
brain tumor using the Adaptive Brush with watershed algorithm.
"""

from SegmentEditorAdaptiveBrushTesterLib.Recipe import Action, Recipe

recipe = Recipe(
    name="brain_tumor_1",
    description="5-click watershed segmentation of brain tumor",
    sample_data="MRBrainTumor1",
    segment_name="Tumor",
    actions=[
        # Action 1 - Initial stroke at tumor center
        Action.adaptive_brush(
            ras=(-5.31, 34.77, 20.83),
            algorithm="watershed",
            brush_radius_mm=25.0,
            edge_sensitivity=40,
            threshold_zone=50,
            watershedGradientScale=1.5,
            watershedSmoothing=0.5,
            mode="add",
        ),
        # Action 2 - Expand into tumor core
        Action.adaptive_brush(
            ras=(-5.31, 25.12, 35.97),
            algorithm="watershed",
            brush_radius_mm=25.0,
            edge_sensitivity=40,
            threshold_zone=50,
            watershedGradientScale=1.5,
            watershedSmoothing=0.5,
            mode="add",
        ),
        # Action 3 - Capture lower portion
        Action.adaptive_brush(
            ras=(-5.31, 20.70, 22.17),
            algorithm="watershed",
            brush_radius_mm=15.0,
            edge_sensitivity=50,
            threshold_zone=50,
            watershedGradientScale=1.5,
            watershedSmoothing=0.5,
            mode="add",
        ),
        # Action 4 - Expand laterally
        Action.adaptive_brush(
            ras=(-6.16, 38.28, 30.61),
            algorithm="watershed",
            brush_radius_mm=20.0,
            edge_sensitivity=40,
            threshold_zone=50,
            watershedGradientScale=1.5,
            watershedSmoothing=0.5,
            mode="add",
        ),
        # Action 5 - Final touch
        Action.adaptive_brush(
            ras=(-1.35, 28.65, 18.90),
            algorithm="watershed",
            brush_radius_mm=15.0,
            edge_sensitivity=45,
            threshold_zone=50,
            watershedGradientScale=1.5,
            watershedSmoothing=0.5,
            mode="add",
        ),
    ],
)

# Optimization hints for parameter tuning
optimization_hints = {
    "vary_globally": ["edge_sensitivity", "threshold_zone", "watershedGradientScale"],
    "vary_per_action": ["brush_radius_mm"],
    "algorithm_options": ["watershed", "level_set_cpu", "connected_threshold", "region_growing"],
}
