"""
Segmentation recipe: TEMPLATE

Description: Template for creating new segmentation recipes
Created: YYYY-MM-DD
Sample Data: <SampleDataName>

Instructions:
1. Copy this file and rename to your recipe name (e.g., my_segmentation.py)
2. Update the recipe metadata (name, description, sample_data, segment_name)
3. Add actions for each brush stroke or effect operation
4. Update optimization_hints to indicate which parameters should be varied

Supported Actions:
- Action.adaptive_brush(...) - Adaptive Brush effect
- Action.paint(...) - Standard Paint effect
- Action.threshold(...) - Threshold effect
- Action.grow_from_seeds() - Grow from Seeds
- Action.islands(...) - Islands operation
- Action.smoothing(...) - Smoothing operation
"""

from SegmentEditorAdaptiveBrushTesterLib.Recipe import Recipe

recipe = Recipe(
    name="template",
    description="Template recipe - copy and modify",
    sample_data="MRHead",  # Change to your sample data
    segment_name="Segment_1",  # Change to your segment name
    actions=[
        # Example: Adaptive Brush stroke
        # Action.adaptive_brush(
        #     ras=(0.0, 0.0, 0.0),  # RAS coordinates
        #     algorithm="watershed",
        #     brush_radius_mm=15.0,
        #     edge_sensitivity=50,
        #     threshold_zone=50,
        #     mode="add",  # or "erase"
        # ),
        # Example: Standard Paint stroke
        # Action.paint(
        #     ras=(0.0, 0.0, 0.0),
        #     radius_mm=5.0,
        #     mode="add",
        # ),
        # Example: Threshold
        # Action.threshold(
        #     min_value=100,
        #     max_value=500,
        # ),
        # Example: Islands - keep largest
        # Action.islands(
        #     operation="KEEP_LARGEST",
        # ),
        # Example: Smoothing
        # Action.smoothing(
        #     method="MEDIAN",
        #     kernel_size_mm=3.0,
        # ),
    ],
)

# Optimization hints for parameter tuning
optimization_hints = {
    # Parameters to vary with same value across all actions
    "vary_globally": ["edge_sensitivity", "threshold_zone"],
    # Parameters that can vary per action
    "vary_per_action": ["brush_radius_mm"],
    # Alternative algorithms to try
    "algorithm_options": ["watershed", "level_set_cpu", "connected_threshold"],
}
