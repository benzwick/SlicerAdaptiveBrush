# create-gold-standard

Create a gold standard segmentation from the current Slicer session.

## Usage

```
/create-gold-standard <name> [--algorithm <algo>] [--description "<desc>"]
```

Where:
- `name` - Name for the gold standard (e.g., "MRBrainTumor1_tumor")
- `--algorithm` - Algorithm used (default: watershed)
- `--description` - Description of the gold standard

## What This Skill Does

1. Reads the current Slicer session state
2. Prompts user to verify/refine the segmentation
3. Records click locations and parameters
4. Saves to `GoldStandards/<name>/`
5. Generates reference screenshots

## Prerequisites

1. Slicer is running with the extension loaded
2. A segmentation has been created that you want to save as gold standard
3. The source volume is loaded (e.g., MRBrainTumor1)

## Manual Steps

Since gold standards require manual verification, this skill provides guidance:

### Step 1: Prepare the Segmentation

In Slicer:
1. Load sample data (e.g., MRBrainTumor1)
2. Create a segmentation using Adaptive Brush
3. Refine until quality is acceptable
4. Note the click locations used

### Step 2: Save Using Python Console

Open Slicer's Python console (View > Python Interactor) and run:

```python
from SegmentEditorAdaptiveBrushTesterLib import GoldStandardManager

manager = GoldStandardManager()

# Get current nodes
seg_node = slicer.util.getNode("Segmentation")
vol_node = slicer.util.getNode("MRBrainTumor1")

# Define click locations (RAS coordinates)
clicks = [
    {"ras": [-5.31, 34.77, 20.83]},
    {"ras": [-5.31, 25.12, 35.97]},
    # Add more as needed
]

# Save gold standard
manager.save_as_gold(
    segmentation_node=seg_node,
    volume_node=vol_node,
    segment_id="Segment_1",  # Or your segment ID
    name="MRBrainTumor1_tumor",
    click_locations=clicks,
    description="Tumor segmentation using watershed, 5 clicks",
    algorithm="watershed",
    parameters={
        "brush_radius_mm": 25.0,
        "edge_sensitivity": 40,
    }
)
```

### Step 3: Capture Reference Screenshots

After saving, capture reference screenshots:

```python
# Navigate to key slices and use the tester panel to capture screenshots
# Or use the ScreenshotCapture utility
```

## Gold Standard Directory

Gold standards are saved to:

```
SegmentEditorAdaptiveBrushTester/GoldStandards/
└── MRBrainTumor1_tumor/
    ├── gold.seg.nrrd          # The segmentation
    ├── metadata.json          # Parameters, clicks, etc.
    └── reference_screenshots/
```

## Verification

After creating a gold standard:

1. Run `/run-regression` to verify it can be reproduced
2. Check the Dice coefficient matches expectations
3. Review reference screenshots for visual quality

## Tips

- Use consistent click locations that can be reproduced
- Document the algorithm and parameters used
- Include enough clicks to achieve > 90% Dice on reproduction
- Take screenshots from multiple views for reference
