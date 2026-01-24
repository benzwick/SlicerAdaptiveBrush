# optimize-segmentation

Optimize AdaptiveBrush segmentation parameters for a specific target structure.

## Usage

```
/optimize-segmentation [sample-data]
```

Where `sample-data` is optional:
- `MRBrainTumor1` (default) - Brain tumor MRI
- `MRHead` - Normal brain MRI
- Or specify a loaded volume name

## What This Skill Does

1. Loads sample data (or uses specified volume)
2. Prompts user to place 5 reference points on target structure
3. Runs all working algorithms with those points
4. Compares results (voxel count, timing, consistency)
5. Recommends best algorithm and parameters
6. Generates optimization report

## Workflow

### Step 1: Load Data and Get Reference Points

Launch Slicer for interactive point placement:

```bash
# Read SLICER_PATH from .env first
SLICER_PATH=$(cat .env | grep SLICER_PATH | cut -d= -f2)
$SLICER_PATH
```

Instruct user:
1. Load sample data: `File > Download Sample Data > [dataset]`
2. Go to Markups module
3. Place 5 fiducial points on the target structure:
   - Point 1: Center of structure
   - Point 2: Superior edge
   - Point 3: Anterior edge
   - Point 4: Posterior edge
   - Point 5: Lateral edge
4. Export markups: `Right-click > Export as...` → save as CSV or copy coordinates

### Step 2: Convert Coordinates (if needed)

**CRITICAL:** Slicer markups export in **LPS** format, but the API uses **RAS**.

Conversion formula:
- R = -L (negate first coordinate)
- A = -P (negate second coordinate)
- S = S (keep third coordinate)

Example:
```
LPS: (5.31, -34.77, 20.83)
RAS: (-5.31, 34.77, 20.83)
```

### Step 3: Update Test Coordinates

Edit `SegmentEditorAdaptiveBrushTester/TestCases/test_optimization_tumor.py`:

```python
TUMOR_CLICK_POINTS = [
    # Convert user's LPS points to RAS
    (-L1, -P1, S1),  # Point 1: Center
    (-L2, -P2, S2),  # Point 2: Superior
    (-L3, -P3, S3),  # Point 3: Anterior
    (-L4, -P4, S4),  # Point 4: Posterior
    (-L5, -P5, S5),  # Point 5: Lateral
]
```

Also update parameters to test:

```python
OPTIMIZATION_PARAMS = {
    "brush_radius_mm": 20.0,  # Adjust based on structure size
    "edge_sensitivity": 35,   # 0-100, lower = more propagation
    "inner_radius_ratio": 0.3,
}
```

### Step 4: Run Optimization Test

```bash
SLICER_PATH=$(cat .env | grep SLICER_PATH | cut -d= -f2)
$SLICER_PATH --python-script scripts/run_tests.py --exit optimization
```

### Step 5: Analyze Results

Read the results file:

```bash
# Find latest optimization run
ls -t test_runs/ | grep optimization | head -1
```

Then read `test_runs/<run>/results.json` to get:
- Voxel counts per algorithm
- Timing data
- Screenshots

### Step 6: Generate Report

Create optimization report with:

1. **Data Summary**
   - Dataset used
   - Reference points (RAS coordinates)
   - Structure being segmented

2. **Algorithm Comparison Table**
   | Algorithm | Voxels | Time (ms) | Voxels/ms | Rank |
   |-----------|--------|-----------|-----------|------|

3. **Recommendations**
   - Best algorithm for this structure
   - Optimal brush radius
   - Optimal edge sensitivity

4. **Screenshots**
   - Link to screenshots for visual verification

## Working Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `connected_threshold` | Flood fill within intensity range | Maximum coverage |
| `threshold_brush` | Threshold + brush mask | Consistent results |
| `region_growing` | Iterative region expansion | Homogeneous regions |
| `watershed` | Gradient-based boundaries | Edge-sensitive segmentation |

**DO NOT USE:** `level_set_cpu` - Currently broken (produces 0 voxels)

## Parameter Guidelines

### Brush Radius
- Small structures (< 10mm): 5-10mm brush
- Medium structures (10-30mm): 15-20mm brush
- Large structures (> 30mm): 25-35mm brush

### Edge Sensitivity (0-100)
- Low (10-30): More propagation, may leak
- Medium (30-50): Balanced
- High (50-80): Stops at weak edges, may under-segment

### Number of Clicks
- Minimum: 3 (center + 2 edges)
- Recommended: 5 (center + 4 cardinal edges)
- Maximum useful: 8-10 for complex shapes

## Example Output

```
OPTIMIZATION RESULTS: MRBrainTumor1
===================================
Reference Points: 5 (user-placed on tumor)
Brush Radius: 20mm
Edge Sensitivity: 35

ALGORITHM COMPARISON:
---------------------
1. connected_threshold: 22,696 voxels (1,252ms) - BEST COVERAGE
2. threshold_brush:     15,298 voxels (1,095ms) - MOST CONSISTENT
3. watershed:           13,159 voxels (1,808ms) - GOOD EDGES
4. region_growing:      11,694 voxels (1,172ms) - BALANCED

RECOMMENDATIONS:
----------------
- For maximum recall: connected_threshold
- For consistency: threshold_brush
- For edge precision: watershed

Optimal parameters for this structure:
- Brush: 20mm
- Sensitivity: 35
- Algorithm: threshold_brush (recommended)
```

## Troubleshooting

### "0 voxels" for all algorithms
- Check coordinate conversion (LPS vs RAS)
- Verify points are inside the volume
- Check brush radius is large enough

### Segmentation leaks outside structure
- Increase edge sensitivity
- Decrease brush radius
- Use threshold_brush instead of connected_threshold

### Inconsistent results between clicks
- Increase brush radius for more overlap
- Use region_growing for homogeneous structures
- Place points more centrally in the structure
