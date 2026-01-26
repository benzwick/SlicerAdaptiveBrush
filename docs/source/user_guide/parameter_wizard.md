# Parameter Wizard

The Parameter Wizard helps you find optimal settings by analyzing your specific image
and segmentation task. It samples the foreground, background, and boundary to recommend
the best algorithm and parameters.

## Starting the Wizard

1. Load your volume in Slicer
2. Open the **Segment Editor** module
3. Select the **Adaptive Brush** effect
4. Click **Quick Select Parameters...** in the Brush Settings section

## Step 1: Sample Foreground

Click on several points **inside** the structure you want to segment.

```{tip}
Sample at least 3-5 points to capture the intensity variation within your target structure.
```

**Good sampling:**
- Click in different regions of the target
- Include both bright and dark areas if present
- Cover the full extent of the structure

**Avoid:**
- Clicking only in one spot
- Sampling at the very edge (that's for step 3)

The wizard displays the current sample count. Click **Next** when ready.

## Step 2: Sample Background

Click on several points **outside** the structure, in areas you want to exclude.

```{tip}
Sample the tissue immediately surrounding your target - this is what the algorithm
needs to distinguish from.
```

**Good sampling:**
- Click on adjacent structures
- Sample tissues with similar appearance to the target
- Include the most challenging "similar but different" areas

The wizard shows real-time analysis of intensity separation:
- **Good separation**: Clear intensity difference between foreground and background
- **Poor separation**: Similar intensities - algorithm selection becomes more important

## Step 3: Trace Boundary

Click along the **boundary** between foreground and background.

This step helps the wizard understand:
- How sharp or gradual the edge is
- The boundary roughness
- Whether edge-based algorithms will work well

**Good sampling:**
- Click along the actual edge
- Follow curves and corners
- Sample both clear and ambiguous boundary regions

## Step 4: Optional Questions

The wizard may ask clarifying questions:

### Structure Type

Select the type of structure you're segmenting:
- **Tumor/Lesion**: Typically well-defined masses
- **Blood Vessel**: Tubular structures
- **Bone**: High contrast on CT
- **Brain Tissue**: Variable contrast
- **Organ**: Large structures with clear boundaries
- **Other**: General purpose

### Image Modality

Select your imaging modality:
- **CT**: Hounsfield units, good for bone/air
- **MRI T1**: Bright fat, dark CSF
- **MRI T2**: Bright fluid, dark bone
- **PET**: Metabolic activity
- **Ultrasound**: Speckle noise considerations
- **Other**: General purpose

### Priority

What matters most for your task:
- **Speed**: Prioritize fast algorithms
- **Precision**: Prioritize accurate algorithms
- **Balanced**: Best overall compromise

## Step 5: Review Recommendations

The wizard presents its analysis and recommendations:

### Algorithm Recommendation

Shows the recommended algorithm with confidence score and explanation:

```
Recommended: Watershed (85% confidence)

Rationale:
- Good intensity separation (0.72)
- Clear boundary gradient detected
- Suitable for general tumor segmentation
```

### Parameter Recommendations

Specific parameter values tailored to your image:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Brush radius | 25mm | Matches structure size |
| Edge sensitivity | 65 | Good boundary contrast |
| Threshold zone | 60 | Based on intensity analysis |

### Alternative Algorithms

If multiple algorithms could work, alternatives are shown:

```
Alternatives:
1. Geodesic Distance (72% confidence) - Good for edge following
2. Level Set (65% confidence) - Higher precision, slower
```

### Warnings

The wizard warns about potential issues:

- **Low intensity separation**: May need manual threshold adjustment
- **Noisy boundary**: Consider increasing smoothing
- **Very large structure**: May need multiple brush strokes

## Applying Recommendations

Click **Apply** to set all recommended parameters automatically.

You can also:
- **Apply and Close**: Set parameters and close wizard
- **Cancel**: Keep current parameters

## Tips for Best Results

### When Wizard Works Best

- Clear intensity difference between target and background
- Well-defined boundaries
- Sufficient sampling (5+ points per category)

### When to Adjust Manually

- Very noisy images (increase smoothing)
- Partial volume effects (adjust threshold)
- Multi-component structures (use multiple clicks)

### Re-running the Wizard

You can run the wizard again:
- For different structures in the same image
- After changing the view/slice
- If initial recommendations don't work well

## Behind the Scenes

The wizard performs these analyses:

1. **Intensity Analysis**
   - Computes mean/std for foreground and background
   - Calculates separation score (0-1)
   - Estimates optimal threshold

2. **Boundary Analysis**
   - Measures gradient magnitude at sampled points
   - Estimates boundary sharpness
   - Calculates roughness (circularity deviation)

3. **Algorithm Scoring**
   - Each algorithm is scored based on the analysis
   - Scores consider intensity separation, boundary quality, and structure type
   - Confidence reflects how well the analysis matches algorithm strengths

4. **Parameter Optimization**
   - Parameters are calculated from intensity statistics
   - Brush size estimated from sample spread
   - Edge sensitivity tuned to boundary contrast
