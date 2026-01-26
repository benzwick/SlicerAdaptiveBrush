# Getting Started

## Installation

```{todo}
SlicerAdaptiveBrush is not yet available in the Extension Index.
Once released, installation will be via Extension Manager.
```

### Current Installation (Pre-release)

Until the extension is published to the Extension Index, install from GitHub:

1. Download the latest release from [GitHub](https://github.com/benzwick/SlicerAdaptiveBrush/releases)
   - Or clone the repository: `git clone https://github.com/benzwick/SlicerAdaptiveBrush.git`
2. Open 3D Slicer
3. Go to **Edit** → **Application Settings** → **Modules**
4. Add the path to `SlicerAdaptiveBrush/SlicerAdaptiveBrush` to **Additional module paths**
5. Restart Slicer

### Future Installation (After Release)

Once published, installation will be simpler:

1. Open 3D Slicer
2. Go to **View** → **Extension Manager**
3. Search for "AdaptiveBrush"
4. Click **Install**
5. Restart Slicer when prompted

## Quick Start

### Loading Data

1. Load a volume using **File** → **Add Data** or drag and drop
2. For practice, use **File** → **Download Sample Data** → **MRBrainTumor1**

### Using the Adaptive Brush

1. Open the **Segment Editor** module
2. Create a new segmentation or select an existing one
3. Click the **Adaptive Brush** effect (brush with sparkles icon)
4. Adjust the brush radius using the slider or **Shift + scroll wheel**
5. Click on the region you want to segment

```{image} /_static/adaptive_brush_panel.png
:alt: Adaptive Brush options panel
:width: 300px
```

### Basic Controls

| Action | Result |
|--------|--------|
| Left-click | Paint with adaptive brush |
| Ctrl + Left-click | Erase mode |
| Shift + Scroll | Adjust brush radius |
| Ctrl + Shift + Scroll | Adjust threshold zone |

## Choosing an Algorithm

The Adaptive Brush provides multiple algorithms for different use cases:

| Algorithm | Speed | Best For |
|-----------|-------|----------|
| **Watershed** (default) | Medium | General use, good boundary adherence |
| **Connected Threshold** | Fast | Quick rough segmentation |
| **Region Growing** | Fast | Homogeneous regions |
| **Threshold Brush** | Very Fast | Simple intensity-based painting |
| **Geodesic Distance** | Medium | Structures with clear edges |
| **Level Set** | Slow | High precision needs |
| **Random Walker** | Slow | Complex boundaries |

See [Algorithms](algorithms.md) for detailed descriptions.

## Parameter Wizard

Not sure which parameters to use? The **Parameter Wizard** analyzes your image and recommends optimal settings.

1. Click **Quick Select Parameters...** in the Brush Settings section
2. Follow the 5-step wizard:
   - Sample the **foreground** (what you want to segment)
   - Sample the **background** (what you want to exclude)
   - Trace the **boundary** between them
   - Answer optional questions about the structure
   - Review and apply recommendations

See [Parameter Wizard](parameter_wizard.md) for details.

## Tips for Best Results

### Brush Size

- Use a brush slightly smaller than the structure you're segmenting
- For small structures, use smaller brushes for precision
- For large homogeneous regions, use larger brushes for efficiency

### Edge Sensitivity

- **Higher values** (70-100): Better boundary adherence, may under-segment
- **Lower values** (20-50): More aggressive segmentation, may leak into neighbors
- **Default (50)**: Balanced for most cases

### Multiple Clicks

Complex structures often require multiple clicks:

1. Start with the center of the structure
2. Add clicks at edges that weren't captured
3. Use **Ctrl + click** to erase any over-segmentation

## Next Steps

- Learn about [available algorithms](algorithms.md)
- Use the [Parameter Wizard](parameter_wizard.md) for optimal settings
- Create [segmentation recipes](recipes.md) for reproducible workflows
