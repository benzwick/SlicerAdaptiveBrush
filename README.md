# SlicerAdaptiveBrush

[![Tests](https://github.com/benzwick/SlicerAdaptiveBrush/actions/workflows/tests.yml/badge.svg)](https://github.com/benzwick/SlicerAdaptiveBrush/actions/workflows/tests.yml)
[![Documentation](https://github.com/benzwick/SlicerAdaptiveBrush/actions/workflows/docs.yml/badge.svg)](https://benzwick.github.io/SlicerAdaptiveBrush/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![3D Slicer](https://img.shields.io/badge/3D_Slicer-5.10%2B-green.svg)](https://www.slicer.org/)

An **Adaptive Brush** segment editor effect for [3D Slicer](https://www.slicer.org/).

![Adaptive Brush in action](Screenshots/main-ui.png)

**[Documentation](https://benzwick.github.io/SlicerAdaptiveBrush/)** |
[User Guide](https://benzwick.github.io/SlicerAdaptiveBrush/user_guide/getting_started.html) |
[Algorithms](https://benzwick.github.io/SlicerAdaptiveBrush/user_guide/algorithms.html)

## What is Adaptive Brush?

The Adaptive Brush automatically segments regions based on image intensity similarity. Unlike a fixed geometric brush, it adapts to image features (edges, boundaries), similar to tools in ITK-SNAP and ImFusion Labels.

## Features

- **Multiple algorithms** - Watershed, Geodesic Distance, Level Set, and more
- **Auto-threshold detection** - Otsu, Huang, Triangle, and other methods
- **Edge-aware boundaries** - Respects anatomical boundaries automatically
- **2D and 3D modes** - Paint on slices or volumetrically
- **Full undo support** - Integrated with Slicer's undo/redo system

## Installation

> **Note:** SlicerAdaptiveBrush is not yet in the Extension Index.

### From GitHub Release

1. Download the package for your platform from [Releases](https://github.com/benzwick/SlicerAdaptiveBrush/releases)
2. In Slicer: **View > Extension Manager > Install from file...**
3. Select the downloaded file and restart Slicer

### From Source

1. Clone: `git clone https://github.com/benzwick/SlicerAdaptiveBrush.git`
2. Drag-and-drop the `SlicerAdaptiveBrush` folder onto Slicer
3. Select **Add Python scripted modules** and click **Yes**

## Quick Start

1. Load a volume (CT, MRI)
2. Open **Segment Editor**
3. Create or select a segment
4. Select **Adaptive Brush** from effects
5. Click and drag on the image to paint

## Controls

| Action | Shortcut |
|--------|----------|
| Paint | Left-click drag |
| Erase | Ctrl + Left-click drag |
| Erase (alternate) | Middle + Left-click drag |
| Adjust brush size | Shift + Scroll wheel |
| Adjust threshold zone | Ctrl + Shift + Scroll wheel |

## Algorithms

| Algorithm | Speed | Best For |
|-----------|-------|----------|
| Geodesic Distance | Fast | General use (default) |
| Watershed | Medium | Marker-based segmentation |
| Level Set | Slow | High precision, irregular boundaries |
| Connected Threshold | Very Fast | Quick rough segmentation |
| Threshold Brush | Very Fast | Simple intensity thresholding |

See the [Algorithms Guide](https://benzwick.github.io/SlicerAdaptiveBrush/user_guide/algorithms.html) for details.

## Troubleshooting

### Brush outline not appearing
- Ensure a segmentation node exists
- Verify you're in a slice view (Red, Yellow, or Green)
- Check that the effect is selected in Segment Editor

### Segmentation leaking into unwanted regions
- Increase **Edge Sensitivity** (higher = stricter boundaries)
- Use a smaller brush radius
- Try a different algorithm (Watershed or Level Set)

### Algorithm is slow
- Use 2D mode instead of 3D for faster interaction
- Try Connected Threshold or Threshold Brush for speed
- Reduce brush radius

## Requirements

- 3D Slicer 5.10 or later
- No additional dependencies (uses bundled SimpleITK, NumPy, VTK)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## References

- [ITK-SNAP](https://www.itksnap.org/) - Original inspiration
- [3D Slicer Segment Editor](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html)
- [SlicerSegmentEditorExtraEffects](https://github.com/lassoan/SlicerSegmentEditorExtraEffects)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
