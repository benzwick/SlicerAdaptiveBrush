# SlicerAdaptiveBrush

[![CI](https://github.com/benzwick/SlicerAdaptiveBrush/actions/workflows/ci.yml/badge.svg)](https://github.com/benzwick/SlicerAdaptiveBrush/actions/workflows/ci.yml)
[![Documentation](https://github.com/benzwick/SlicerAdaptiveBrush/actions/workflows/docs.yml/badge.svg)](https://github.com/benzwick/SlicerAdaptiveBrush/actions/workflows/docs.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![3D Slicer](https://img.shields.io/badge/3D_Slicer-5.10%2B-green.svg)](https://www.slicer.org/)

An **Adaptive Brush** segment editor effect extension for [3D Slicer](https://www.slicer.org/).

## Overview

The Adaptive Brush automatically segments regions based on image intensity similarity within the brush area, adapting to image features (edges, boundaries) rather than using a fixed geometric shape. This is similar to the adaptive brush tools found in ITK-SNAP and ImFusion Labels.

## Features

- **Multiple algorithm choices** - Geodesic Distance, Watershed, Random Walker, Level Set, Connected Threshold, Region Growing, Threshold Brush
- **Auto-threshold methods** - Otsu, Huang, Triangle, Maximum Entropy, IsoData, Li
- **Automatic intensity analysis** - GMM-based threshold estimation adapts to image content
- **Edge-aware boundaries** - Respects anatomical boundaries automatically
- **2D and 3D modes** - Works on single slices or volumetrically (sphere mode)
- **Adjustable parameters** - Control brush radius, edge sensitivity, algorithm choice
- **Visual brush outline** - See brush radius as you paint
- **Undo support** - Full undo/redo integration with Slicer

## Planned Features

- GPU acceleration for Level Set algorithm (OpenCL/CUDA)
- Performance caching optimization for smoother drag operations
- Preview mode during drag (reduced resolution for speed)

## Installation

> **Note:** SlicerAdaptiveBrush is not yet available in the Extension Index.
> Once published, installation will be via Extension Manager search.

### From GitHub Release (Recommended)

1. Download the package for your platform from [GitHub Releases](https://github.com/benzwick/SlicerAdaptiveBrush/releases)
2. Open 3D Slicer
3. Go to **View** > **Extension Manager**
4. Click **Install from file...**
5. Select the downloaded `.tar.gz` (Linux/macOS) or `.zip` (Windows) file
6. Restart Slicer

### From Source

1. Clone this repository: `git clone https://github.com/benzwick/SlicerAdaptiveBrush.git`
2. Open 3D Slicer
3. Drag-and-drop the `SlicerAdaptiveBrush/SlicerAdaptiveBrush` folder onto the Slicer application window
4. In the popup, select **Add Python scripted modules to the application**
5. Select which modules to load and click **Yes**
6. Check **Add selected modules to 'Additional module paths'** if you want them to load on future sessions

**Alternative method:** Go to **Edit** > **Application Settings** > **Modules** and drag-and-drop individual module folders to the **Additional module paths** list, then restart Slicer.

Included modules:
- **Segment Editor Adaptive Brush** - Adaptive brush effect for Segment Editor
- **Adaptive Brush Reviewer** - Review optimization results and manage gold standards
- **Adaptive Brush Tester** - Testing framework with automated and manual testing

<!--
### From Extension Manager (After Publishing)

1. Open 3D Slicer
2. Go to **View** > **Extension Manager**
3. Search for "AdaptiveBrush"
4. Click **Install**
5. Restart Slicer
-->

## Usage

1. Load a volume (CT, MRI, etc.)
2. Open the **Segment Editor** module
3. Create or select a segment
4. Select the **Adaptive Brush** effect from the effects toolbar
5. Adjust parameters:
   - **Radius**: Size of the brush in mm
   - **Edge Sensitivity**: How strictly to follow intensity boundaries (0-100%)
   - **3D Mode**: Enable for volumetric painting
6. Click and drag on the image to paint
7. For **Threshold Brush** algorithm:
   - Enable **Auto threshold** for automatic method selection
   - Choose threshold method (Otsu, Huang, Triangle, etc.)
   - Or disable auto and set manual thresholds with sliders

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **Geodesic Distance** | Fast marching with edge weighting (default) |
| **Watershed** | Marker-based morphological watershed |
| **Random Walker** | Probabilistic diffusion from seeds |
| **Level Set (GPU)** | Geodesic active contours (GPU accelerated) |
| **Level Set (CPU)** | Geodesic active contours (CPU fallback) |
| **Connected Threshold** | Flood-fill within intensity range |
| **Region Growing** | Confidence-connected expansion |
| **Threshold Brush** | Intensity thresholding with auto-detection |

### Shared Pipeline

All algorithms share a common pipeline:
1. **ROI Extraction** - Extract region around cursor
2. **Intensity Analysis** - Automatically estimate optimal thresholds using GMM
3. **Algorithm-specific segmentation** - Your chosen algorithm
4. **Post-processing** - Apply circular/spherical brush mask

## Requirements

- 3D Slicer 5.10 or later
- No additional dependencies (uses bundled SimpleITK, NumPy, VTK)
- Optional: scikit-learn for GMM analysis (falls back to simple statistics if unavailable)

## Development

This project follows Test-Driven Development (TDD). See [CLAUDE.md](CLAUDE.md) for development guidelines.

```bash
# Local development with uv
uv sync --extra dev
uv run pytest -v
uv run ruff check .

# Run tests in Slicer Python console
import SegmentEditorAdaptiveBrush
SegmentEditorAdaptiveBrush.SegmentEditorAdaptiveBrushTest().runTest()
```

## References

- [ITK-SNAP Adaptive Brush](https://www.itksnap.org/) - Original inspiration
- [3D Slicer Segment Editor](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html)
- [SlicerSegmentEditorExtraEffects](https://github.com/lassoan/SlicerSegmentEditorExtraEffects)
- [SimpleITK Documentation](https://simpleitk.readthedocs.io/)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read [CLAUDE.md](CLAUDE.md) for development guidelines and see [ROADMAP.md](ROADMAP.md) for planned features.
