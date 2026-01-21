# SlicerAdaptiveBrush

An **Adaptive Brush** segment editor effect extension for [3D Slicer](https://www.slicer.org/).

## Overview

The Adaptive Brush automatically segments regions based on image intensity similarity within the brush area, adapting to image features (edges, boundaries) rather than using a fixed geometric shape. This is similar to the adaptive brush tools found in ITK-SNAP and ImFusion Labels.

## Features

- **Multiple algorithm choices** - Watershed, Level Set, Connected Threshold, Region Growing
- **CPU and GPU support** - Use GPU acceleration when available, automatic fallback to CPU
- **Automatic intensity-based segmentation** - Brush adapts to image content
- **Edge-aware boundaries** - Respects anatomical boundaries automatically
- **2D and 3D modes** - Works on single slices or volumetrically
- **Adjustable parameters** - Control brush radius, edge sensitivity, algorithm choice
- **Real-time preview** - See segmentation results as you paint

## Installation

### From Extension Manager (Recommended)

1. Open 3D Slicer
2. Go to **View** > **Extension Manager**
3. Search for "AdaptiveBrush"
4. Click **Install**
5. Restart Slicer

### From Source

1. Clone this repository
2. In Slicer, go to **Edit** > **Application Settings** > **Modules**
3. Add the path to `SlicerAdaptiveBrush` to **Additional module paths**
4. Restart Slicer

## Usage

1. Load a volume (CT, MRI, etc.)
2. Open the **Segment Editor** module
3. Create or select a segment
4. Select the **Adaptive Brush** effect from the effects toolbar
5. Adjust parameters:
   - **Radius**: Size of the brush in mm
   - **Edge Sensitivity**: How strictly to follow intensity boundaries
   - **3D Mode**: Enable for volumetric painting
6. Click and drag on the image to paint

## Algorithms

The adaptive brush offers multiple algorithm choices:

| Algorithm | Backend | Speed | Precision | Best For |
|-----------|---------|-------|-----------|----------|
| **Watershed** | CPU | Medium | High | General use (default) |
| **Level Set** | GPU | Fast | Very High | Users with GPU |
| **Level Set** | CPU | Slow | Very High | Precision without GPU |
| **Connected Threshold** | CPU | Very Fast | Low | Quick rough segmentation |
| **Region Growing** | CPU | Fast | Medium | Homogeneous regions |

All algorithms share a common pipeline:
1. **Intensity Analysis** - Automatically estimates optimal thresholds using GMM
2. **Algorithm-specific segmentation** - Your chosen algorithm
3. **Post-processing** - Mask to brush radius, optional smoothing

## Requirements

- 3D Slicer 5.10 or later
- No additional dependencies (uses bundled SimpleITK, NumPy, VTK)

## Development

This project follows Test-Driven Development (TDD). See [CLAUDE.md](CLAUDE.md) for development guidelines.

```bash
# Run tests in Slicer Python console
import SegmentEditorAdaptiveBrush
SegmentEditorAdaptiveBrush.SegmentEditorAdaptiveBrushTest().runTest()
```

## References

- [ITK-SNAP Adaptive Brush](https://www.itksnap.org/) - Original inspiration
- [3D Slicer Segment Editor](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html)
- [SlicerSegmentEditorExtraEffects](https://github.com/lassoan/SlicerSegmentEditorExtraEffects)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read [CLAUDE.md](CLAUDE.md) for development guidelines and see [ROADMAP.md](ROADMAP.md) for planned features.
