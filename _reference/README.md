# Reference Code

This folder contains reference implementations and source code for development.

**This folder is gitignored** - each developer sets up their own references based on their needs and local paths.

## Purpose

- Study existing implementations when developing new features
- Cross-reference Slicer APIs and patterns
- Compare algorithm implementations

## Setup

Use the `/setup-references` skill to configure this folder:

```
/setup-references
```

Your choices are saved in `.setup.json` (gitignored). See `.setup.json.example` for the format.

### Manual Setup

### Option 1: Clone repositories

```bash
cd _reference
git clone --depth 1 https://github.com/lassoan/SlicerSegmentEditorExtraEffects.git
git clone --depth 1 https://github.com/QIICR/QuantitativeReporting.git
git clone --depth 1 https://github.com/ImagingDataCommons/SlicerCrossSegmentationExplorer.git CrossSegmentationExplorer
```

### Option 2: Symlink to existing local copies

```bash
cd _reference
ln -s /path/to/your/SlicerSegmentEditorExtraEffects .
ln -s /path/to/your/Slicer SlicerSource
```

## Available References

| Reference | Description | Recommended Setup |
|-----------|-------------|-------------------|
| `SlicerSource/` | 3D Slicer source code | Symlink to local clone |
| `SlicerSegmentEditorExtraEffects/` | Extra segment editor effects | Clone |
| `QuantitativeReporting/` | DICOM SEG handling | Clone |
| `CrossSegmentationExplorer/` | Segmentation comparison | Clone |

## For Claude

When exploring unfamiliar Slicer APIs or patterns, check this folder for reference implementations:

- **Segment Editor effects**: `SlicerSegmentEditorExtraEffects/`
- **DICOM handling**: `QuantitativeReporting/`
- **Segmentation comparison**: `CrossSegmentationExplorer/`
- **Slicer Python API**: `SlicerSource/Base/Python/slicer/`
- **MRML nodes**: `SlicerSource/Libs/MRML/Core/`
