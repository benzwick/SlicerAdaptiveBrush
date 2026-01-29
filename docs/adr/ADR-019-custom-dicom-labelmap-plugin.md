# ADR-019: Custom DICOM Label Map Segmentation Plugin

## Status

**Accepted** (2026-01-29)

## Context

### DICOM Segmentation SOP Classes

DICOM defines two SOP Classes for segmentation storage:

| SOP Class UID | Name | Encodings |
|---------------|------|-----------|
| `1.2.840.10008.5.1.4.1.1.66.4` | Segmentation Storage | BINARY, FRACTIONAL |
| `1.2.840.10008.5.1.4.1.1.66.7` | Label Map Segmentation Storage | LABELMAP |

The LABELMAP encoding (defined in DICOM Supplement 243) is significantly more efficient for multi-segment segmentations:
- Single integer per voxel instead of N binary frames
- 10-15x smaller file sizes for typical cases
- Supports lossless compression (RLE, JPEG2000, JPEGLS)

### The Problem

**highdicom** correctly creates DICOM SEG files with LABELMAP encoding using the Label Map Segmentation Storage SOP Class (`1.2.840.10008.5.1.4.1.1.66.7`).

**dcmqi** (used by Slicer's DICOMSegmentationPlugin and QuantitativeReporting extension) only supports the Segmentation Storage SOP Class (`1.2.840.10008.5.1.4.1.1.66.4`).

This creates a compatibility gap:
- We export LABELMAP files with highdicom
- Slicer cannot load them via standard DICOM infrastructure
- CrossSegmentationExplorer (CSE) cannot display our optimization results

### Upstream Status

- **dcmqi issue [#518](https://github.com/QIICR/dcmqi/issues/518)** - Tracks LABELMAP support
  - Status: High priority, assigned to developer
  - Timeline: Unknown
- **OHIF v3.11+** - Already supports LABELMAP natively
  - [PR#5158](https://github.com/OHIF/Viewers/pull/5158) merged in 2024
  - "Revolutionizing how we handle large segmentations"

## Decision

Create a custom DICOM plugin (`DICOMLabelMapSegPlugin`) that uses highdicom for loading LABELMAP DICOM SEG files in Slicer.

### Why Not Wait for dcmqi?

1. **Unknown timeline** - dcmqi LABELMAP support has been requested but no release date
2. **Immediate need** - Our optimization workflow requires CSE compatibility now
3. **Simple solution** - highdicom already handles parsing, we just bridge to Slicer
4. **Temporary measure** - Plugin becomes redundant once dcmqi supports LABELMAP

### Why highdicom for Loading?

| Aspect | highdicom | Manual pydicom |
|--------|-----------|----------------|
| LABELMAP parsing | Built-in | Must implement |
| Frame geometry | Automatic | Manual calculation |
| Segment metadata | Structured API | Raw sequences |
| Maintenance | Library updates | Our responsibility |
| Correctness | Well-tested | Risk of bugs |

## Implementation

### DICOMLabelMapSegPlugin

Location: `SegmentEditorAdaptiveBrushReviewerLib/DICOMLabelMapSegPlugin.py`

```python
class DICOMLabelMapSegPlugin:
    """DICOM plugin to load Label Map Segmentation Storage using highdicom."""

    loadType = "DICOM Labelmap Segmentation (highdicom)"

    def examineForImport(self, fileLists: list[list[str]]) -> list:
        """Check if files have LABELMAP SOP Class."""
        # Returns DICOMLoadable for files with SOP Class 1.2.840.10008.5.1.4.1.1.66.7

    def load(self, loadable) -> bool:
        """Load LABELMAP SEG using highdicom and create Slicer segmentation node."""
        # 1. Read with highdicom.seg.Segmentation.from_dcm()
        # 2. Get pixel_array (already in labelmap format)
        # 3. Find/load referenced volume
        # 4. Create vtkMRMLSegmentationNode
        # 5. Set geometry from volume or DICOM
        # 6. Import labelmap to segmentation
        # 7. Apply segment metadata (names, colors)
```

### Plugin Registration

The plugin auto-registers when the Reviewer module library is imported:

```python
# In SegmentEditorAdaptiveBrushReviewerLib/__init__.py
from .DICOMLabelMapSegPlugin import register_plugin as register_labelmap_plugin

# Auto-register when library loads
try:
    register_labelmap_plugin()
except Exception:
    pass  # Silently ignore outside Slicer
```

### Loading Workflow

1. User imports DICOM folder containing LABELMAP SEG files
2. DICOM browser calls `examineForImport()` on all plugins
3. DICOMLabelMapSegPlugin identifies LABELMAP files (high confidence: 0.95)
4. User loads the segmentation
5. Plugin uses highdicom to parse and create Slicer nodes

## Consequences

### Positive

- **CSE compatibility** - CrossSegmentationExplorer works with our LABELMAP files
- **Small file sizes** - LABELMAP encoding preserved
- **OHIF compatibility** - Files also work in OHIF v3.11+
- **No highdicom changes needed** - Uses existing API
- **Automatic** - Plugin registers on module load
- **Metadata preserved** - Colors, names, terminology maintained

### Negative

- **Additional code to maintain** - ~300 lines of plugin code
- **highdicom dependency** - Requires pip install (auto-installed)
- **May become redundant** - When dcmqi adds LABELMAP support
- **Testing burden** - Need tests for various LABELMAP files

### Migration Path

When dcmqi adds LABELMAP support:

1. **Test interoperability** - Verify dcmqi-loaded files match plugin-loaded
2. **Deprecate plugin** - Mark as deprecated, prefer native support
3. **Remove plugin** - In future release when dcmqi is widely deployed

## Alternatives Considered

### Wait for dcmqi

**Rejected:** Unknown timeline, blocks our optimization workflow indefinitely.

### Use BINARY Encoding

**Rejected:** 10-15x larger files, defeats purpose of LABELMAP encoding. Would require changes to DicomManager export code.

### Fork dcmqi

**Rejected:** Maintenance burden, upstream contribution preferred. dcmqi is complex C++ code.

### Convert LABELMAP to BINARY on Import

**Rejected:** Lossy conversion (increases file size), doesn't solve the loading problem (still need to read LABELMAP first).

### External Conversion Tool

**Rejected:** Breaks workflow, requires manual steps outside Slicer.

## References

### DICOM Standards

- [DICOM Supplement 243: Label Map Segmentation](https://www.dicomstandard.org/news-dir/current/docs/sups/sup243.pdf)
- [Segmentation IOD (C.8.20)](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.html)

### Libraries

- [highdicom Documentation](https://highdicom.readthedocs.io/en/latest/seg.html)
- [highdicom GitHub](https://github.com/ImagingDataCommons/highdicom)
- [dcmqi GitHub](https://github.com/QIICR/dcmqi)
- [dcmqi issue #518](https://github.com/QIICR/dcmqi/issues/518) - LABELMAP support tracking

### Related ADRs

- [ADR-017: DICOM SEG Data Format Standard](ADR-017-dicom-seg-data-format.md) - Why we use LABELMAP encoding
- [ADR-018: CrossSegmentationExplorer Integration](ADR-018-cross-segmentation-explorer-integration.md) - CSE workflow

### Viewers

- [OHIF v3.11 LABELMAP Support](https://ohif.org/release-notes/3p11/)
- [OHIF PR#5158](https://github.com/OHIF/Viewers/pull/5158) - LABELMAP implementation
