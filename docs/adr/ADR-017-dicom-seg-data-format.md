# ADR-017: DICOM SEG Data Format Standard

## Status

**Accepted** (2026-01-28)

Updated 2026-01-29: Added SOP Class details and custom LABELMAP plugin solution (ADR-019).

## Context

The project currently stores segmentation results in Slicer's native `.seg.nrrd` format. While this works for internal use, it has limitations:

1. **No compatibility with DICOM-based tools** - CrossSegmentationExplorer, OHIF, and other clinical tools require DICOM SEG
2. **Limited metadata** - `.seg.nrrd` doesn't carry standardized medical metadata (patient ID, study UID, terminology codes)
3. **No reference volume linkage** - Segmentations don't formally reference their source volumes
4. **Clinical workflow barrier** - Results can't be easily shared with PACS or clinical review systems

### Current Format

```
optimization_results/<run>/
├── segmentations/
│   ├── trial_000.seg.nrrd    # Slicer native format
│   ├── trial_001.seg.nrrd
│   └── ...
└── results.json              # Metadata separate from segmentation
```

### DICOM SEG Encoding Types

The DICOM standard supports multiple segmentation encoding types:

| Type | Storage | Overlapping Segments | Use Case |
|------|---------|---------------------|----------|
| **BINARY** | N segments × M slices = N×M binary frames | ✓ Yes | Single segment, overlapping segments |
| **FRACTIONAL** | Continuous values (0.0-1.0) | ✓ Yes | Probabilistic segmentation |
| **LABELMAP** (Sup 243) | Single integer per voxel | ✗ No | Multi-segment, non-overlapping |

**BINARY encoding is inefficient for multi-segment segmentations:**
- 10 segments × 256×256×100 volume = ~82MB uncompressed
- Each segment stored as separate binary frames

**LABELMAP encoding (DICOM Supplement 243, DICOM 2025b) is efficient:**
- Same volume with 10 segments = ~6.5MB uncompressed
- Single integer per voxel (like traditional labelmap)
- Full segment metadata preserved (names, colors, terminology)
- Supports lossless compression (RLE, JPEG2000, JPEGLS)

### DICOM SOP Classes

BINARY and FRACTIONAL encodings use a different SOP Class UID than LABELMAP:

| Encoding | SOP Class UID | Name |
|----------|---------------|------|
| BINARY | `1.2.840.10008.5.1.4.1.1.66.4` | Segmentation Storage |
| FRACTIONAL | `1.2.840.10008.5.1.4.1.1.66.4` | Segmentation Storage |
| **LABELMAP** | `1.2.840.10008.5.1.4.1.1.66.7` | Label Map Segmentation Storage |

This distinction is important because:
1. **Older tools only support the BINARY/FRACTIONAL SOP Class** - including dcmqi/QuantitativeReporting
2. **highdicom correctly uses the LABELMAP SOP Class** - when creating LABELMAP-encoded segmentations
3. **OHIF v3.11+ supports both SOP Classes** - full LABELMAP support added in 2024

### Tool Compatibility

| Tool | LABELMAP Support | Compression | Notes |
|------|-----------------|-------------|-------|
| **DICOM 2025b** | ✓ Standard | ✓ RLE, JPEG2000, JPEGLS, Deflate | Supplement 243 |
| **OHIF v3.11** | ✓ Full | ✓ | "Revolutionizing how we handle large segmentations" |
| **highdicom** | ✓ Full | ✓ RLE, JPEG2000, JPEGLS | Python library for reading/writing |
| **dcmqi/Slicer** | ✗ Not yet | ✗ Limited | Only supports SOP Class `1.2.840.10008.5.1.4.1.1.66.4` ([issue #518](https://github.com/QIICR/dcmqi/issues/518)) |
| **DICOMLabelMapSegPlugin** | ✓ Full | ✓ | Our custom plugin using highdicom (ADR-019) |
| **CrossSegmentationExplorer** | ✓ Via custom plugin | ✓ | Works with DICOMLabelMapSegPlugin |

**Note:** dcmqi issue [#518](https://github.com/QIICR/dcmqi/issues/518) tracks native LABELMAP support. Until then, use our custom DICOMLabelMapSegPlugin (ADR-019) for loading LABELMAP files in Slicer.

### CrossSegmentationExplorer Requirements

CrossSegmentationExplorer (reference: `_reference/CrossSegmentationExplorer/`) requires:
- DICOM volumes with `DICOM.instanceUIDs` attribute
- DICOM SEG files with `ReferencedSeriesSequence` pointing to source volume
- Slicer DICOM database for querying relationships
- `TerminologyEntry` tags for anatomical structure names

## Decision

Adopt **DICOM SEG with LABELMAP encoding (Supplement 243)** as the primary segmentation storage format, using **highdicom** for creation instead of QuantitativeReporting/dcmqi.

### Why LABELMAP Encoding?

| Aspect | BINARY | LABELMAP |
|--------|--------|----------|
| Multi-segment efficiency | Poor (N× frames) | Excellent (1× frames) |
| Compression | Limited | Full (RLE, JPEG2000, JPEGLS) |
| OHIF v3.11 | Supported | Optimized |
| Overlapping segments | ✓ Yes | ✗ No |
| Our use case | Overkill | Perfect fit |

Our optimization trials produce **non-overlapping segments** (tumor, tissue, etc.), making LABELMAP the optimal choice.

### Why highdicom Instead of dcmqi?

| Aspect | dcmqi/QuantitativeReporting | highdicom |
|--------|----------------------------|-----------|
| LABELMAP support | ✗ Not implemented | ✓ Full |
| Compression | ✗ Limited/pending | ✓ RLE, JPEG2000, JPEGLS |
| Python API | CLI wrapper | Native Python |
| Maintenance | Extension dependency | pip install |
| DICOM 2025b | Behind | Current |

### Synthetic DICOM for SampleData

Since SampleData volumes (MRHead, MRBrainTumor1, etc.) are not DICOM, we create **synthetic DICOM series**:

```python
class DicomManager:
    """Manage DICOM database operations for optimization results."""

    def create_synthetic_dicom(self, volume_node, patient_id: str, ...) -> str:
        """
        Convert SampleData volume to DICOM series.

        Uses Slicer's DICOMScalarVolumePlugin for volume export.
        Creates synthetic PatientID, StudyInstanceUID, SeriesInstanceUID.

        Returns: SeriesInstanceUID
        """

    def export_segmentation_as_dicom_seg(
        self,
        segmentation_node,
        reference_volume_node,
        series_description: str,
        output_dir: Path,
        compression: str = "JPEG2000Lossless",
        segment_metadata: dict | None = None,
    ) -> str:
        """
        Export segmentation as DICOM SEG with LABELMAP encoding.

        Uses highdicom for DICOM SEG creation with:
        - LABELMAP segmentation type (Supplement 243)
        - Lossless compression (JPEG2000 default)
        - Full segment metadata (names, colors, terminology)
        - ReferencedSeriesSequence linking to source volume

        Returns: SeriesInstanceUID
        """
```

### New Data Structure

```
optimization_results/<run>/
├── dicom/
│   ├── volume/                    # Synthetic DICOM of source volume
│   │   ├── 0001.dcm
│   │   ├── 0002.dcm
│   │   └── ...
│   └── segmentations/
│       ├── trial_000.dcm          # DICOM SEG files
│       ├── trial_001.dcm
│       └── ...
├── results.json                   # Updated with DICOM UIDs
└── config.yaml
```

### Updated results.json Schema

```json
{
  "dicom": {
    "patient_id": "AdaptiveBrush_MRBrainTumor1",
    "study_instance_uid": "1.2.826.0.1.3680043.8.498.xxx",
    "volume_series_uid": "1.2.826.0.1.3680043.8.498.xxx",
    "volume_description": "MRBrainTumor1"
  },
  "trials": [
    {
      "trial_number": 0,
      "params": {...},
      "value": 0.95,
      "dicom_series_uid": "1.2.826.0.1.3680043.8.498.xxx",
      "dicom_seg_path": "dicom/segmentations/trial_000.dcm"
    }
  ]
}
```

### Metadata Mapping

| Optimization Concept | DICOM Tag | Example Value |
|---------------------|-----------|---------------|
| Sample data name | PatientID | `AdaptiveBrush_MRBrainTumor1` |
| Run name | StudyDescription | `Tumor_Optimization_2026-01-28` |
| Trial + Algorithm | SeriesDescription | `trial_042_watershed` |
| Segment name | SegmentLabel | `Tumor` |
| Algorithm params | PrivateTag or SegmentDescription | JSON-encoded params |

### Gold Standard Storage

Gold standards also migrate to DICOM SEG:

```
GoldStandards/<name>/
├── dicom/
│   ├── volume/           # Synthetic DICOM (if from SampleData)
│   └── gold.dcm          # DICOM SEG
└── metadata.json         # Extended with DICOM UIDs
```

## Implementation

### Dependencies

**Python packages (install in Slicer Python):**
- **highdicom** - DICOM SEG creation with LABELMAP support and compression
- `pydicom` - DICOM file reading/writing (bundled with Slicer)
- `DICOMLib` - Slicer DICOM utilities (bundled)

**No longer required:**
- ~~QuantitativeReporting~~ - Replaced by highdicom for SEG export

### Installation

```python
# In Slicer Python console or at module startup
import slicer
slicer.util.pip_install("highdicom")
```

### Compression Options

| Transfer Syntax | Compression | Speed | Support | Python Dependencies |
|-----------------|-------------|-------|---------|---------------------|
| `JPEGLSLossless` | Best | Fast | Good | pylibjpeg, pylibjpeg-libjpeg |
| `JPEG2000Lossless` | Excellent | Slow | Excellent | pylibjpeg, pylibjpeg-openjpeg |
| `RLELossless` | Good | Fast | Universal | pylibjpeg or pydicom RLE plugin |
| `ExplicitVRLittleEndian` | None | N/A | Universal | None (default fallback) |

**Default: `ExplicitVRLittleEndian`** - Works without additional dependencies.

**For optimal compression**, install the pylibjpeg family:
```python
slicer.util.pip_install("pylibjpeg pylibjpeg-openjpeg")  # For JPEG2000
slicer.util.pip_install("pylibjpeg pylibjpeg-libjpeg")   # For JPEG-LS
```

Then use `JPEG2000Lossless` for best compression ratio with excellent viewer compatibility.

### DicomManager Class

```python
class DicomManager:
    """Manage DICOM database operations for optimization results."""

    # Compression transfer syntax UIDs
    TRANSFER_SYNTAXES = {
        "RLELossless": "1.2.840.10008.1.2.5",
        "JPEG2000Lossless": "1.2.840.10008.1.2.4.90",
        "JPEGLSLossless": "1.2.840.10008.1.2.4.80",
        "ExplicitVRLittleEndian": "1.2.840.10008.1.2.1",
    }

    def export_segmentation_as_dicom_seg(
        self,
        segmentation_node,
        reference_volume_node,
        series_description: str,
        output_dir: Path,
        compression: str = "JPEG2000Lossless",
        segment_metadata: dict | None = None,
    ) -> str:
        """
        Export segmentation as DICOM SEG with LABELMAP encoding.

        Uses highdicom for efficient multi-segment storage.
        """
        import highdicom as hd
        from highdicom.seg import Segmentation, SegmentationTypeValues

        # Extract labelmap array from Slicer segmentation node
        labelmap_array = self._get_labelmap_from_segmentation(segmentation_node)

        # Build segment descriptions from Slicer segment metadata
        segment_descriptions = self._build_segment_descriptions(segmentation_node)

        # Create DICOM SEG with LABELMAP encoding
        seg = Segmentation(
            source_images=self._get_source_images(reference_volume_node),
            pixel_array=labelmap_array,
            segmentation_type=SegmentationTypeValues.LABELMAP,
            segment_descriptions=segment_descriptions,
            series_description=series_description,
            transfer_syntax_uid=self.TRANSFER_SYNTAXES[compression],
            ...
        )

        output_path = output_dir / "seg.dcm"
        seg.save_as(str(output_path))
        return seg.SeriesInstanceUID

    def ensure_database_initialized(self) -> bool:
        """Ensure DICOM database is available."""

    def import_dicom_folder(self, folder: Path) -> list[str]:
        """Import DICOM files to database, return series UIDs."""

    def get_segmentations_for_volume(self, volume_series_uid: str) -> list[str]:
        """Find all DICOM SEG referencing a volume."""

    def load_segmentation_by_uid(self, series_uid: str) -> vtkMRMLSegmentationNode:
        """Load DICOM SEG from database."""
```

### Migration Strategy

1. **No backwards compatibility** - New code only supports DICOM
2. **Migration script** (`scripts/migrate_to_dicom.py`) for existing data
3. **Old files preserved** - User deletes manually after verification
4. **Migration report** - Documents what was converted

## Consequences

### Positive

- **CrossSegmentationExplorer compatibility** - Multi-trial comparison works
- **OHIF v3.11 optimized** - LABELMAP encoding specifically optimized for OHIF
- **Clinical interoperability** - Results can go to PACS
- **Efficient multi-segment storage** - LABELMAP encoding ~10-15x smaller than BINARY
- **Lossless compression** - JPEG2000/JPEGLS further reduces size
- **Standardized metadata** - SNOMED codes, proper UIDs, segment colors
- **Volume linkage** - Segmentations formally reference source
- **Future-proof** - DICOM 2025b standard

### Negative

- **No overlapping segments** - LABELMAP requires non-overlapping (acceptable for our use case)
- **Complexity** - DICOM database required
- **New dependency** - highdicom (pip install)
- **Migration effort** - Existing data must be converted

### Trade-offs

| Aspect | .seg.nrrd | DICOM BINARY | DICOM LABELMAP |
|--------|-----------|--------------|----------------|
| Multi-segment size | Small | Very large | Small |
| Compression | gzip | Limited | JPEG2000/JPEGLS |
| Overlapping segments | Yes | Yes | No |
| OHIF support | No | Yes | Optimized |
| Clinical PACS | No | Yes | Yes |
| Slicer export | Native | QuantitativeReporting | highdicom |

### File Size Comparison (256×256×100 volume, 10 segments)

| Format | Uncompressed | Compressed |
|--------|--------------|------------|
| .seg.nrrd | ~6.5MB | ~500KB (gzip) |
| DICOM SEG BINARY | ~82MB | ~8MB (RLE) |
| DICOM SEG LABELMAP | ~6.5MB | ~500KB (JPEG2000) |

## Alternatives Considered

### Keep .seg.nrrd with DICOM Export on Demand

**Rejected**: Creates two code paths, increases maintenance burden. User requested full DICOM native.

### Use DICOM SEG BINARY Encoding

**Rejected**: Inefficient for multi-segment segmentations. 10 segments = 10× storage.

### Use QuantitativeReporting/dcmqi

**Rejected**: Does not support LABELMAP encoding (Supplement 243). Does not support compression. highdicom provides better DICOM 2025b support.

### Support Both Formats

**Rejected**: User explicitly requested no backwards compatibility. Clean break is simpler.

### Use NIFTI Instead

**Rejected**: NIFTI lacks metadata richness of DICOM. Doesn't solve OHIF/CrossSegmentationExplorer compatibility.

## References

### DICOM Standards

- [DICOM Supplement 243: Label Map Segmentation](https://www.dicomstandard.org/news-dir/current/docs/sups/sup243.pdf) - LABELMAP encoding specification
- [DICOM SEG Specification](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.html)
- [DICOM 2025b Standard](https://www.dicomstandard.org/news-dir/progress/index.html)

### Libraries

- [highdicom Documentation](https://highdicom.readthedocs.io/en/latest/seg.html) - Python DICOM SEG with LABELMAP support
- [highdicom GitHub](https://github.com/ImagingDataCommons/highdicom)
- [pydicom](https://pydicom.github.io/)

### Viewers

- [OHIF v3.11 Release Notes](https://ohif.org/release-notes/3p11/) - LABELMAP support announcement
- [CrossSegmentationExplorer](https://github.com/ImagingDataCommons/CrossSegmentationExplorer)

### Slicer

- [Slicer DICOM Module](https://slicer.readthedocs.io/en/latest/user_guide/modules/dicom.html)

### Background

- [DICOM SEG Optimization Project Week](https://projectweek.na-mic.org/PW38_2023_GranCanaria/Projects/DICOMSEG/) - Discussion of SEG efficiency issues
- [dcmqi compression issue #244](https://github.com/QIICR/dcmqi/issues/244) - Why dcmqi compression is pending
