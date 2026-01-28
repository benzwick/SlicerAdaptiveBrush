# ADR-017: DICOM SEG Data Format Standard

## Status

**Proposed** (2026-01-28)

## Context

The project currently stores segmentation results in Slicer's native `.seg.nrrd` format. While this works for internal use, it has limitations:

1. **No compatibility with DICOM-based tools** - CrossSegmentationExplorer and other clinical tools require DICOM SEG
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

### CrossSegmentationExplorer Requirements

CrossSegmentationExplorer (reference: `__reference__/CrossSegmentationExplorer/`) requires:
- DICOM volumes with `DICOM.instanceUIDs` attribute
- DICOM SEG files with `ReferencedSeriesSequence` pointing to source volume
- Slicer DICOM database for querying relationships
- `TerminologyEntry` tags for anatomical structure names

## Decision

Adopt **DICOM SEG as the primary segmentation storage format** with no backwards compatibility for `.seg.nrrd`.

### Why DICOM SEG?

| Aspect | .seg.nrrd | DICOM SEG |
|--------|-----------|-----------|
| CrossSegmentationExplorer | Not supported | Native support |
| Clinical interoperability | None | Full PACS compatibility |
| Metadata | External JSON | Embedded in file |
| Volume reference | None | ReferencedSeriesSequence |
| Terminology | Segment name only | SNOMED codes |
| File size | Smaller | Larger (but compressible) |

### Synthetic DICOM for SampleData

Since SampleData volumes (MRHead, MRBrainTumor1, etc.) are not DICOM, we create **synthetic DICOM series**:

```python
class SyntheticDicomCreator:
    """Create DICOM series from non-DICOM Slicer volumes."""

    def create_dicom_from_sample_data(self, sample_name: str, patient_id: str) -> str:
        """
        Convert SampleData volume to DICOM series.

        Creates:
        - Synthetic PatientID based on sample name
        - Generated StudyInstanceUID and SeriesInstanceUID
        - Proper geometry (spacing, orientation, dimensions)
        - Import into Slicer DICOM database

        Returns: SeriesInstanceUID
        """

    def export_segmentation_as_dicom_seg(
        self,
        seg_node,
        reference_series_uid: str,
        segment_metadata: dict
    ) -> Path:
        """
        Export segmentation node as DICOM SEG.

        Uses QuantitativeReporting/dcmqi for proper DICOM SEG encoding.
        Includes ReferencedSeriesSequence pointing to source volume.

        Returns: Path to DICOM SEG file
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

**Required Slicer extension:**
- **QuantitativeReporting** - Provides DICOM SEG export via dcmqi

**Python packages (bundled with Slicer):**
- `pydicom` - DICOM file reading/writing
- `DICOMLib` - Slicer DICOM utilities

### DicomManager Class

```python
class DicomManager:
    """Manage DICOM database operations for optimization results."""

    def __init__(self):
        self.db = slicer.dicomDatabase

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
- **Clinical interoperability** - Results can go to PACS
- **Standardized metadata** - SNOMED codes, proper UIDs
- **Volume linkage** - Segmentations formally reference source
- **Future-proof** - Standard medical imaging format

### Negative

- **Larger files** - DICOM overhead (~20-50% larger)
- **Complexity** - DICOM database required
- **Dependency** - Requires QuantitativeReporting extension
- **Migration effort** - Existing data must be converted
- **Slower export** - DICOM SEG creation more complex than .nrrd

### Trade-offs

| Aspect | .seg.nrrd Approach | DICOM Approach |
|--------|-------------------|----------------|
| Simplicity | Simple | More complex |
| Compatibility | Internal only | Universal |
| Speed | Faster | Slower |
| Storage | Smaller | Larger |
| Maintenance | Less | More |

## Alternatives Considered

### Keep .seg.nrrd with DICOM Export on Demand

**Rejected**: Creates two code paths, increases maintenance burden. User requested full DICOM native.

### Support Both Formats

**Rejected**: User explicitly requested no backwards compatibility. Clean break is simpler.

### Use NIFTI Instead

**Rejected**: NIFTI lacks metadata richness of DICOM. Doesn't solve CrossSegmentationExplorer compatibility.

## References

- [DICOM SEG Specification](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.20.html)
- [QuantitativeReporting Extension](https://github.com/QIICR/QuantitativeReporting)
- [dcmqi Documentation](https://qiicr.gitbook.io/dcmqi-guide/)
- [CrossSegmentationExplorer](https://github.com/ImagingDataCommons/CrossSegmentationExplorer)
- [Slicer DICOM Module](https://slicer.readthedocs.io/en/latest/user_guide/modules/dicom.html)
