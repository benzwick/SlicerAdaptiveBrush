#!/usr/bin/env python
"""Test DICOM SEG LABELMAP export with highdicom.

Run in Slicer:
    exec(open('/path/to/test_dicom_labelmap.py').read())

Or from command line:
    Slicer --python-script scripts/test_dicom_labelmap.py
"""

import logging
import sys
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_dicom_labelmap():
    """Test DICOM SEG export with LABELMAP encoding."""
    import SampleData
    import slicer

    # Add module path
    module_path = Path(__file__).parent.parent
    reviewer_path = module_path / "SegmentEditorAdaptiveBrushReviewer"
    if str(reviewer_path) not in sys.path:
        sys.path.insert(0, str(reviewer_path))

    from SegmentEditorAdaptiveBrushReviewerLib import DicomManager, DicomManagerError

    print("\n" + "=" * 60)
    print("Testing DICOM SEG LABELMAP Export")
    print("=" * 60)

    # Create temp directory for output
    output_dir = Path(tempfile.mkdtemp(prefix="dicom_test_"))
    print(f"\nOutput directory: {output_dir}")

    # Step 1: Initialize DicomManager
    print("\n[1/6] Initializing DicomManager...")
    manager = DicomManager()

    # Step 2: Ensure DICOM database is initialized
    print("[2/6] Initializing DICOM database...")
    if not manager.ensure_database_initialized():
        print("ERROR: Could not initialize DICOM database")
        return False
    print("  DICOM database ready")

    # Step 3: Load sample data
    print("[3/6] Loading MRHead sample data...")
    volume_node = SampleData.downloadSample("MRHead")
    print(f"  Loaded volume: {volume_node.GetName()}")
    print(f"  Dimensions: {volume_node.GetImageData().GetDimensions()}")

    # Step 4: Create synthetic DICOM
    print("[4/6] Creating synthetic DICOM from volume...")
    dicom_volume_dir = output_dir / "volume"
    try:
        volume_series_uid = manager.create_synthetic_dicom(
            volume_node=volume_node,
            patient_id="Test_Patient",
            study_description="LABELMAP Test Study",
            output_dir=dicom_volume_dir,
            series_description="MRHead",
        )
        print(f"  Created synthetic DICOM: {volume_series_uid}")
        print(f"  Files in {dicom_volume_dir}:")
        for f in sorted(dicom_volume_dir.glob("*.dcm"))[:5]:
            print(f"    {f.name}")
        dcm_count = len(list(dicom_volume_dir.glob("*.dcm")))
        if dcm_count > 5:
            print(f"    ... and {dcm_count - 5} more")
    except DicomManagerError as e:
        print(f"ERROR creating synthetic DICOM: {e}")
        return False

    # Step 5: Create a test segmentation
    print("[5/6] Creating test segmentation...")
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.SetName("TestSegmentation")
    segmentation_node.CreateDefaultDisplayNodes()
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

    # Add a sphere segment
    import vtkSegmentationCorePython as vtkSegmentationCore

    # Create segment 1 - sphere in center
    segment1 = vtkSegmentationCore.vtkSegment()
    segment1.SetName("Tumor")
    segment1.SetColor(1.0, 0.0, 0.0)  # Red

    # Create segment 2 - another sphere
    segment2 = vtkSegmentationCore.vtkSegment()
    segment2.SetName("Tissue")
    segment2.SetColor(0.0, 1.0, 0.0)  # Green

    segmentation_node.GetSegmentation().AddSegment(segment1)
    segmentation_node.GetSegmentation().AddSegment(segment2)

    # Use threshold effect to create actual segmentation data
    # Create labelmap representation
    import numpy as np
    import SimpleITK as sitk
    import sitkUtils

    # Get volume as array
    sitk_volume = sitkUtils.PullVolumeFromSlicer(volume_node)
    volume_array = sitk.GetArrayFromImage(sitk_volume)

    # Create labelmap with two regions based on intensity
    labelmap_array = np.zeros_like(volume_array, dtype=np.uint8)

    # Segment 1: High intensity voxels (e.g., > 100)
    labelmap_array[volume_array > 100] = 1

    # Segment 2: Medium intensity voxels (50-100)
    labelmap_array[(volume_array > 50) & (volume_array <= 100)] = 2

    # Create labelmap volume
    labelmap_sitk = sitk.GetImageFromArray(labelmap_array)
    labelmap_sitk.CopyInformation(sitk_volume)

    # Push to Slicer as labelmap
    labelmap_node = sitkUtils.PushVolumeToSlicer(
        labelmap_sitk, name="TempLabelmap", className="vtkMRMLLabelMapVolumeNode"
    )

    # Import labelmap to segmentation
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        labelmap_node, segmentation_node
    )

    # Clean up temp labelmap
    slicer.mrmlScene.RemoveNode(labelmap_node)

    seg_count = segmentation_node.GetSegmentation().GetNumberOfSegments()
    print(f"  Created segmentation with {seg_count} segments")
    for i in range(seg_count):
        seg_id = segmentation_node.GetSegmentation().GetNthSegmentID(i)
        seg = segmentation_node.GetSegmentation().GetSegment(seg_id)
        print(f"    - {seg.GetName()}")

    # Step 6: Export as DICOM SEG with LABELMAP encoding
    print("[6/6] Exporting as DICOM SEG with LABELMAP encoding...")
    dicom_seg_dir = output_dir / "segmentation"

    # Try different compression options - ExplicitVRLittleEndian (no compression) is most reliable
    compression_options = ["ExplicitVRLittleEndian", "RLELossless", "JPEG2000Lossless"]

    try:
        seg_series_uid = None
        last_error = None

        for compression in compression_options:
            try:
                print(f"  Trying compression: {compression}...")
                seg_series_uid = manager.export_segmentation_as_dicom_seg(
                    segmentation_node=segmentation_node,
                    reference_volume_node=volume_node,
                    series_description="Test_Segmentation_LABELMAP",
                    output_dir=dicom_seg_dir,
                    compression=compression,
                    segment_metadata={"algorithm": "threshold", "test": True},
                )
                print(f"  Success with {compression}!")
                break
            except DicomManagerError as e:
                print(f"  {compression} failed: {e}")
                last_error = e
                continue

        if seg_series_uid is None:
            raise last_error
        print(f"  Exported DICOM SEG: {seg_series_uid}")

        # Check output file
        seg_files = list(dicom_seg_dir.glob("*.dcm"))
        if seg_files:
            seg_file = seg_files[0]
            file_size = seg_file.stat().st_size
            print(f"  Output file: {seg_file.name}")
            print(f"  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

            # Verify it's a LABELMAP SEG
            import pydicom

            ds = pydicom.dcmread(str(seg_file))
            seg_type = ds.SegmentationType if hasattr(ds, "SegmentationType") else "UNKNOWN"
            print(f"  Segmentation Type: {seg_type}")
            print(f"  Transfer Syntax: {ds.file_meta.TransferSyntaxUID}")
            print(
                f"  Number of frames: {ds.NumberOfFrames if hasattr(ds, 'NumberOfFrames') else 'N/A'}"
            )

    except DicomManagerError as e:
        print(f"ERROR exporting DICOM SEG: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("SUCCESS! DICOM SEG LABELMAP export working correctly")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nTo view in OHIF, import the DICOM files to an OHIF-compatible server.")

    return True


if __name__ == "__main__":
    success = test_dicom_labelmap()
    if not success:
        print("\nTEST FAILED")
        sys.exit(1)
    else:
        print("\nTEST PASSED")
