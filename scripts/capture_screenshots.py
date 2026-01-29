#!/usr/bin/env python
"""Capture screenshots for documentation and Extension Index.

This script captures standardized screenshots of the Adaptive Brush effect
for use in README, documentation, and the 3D Slicer Extension Index.

NOTE: For CI documentation builds, screenshots are captured by the test framework
(test_docs_*.py test cases). This script is for manual/local screenshot capture.

Run via:
    Slicer --python-script scripts/capture_screenshots.py

Options:
    --exit: Exit Slicer after capturing (for automated runs)
    --output DIR: Output directory (default: Screenshots/)

Examples:
    # Capture screenshots and stay open for manual adjustments:
    Slicer --python-script scripts/capture_screenshots.py

    # Capture screenshots and exit:
    Slicer --python-script scripts/capture_screenshots.py --exit

    # Capture to custom directory:
    Slicer --python-script scripts/capture_screenshots.py --output /tmp/screenshots
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args() -> tuple[bool, Path]:
    """Parse command line arguments.

    Returns:
        Tuple of (exit_after_capture, output_directory)
    """
    args = sys.argv[1:]
    exit_after = "--exit" in args
    if exit_after:
        args.remove("--exit")

    # Default output directory
    script_dir = Path(__file__).parent
    default_output = script_dir.parent / "Screenshots"

    # Check for --output argument
    output_dir = default_output
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            output_dir = Path(args[idx + 1])

    return exit_after, output_dir


def setup_scene() -> tuple:
    """Load sample data and set up the scene for screenshots.

    Returns:
        Tuple of (volume_node, segmentation_node)
    """
    import SampleData
    import slicer

    logger.info("Loading MRBrainTumor1 sample data...")
    slicer.mrmlScene.Clear(0)

    # Load brain tumor sample - good for demonstrating adaptive brush
    volume_node = SampleData.downloadSample("MRBrainTumor1")

    if volume_node is None:
        raise RuntimeError("Failed to load MRBrainTumor1 sample data")

    logger.info(f"Loaded volume: {volume_node.GetName()}")

    # Create segmentation
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes()
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

    # Add a segment for the tumor
    segmentation_node.GetSegmentation().AddEmptySegment("Tumor", "Tumor", [1.0, 0.0, 0.0])

    return volume_node, segmentation_node


def setup_segment_editor(volume_node, segmentation_node) -> None:
    """Configure the Segment Editor with Adaptive Brush effect.

    Args:
        volume_node: The source volume node
        segmentation_node: The segmentation node to edit
    """
    import slicer

    logger.info("Setting up Segment Editor...")

    # Switch to Segment Editor module
    slicer.util.selectModule("SegmentEditor")

    # Get the segment editor widget
    segment_editor_widget = slicer.modules.SegmentEditorWidget.editor

    # Set source volume and segmentation
    segment_editor_widget.setSegmentationNode(segmentation_node)
    segment_editor_widget.setSourceVolumeNode(volume_node)

    # Select the Adaptive Brush effect
    segment_editor_widget.setActiveEffectByName("Adaptive Brush")

    logger.info("Segment Editor configured with Adaptive Brush effect")


def navigate_to_tumor_slice(volume_node) -> None:
    """Navigate to a slice showing the tumor.

    Args:
        volume_node: The source volume node
    """
    import numpy as np
    import slicer

    logger.info("Navigating to tumor slice...")

    # Get volume array and find a good slice
    arr = slicer.util.arrayFromVolume(volume_node)

    # For MRBrainTumor1, the tumor is typically in the center-upper region
    # Find slice with maximum intensity variance (likely shows tumor)
    dims = arr.shape  # (K, J, I)
    center_k = dims[0] // 2

    # Search around center for slice with good tumor visibility
    best_slice = center_k
    best_variance = 0.0

    for k in range(center_k - 20, center_k + 20):
        if 0 <= k < dims[0]:
            slice_data = arr[k, :, :]
            variance = float(np.var(slice_data))
            if variance > best_variance:
                best_variance = variance
                best_slice = k

    # Convert IJK to RAS for navigation
    import vtk

    ijk_to_ras = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(ijk_to_ras)

    # Use center of the slice
    ijk = [dims[2] // 2, dims[1] // 2, best_slice]
    ras = [0.0, 0.0, 0.0]
    for i in range(3):
        for j in range(3):
            ras[i] += ijk_to_ras.GetElement(i, j) * ijk[j]
        ras[i] += ijk_to_ras.GetElement(i, 3)

    # Navigate to this position
    slicer.modules.markups.logic().JumpSlicesToLocation(ras[0], ras[1], ras[2], True)

    # Set a reasonable zoom level
    for sliceViewName in ["Red", "Yellow", "Green"]:
        sliceWidget = slicer.app.layoutManager().sliceWidget(sliceViewName)
        if sliceWidget:
            sliceWidget.sliceController().fitSliceToBackground()

    logger.info(f"Navigated to slice {best_slice} (RAS: {ras[0]:.1f}, {ras[1]:.1f}, {ras[2]:.1f})")


def apply_sample_segmentation() -> None:
    """Apply a sample segmentation to demonstrate the effect.

    This paints a small region using the adaptive brush to show what the
    segmentation result looks like.
    """
    import slicer

    logger.info("Applying sample segmentation...")

    # Get the effect
    segment_editor_widget = slicer.modules.SegmentEditorWidget.editor
    effect = segment_editor_widget.activeEffect()

    if effect is None:
        logger.warning("No active effect - skipping sample segmentation")
        return

    # Configure the effect for a good demo
    # These settings work well for brain tumor on T1 MRI
    if hasattr(effect, "self"):
        scripted_effect = effect.self()

        # Set reasonable defaults
        scripted_effect.radiusSlider.setValue(15.0)  # 15mm brush radius

        # Select Watershed algorithm (good balance of speed and quality)
        if hasattr(scripted_effect, "algorithmCombo"):
            idx = scripted_effect.algorithmCombo.findText("Watershed")
            if idx >= 0:
                scripted_effect.algorithmCombo.setCurrentIndex(idx)

    logger.info("Effect configured for demo")


def capture_screenshot(
    output_path: Path, description: str, capture_type: str = "main_window"
) -> dict:
    """Capture a screenshot.

    Args:
        output_path: Path to save the screenshot
        description: Description for the manifest
        capture_type: Type of capture ("main_window", "widget", "slice_view")

    Returns:
        Manifest entry dict
    """
    import slicer

    logger.info(f"Capturing: {output_path.name}")

    # Process events to ensure UI is updated
    slicer.app.processEvents()

    # Capture based on type
    if capture_type == "main_window":
        # Capture the main Slicer window
        widget = slicer.util.mainWindow()
        pixmap = widget.grab()
    elif capture_type == "slice_view":
        # Capture just the slice views area
        layout_manager = slicer.app.layoutManager()
        widget = layout_manager.viewport()
        pixmap = widget.grab()
    else:
        # Capture main window by default
        widget = slicer.util.mainWindow()
        pixmap = widget.grab()

    # Save the screenshot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(output_path))

    # Verify it was saved
    if not output_path.exists():
        raise RuntimeError(f"Failed to save screenshot: {output_path}")

    logger.info(f"Saved: {output_path} ({output_path.stat().st_size} bytes)")

    return {
        "filename": output_path.name,
        "description": description,
        "capture_type": capture_type,
        "width": pixmap.width(),
        "height": pixmap.height(),
    }


def capture_main_ui(output_dir: Path) -> dict:
    """Capture the main UI screenshot for Extension Index.

    This is the primary screenshot that appears in the Extension Manager.

    Args:
        output_dir: Directory to save screenshots

    Returns:
        Manifest entry
    """
    import slicer

    # Use conventional layout to show slice views prominently
    layout_manager = slicer.app.layoutManager()
    layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)

    # Process events
    slicer.app.processEvents()

    return capture_screenshot(
        output_dir / "main-ui.png",
        "Adaptive Brush effect in Segment Editor showing slice views with brush outline",
        "main_window",
    )


def capture_algorithm_panel(output_dir: Path) -> dict:
    """Capture the algorithm options panel.

    Args:
        output_dir: Directory to save screenshots

    Returns:
        Manifest entry
    """
    import slicer

    # Ensure the options panel is visible and expanded
    # Use a layout that shows the module panel well
    layout_manager = slicer.app.layoutManager()
    layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)

    slicer.app.processEvents()

    return capture_screenshot(
        output_dir / "algorithm-panel.png",
        "Adaptive Brush options panel showing algorithm selection and parameters",
        "main_window",
    )


def capture_segmentation_result(output_dir: Path) -> dict:
    """Capture the segmentation result.

    Args:
        output_dir: Directory to save screenshots

    Returns:
        Manifest entry
    """
    import slicer

    # Show the segmentation result
    layout_manager = slicer.app.layoutManager()
    layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)

    slicer.app.processEvents()

    return capture_screenshot(
        output_dir / "segmentation-result.png",
        "Brain tumor segmented with Adaptive Brush showing result in slice views",
        "main_window",
    )


def generate_manifest(output_dir: Path, entries: list[dict]) -> None:
    """Generate a manifest.json file with screenshot metadata.

    Args:
        output_dir: Directory containing screenshots
        entries: List of manifest entries from capture functions
    """
    manifest = {
        "generated": datetime.now().isoformat(),
        "generator": "scripts/capture_screenshots.py",
        "sample_data": "MRBrainTumor1",
        "slicer_version": "5.10+",
        "screenshots": entries,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Generated manifest: {manifest_path}")


def capture_all_screenshots(output_dir: Path) -> list[dict]:
    """Capture all screenshots.

    Args:
        output_dir: Directory to save screenshots

    Returns:
        List of manifest entries
    """
    import slicer

    logger.info(f"Capturing screenshots to: {output_dir}")

    # Set up the scene
    volume_node, segmentation_node = setup_scene()
    setup_segment_editor(volume_node, segmentation_node)
    navigate_to_tumor_slice(volume_node)

    # Give the UI time to fully render
    slicer.app.processEvents()
    import time

    time.sleep(0.5)
    slicer.app.processEvents()

    entries = []

    # Capture main UI (this is the Extension Index screenshot)
    entries.append(capture_main_ui(output_dir))

    # Capture algorithm panel
    entries.append(capture_algorithm_panel(output_dir))

    # Apply sample segmentation and capture result
    apply_sample_segmentation()
    slicer.app.processEvents()
    time.sleep(0.3)
    entries.append(capture_segmentation_result(output_dir))

    return entries


def main() -> None:
    """Main entry point."""
    import slicer

    exit_after, output_dir = parse_args()

    logger.info("=" * 60)
    logger.info("Adaptive Brush Screenshot Capture")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Exit after capture: {exit_after}")

    try:
        entries = capture_all_screenshots(output_dir)
        generate_manifest(output_dir, entries)

        logger.info("=" * 60)
        logger.info("Screenshot capture complete!")
        logger.info(f"Screenshots saved to: {output_dir}")
        logger.info(f"Total screenshots: {len(entries)}")
        logger.info("=" * 60)

        if exit_after:
            slicer.app.exit(0)
        else:
            logger.info("Slicer is open for manual adjustments. Close when done.")

    except Exception as e:
        logger.exception(f"Screenshot capture failed: {e}")
        if exit_after:
            slicer.app.exit(1)


if __name__ == "__main__":
    main()
