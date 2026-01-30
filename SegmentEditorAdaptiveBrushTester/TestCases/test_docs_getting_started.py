"""Test-driven documentation: Getting Started tutorial.

This test walks through the Getting Started workflow, capturing screenshots
at each step and generating the getting_started.md documentation.

The generated documentation includes:
- Step-by-step instructions
- Screenshots captured during the test
- Verified UI state at each step

Run in Slicer:
    Slicer --python-script scripts/run_tests.py --exit docs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import slicer
from SegmentEditorAdaptiveBrushTesterLib import TestCase, TestContext, register_test

logger = logging.getLogger(__name__)


def get_docs_output_dir() -> Path:
    """Get output directory for generated documentation."""
    import os

    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace:
        return Path(workspace) / "docs" / "source" / "user_guide" / "_generated"

    # Local development - find docs folder relative to this file
    current = Path(__file__).parent
    for _ in range(5):
        candidate = current / "docs" / "source" / "user_guide" / "_generated"
        if (current / "docs").exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent

    # Default fallback
    return Path(__file__).parents[3] / "docs" / "source" / "user_guide" / "_generated"


@register_test(category="docs")
class TestDocsGettingStarted(TestCase):
    """Generate Getting Started documentation with screenshots."""

    name = "docs_getting_started"
    description = "Generate Getting Started tutorial with step-by-step screenshots"

    def __init__(self) -> None:
        super().__init__()
        self.volume_node = None
        self.segmentation_node = None
        self.segment_editor_widget = None
        self.effect = None
        self.steps: list[dict] = []
        self.output_dir: Path | None = None
        # Tumor center from gold standard (set in run method)
        self.tumor_center_ras: tuple[float, float, float] | None = None

    def setup(self, ctx: TestContext) -> None:
        """Set up for tutorial documentation."""
        logger.info("Setting up Getting Started tutorial test")

        self.output_dir = get_docs_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Clear scene
        slicer.mrmlScene.Clear(0)

        # Set up main window for screenshots
        main_window = slicer.util.mainWindow()
        main_window.resize(1920, 1080)
        slicer.app.processEvents()

        # Hide Python console for cleaner screenshots
        try:
            import qt

            python_console = main_window.findChild(qt.QDockWidget, "PythonConsoleDockWidget")
            if python_console:
                python_console.setVisible(False)
        except Exception:
            pass

        slicer.app.processEvents()

    def _record_step(
        self, ctx: TestContext, name: str, description: str, screenshot_desc: str
    ) -> None:
        """Record a tutorial step with screenshot."""
        logger.info(f"Step: {name}")

        step_num = len(self.steps) + 1
        # Convert name to slug for doc_tag (e.g., "Load Volume" -> "load_volume")
        name_slug = name.lower().replace(" ", "_").replace("-", "_")

        ctx.screenshot(
            screenshot_desc,
            doc_tags=["getting_started", name_slug],
        )

        self.steps.append(
            {
                "number": step_num,
                "name": name,
                "description": description,
                "screenshot": f"getting_started_{step_num:03d}_{name_slug}.png",
            }
        )

    def run(self, ctx: TestContext) -> None:
        """Run through Getting Started workflow."""
        logger.info("Running Getting Started tutorial")

        # =========================================
        # Step 1: Open Sample Data Module
        # =========================================
        # Show the Sample Data module where users download sample datasets
        slicer.util.selectModule("SampleData")
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Open Sample Data",
            "Go to **File → Download Sample Data** or select the Sample Data module. "
            "This provides built-in datasets for practice.",
            "Sample Data module showing available datasets",
        )

        # =========================================
        # Step 2: Load Volume
        # =========================================
        import SampleData

        self.volume_node = SampleData.downloadSample("MRBrainTumor1")
        slicer.app.processEvents()

        # Set up view - use FourUp layout consistently throughout tutorial
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        slicer.util.resetSliceViews()
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Volume Loaded",
            "After clicking on a dataset (e.g., MRBrainTumor1), the volume loads "
            "and displays in the slice views. This brain MRI contains a visible tumor.",
            "MRBrainTumor1 sample data loaded in four-up layout",
        )

        # =========================================
        # Step 3: Open Segment Editor
        # =========================================
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Open Segment Editor",
            "Navigate to the Segment Editor module using the module selector "
            "or by pressing the Segment Editor button in the toolbar.",
            "Segment Editor module opened",
        )

        # =========================================
        # Step 4: Create Segmentation
        # =========================================
        self.segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        self.segmentation_node.CreateDefaultDisplayNodes()
        self.segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(self.volume_node)

        # Get segment editor widget and configure
        segment_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        self.segment_editor_widget = segment_editor_module.editor
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setSourceVolumeNode(self.volume_node)
        slicer.app.processEvents()

        # Add a segment
        segment_id = self.segmentation_node.GetSegmentation().AddEmptySegment(
            "Tumor", "Tumor", [1.0, 0.0, 0.0]
        )
        self.segment_editor_widget.setCurrentSegmentID(segment_id)
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Create Segment",
            "Create a new segmentation and add a segment for the structure you want "
            "to segment. Click 'Add' to create a new segment.",
            "Segmentation created with Tumor segment",
        )

        # =========================================
        # Step 5: Select Adaptive Brush
        # =========================================
        self.segment_editor_widget.setActiveEffectByName("Adaptive Brush")
        self.effect = self.segment_editor_widget.activeEffect()
        slicer.app.processEvents()

        if self.effect is None:
            raise RuntimeError("Failed to activate Adaptive Brush effect")

        self._record_step(
            ctx,
            "Select Adaptive Brush",
            "Click on the Adaptive Brush effect in the effects toolbar. "
            "The options panel will show algorithm selection and brush settings.",
            "Adaptive Brush effect selected showing options panel",
        )

        # =========================================
        # Step 6: Configure Brush Settings
        # =========================================
        scripted_effect = self.effect.self()

        # Tumor center from gold standard: (-4.69, 29.50, 25.70)
        self.tumor_center_ras = (-4.69, 29.50, 25.70)

        # 1. Apply MRI T1+Gd enhancing tumor preset (configures thresholds)
        scripted_effect.applyPreset("mri_t1gd_tumor")
        slicer.app.processEvents()

        # 2. Select Watershed algorithm
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("watershed")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        # 3. Set 25mm brush radius (from gold standard optimization)
        scripted_effect.radiusSlider.value = 25.0
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Configure Settings",
            "**Preset**: Apply 'MRI T1+Gd Tumor' preset for contrast-enhanced tumors. "
            "Presets configure intensity thresholds automatically.\n\n"
            "**Algorithm**: Select Watershed - a good general-purpose choice for tumors.\n\n"
            "**Brush Radius**: Set to match your target structure (~25mm for this tumor). "
            "Adjust with **Shift + scroll wheel**.\n\n"
            "**Threshold Zone**: Inner circle where intensities are sampled. Smaller zone "
            "(30%) = stricter matching; larger zone (70%) = more variation. "
            "Adjust with **Ctrl + Shift + scroll wheel**.",
            "Brush settings configured - MRI T1+Gd preset, Watershed, 25mm radius",
        )

        # =========================================
        # Step 7: Navigate to Tumor Region
        # =========================================
        # Center all slice views on tumor centroid and zoom in 2x
        self._center_and_zoom_all_views(self.tumor_center_ras, zoom_factor=2.0)
        slicer.app.processEvents()

        # Show brush preview at the tumor center in Yellow (sagittal) view
        yellow_widget = slicer.app.layoutManager().sliceWidget("Yellow")
        view_widget = yellow_widget
        tumor_xy = self._ras_to_xy(self.tumor_center_ras, view_widget)
        if tumor_xy:
            scripted_effect._updateBrushPreview(tumor_xy, view_widget, eraseMode=False)
        view_widget.sliceView().forceRender()
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Navigate to Region",
            "Use the slice sliders or scroll wheel to navigate to the region "
            "you want to segment. The brush preview shows where painting will occur.",
            "Navigated to slice showing tumor with brush preview",
        )

        # =========================================
        # Step 8: Paint Segmentation
        # =========================================
        # Paint at tumor center (from gold standard)
        scripted_effect.paintAt(*self.tumor_center_ras)
        view_widget.sliceView().forceRender()
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Paint",
            "Click on the slice view to paint. The adaptive brush automatically "
            "detects edges and segments the region based on intensity similarity. "
            "The red overlay shows the segmented area.",
            "First paint stroke showing adaptive segmentation result",
        )

        # =========================================
        # Step 9: Continue Painting
        # =========================================
        # Add another stroke from the recipe to extend segmentation
        # Using click point 2 from brain_tumor_1 recipe: (-5.31, 25.12, 35.97)
        second_click_ras = (-5.31, 25.12, 35.97)
        scripted_effect.paintAt(*second_click_ras)
        view_widget.sliceView().forceRender()
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Refine Segmentation",
            "**Building up**: Click multiple times to extend the segmentation. "
            "Each click adds the adaptively-detected region.\n\n"
            "**Erase mode**: Hold **Ctrl** (or **Middle+Left-click**) to remove areas "
            "you over-segmented. The brush will adaptively detect what to remove.\n\n"
            "**Sampling settings** (in Advanced): Control how intensities are sampled:\n"
            "- *Mean ± Std*: Uses mean intensity with standard deviation range\n"
            "- *Percentile*: Uses intensity percentiles (more robust to outliers)\n"
            "- *Gaussian weighting*: Weights center pixels more heavily\n\n"
            "**Edge sensitivity**: Higher values stop at weaker edges; "
            "lower values allow more aggressive segmentation.",
            "Multiple strokes building up the segmentation",
        )

        # =========================================
        # Step 10: View in 3D
        # =========================================
        # Create 3D surface representation of the segmentation
        self.segmentation_node.CreateClosedSurfaceRepresentation()
        slicer.app.processEvents()

        # Ensure 3D visibility is enabled
        display_node = self.segmentation_node.GetDisplayNode()
        if display_node:
            display_node.SetVisibility3D(True)
            display_node.SetVisibility2D(True)
        slicer.app.processEvents()

        # Reset 3D view and zoom to show the segmentation
        threeDWidget = slicer.app.layoutManager().threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()
        threeDView.resetCamera()
        # Zoom in to see the segmentation better
        threeDView.zoomFactor = 2.0
        threeDView.zoomIn()
        slicer.app.processEvents()

        # Force render to ensure 3D surface appears
        threeDView.forceRender()
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "View in 3D",
            "The 3D view (bottom-right) shows your segmentation as a surface. "
            "The segmented region appears as a colored surface that can be rotated and examined.",
            "Segmentation shown in 3D view alongside slice views",
        )

    def _xy_to_ras(self, xy: tuple[int, int], slice_widget) -> tuple[float, float, float]:
        """Convert XY screen coordinates to RAS world coordinates."""
        slice_logic = slice_widget.sliceLogic()
        slice_node = slice_logic.GetSliceNode()
        xy_to_ras = slice_node.GetXYToRAS()

        ras_point = [0.0, 0.0, 0.0, 1.0]
        xy_point = [float(xy[0]), float(xy[1]), 0.0, 1.0]
        xy_to_ras.MultiplyPoint(xy_point, ras_point)

        return (ras_point[0], ras_point[1], ras_point[2])

    def _ras_to_xy(self, ras: tuple[float, float, float], slice_widget) -> tuple[int, int] | None:
        """Convert RAS world coordinates to XY screen coordinates."""
        import vtk

        slice_logic = slice_widget.sliceLogic()
        slice_node = slice_logic.GetSliceNode()

        xy_to_ras = slice_node.GetXYToRAS()
        ras_to_xy = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(xy_to_ras, ras_to_xy)

        ras_point = [ras[0], ras[1], ras[2], 1.0]
        xy_point = [0.0, 0.0, 0.0, 1.0]
        ras_to_xy.MultiplyPoint(ras_point, xy_point)

        return (int(xy_point[0]), int(xy_point[1]))

    def _center_and_zoom_all_views(
        self, ras: tuple[float, float, float], zoom_factor: float = 2.0
    ) -> None:
        """Center all slice views on RAS coordinates and zoom in.

        Args:
            ras: (R, A, S) coordinates to center on.
            zoom_factor: Zoom factor (2.0 = 2x zoom in).
        """
        layout_manager = slicer.app.layoutManager()

        for view_name in ["Red", "Yellow", "Green"]:
            slice_widget = layout_manager.sliceWidget(view_name)
            if slice_widget is None:
                continue

            slice_logic = slice_widget.sliceLogic()
            slice_node = slice_logic.GetSliceNode()

            # Center on RAS coordinates
            slice_node.JumpSliceByCentering(ras[0], ras[1], ras[2])

            # Zoom in by reducing field of view
            fov = slice_node.GetFieldOfView()
            slice_node.SetFieldOfView(fov[0] / zoom_factor, fov[1] / zoom_factor, fov[2])

            slice_widget.sliceView().forceRender()

        slicer.app.processEvents()

    def verify(self, ctx: TestContext) -> None:
        """Verify tutorial completed and generate documentation."""
        logger.info("Verifying Getting Started tutorial")

        # Check we captured all steps
        ctx.assert_equal(len(self.steps), 10, "Should have captured 10 tutorial steps")

        # Generate the documentation
        self._generate_markdown()

        ctx.screenshot(
            "Getting Started tutorial complete",
            doc_tags=["getting_started", "complete"],
        )

    def _generate_markdown(self) -> None:
        """Generate getting_started.md from captured steps."""
        if self.output_dir is None:
            logger.warning("No output directory set, skipping markdown generation")
            return

        lines = [
            "# Getting Started",
            "",
            "This guide walks you through using the Adaptive Brush for the first time.",
            "",
            "## Prerequisites",
            "",
            "- 3D Slicer 5.10 or later",
            "- A volume loaded (CT, MRI, etc.)",
            "",
            "## Tutorial",
            "",
        ]

        for step in self.steps:
            lines.append(f"### Step {step['number']}: {step['name']}")
            lines.append("")
            lines.append(step["description"])
            lines.append("")
            # Reference screenshot from _static/screenshots/workflows
            lines.append(f"```{{image}} /_static/screenshots/workflows/{step['screenshot']}")
            lines.append(f":alt: {step['name']}")
            lines.append(":width: 100%")
            lines.append("```")
            lines.append("")

        # Add tips section
        lines.extend(
            [
                "## Tips for Best Results",
                "",
                "### Brush Size",
                "- Start with a brush slightly smaller than your target region",
                "- Use Shift + scroll wheel to quickly adjust size",
                "",
                "### Edge Sensitivity",
                "- Higher sensitivity = stricter edge detection",
                "- Lower sensitivity = more permissive, may leak",
                "",
                "### Algorithm Selection",
                "- **Watershed**: Good general-purpose choice",
                "- **Geodesic Distance**: Fast, good for clear edges",
                "- **Threshold Brush**: Fastest, simple intensity thresholding",
                "",
                "## Next Steps",
                "",
                "- Explore different [algorithms](algorithms.md)",
                "- Use the [Parameter Wizard](parameter_wizard.md) for optimization",
                "- Create [recipes](recipes.md) for reproducible segmentation",
                "",
                "---",
                "",
                f"*This documentation was auto-generated on {datetime.now().strftime('%Y-%m-%d')}.*",
                "*Screenshots reflect the current UI.*",
                "",
            ]
        )

        # Write the markdown file
        md_file = self.output_dir / "getting_started.md"
        with open(md_file, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Generated: {md_file}")

        # Also save step metadata for other tools
        results_file = self.output_dir / "getting_started_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "steps": self.steps,
                },
                f,
                indent=2,
            )

        logger.info(f"Saved results: {results_file}")

    def teardown(self, ctx: TestContext) -> None:
        """Clean up after test."""
        logger.info("Tearing down Getting Started tutorial test")

        if self.segment_editor_widget:
            self.segment_editor_widget.setActiveEffect(None)
            self.segment_editor_widget = None

        ctx.log("Teardown complete")
