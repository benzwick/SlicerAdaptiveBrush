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
        # Step 1: Load Sample Data
        # =========================================
        import SampleData

        self.volume_node = SampleData.downloadSample("MRBrainTumor1")
        slicer.app.processEvents()

        # Set up view
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)
        slicer.util.resetSliceViews()
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Load Volume",
            "Load your volume data. This tutorial uses the MRBrainTumor1 sample dataset "
            "which contains a brain MRI with a visible tumor.",
            "MRBrainTumor1 sample data loaded in conventional layout",
        )

        # =========================================
        # Step 2: Open Segment Editor
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
        # Step 3: Create Segmentation
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
        # Step 4: Select Adaptive Brush
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
        # Step 5: Configure Brush Settings
        # =========================================
        scripted_effect = self.effect.self()

        # Set reasonable defaults for demo
        scripted_effect.radiusSlider.value = 15.0
        slicer.app.processEvents()

        # Select Watershed algorithm
        combo = scripted_effect.algorithmCombo
        idx = combo.findData("watershed")
        if idx >= 0:
            combo.setCurrentIndex(idx)
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Configure Settings",
            "Adjust the brush radius using the slider or Shift + scroll wheel. "
            "Select an algorithm - Watershed is a good general-purpose choice.",
            "Brush radius and algorithm configured",
        )

        # =========================================
        # Step 6: Navigate to Region
        # =========================================
        # Navigate to a slice showing the tumor
        red_widget = slicer.app.layoutManager().sliceWidget("Red")
        red_logic = red_widget.sliceLogic()
        red_logic.SetSliceOffset(0)  # Center of volume
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Navigate to Region",
            "Use the slice sliders or scroll wheel to navigate to the region "
            "you want to segment. Position the cursor over the target structure.",
            "Navigated to slice showing tumor region",
        )

        # =========================================
        # Step 7: Paint Segmentation
        # =========================================
        # Show brush preview at center
        view_widget = red_widget

        # Get center of slice view
        size = view_widget.sliceView().renderWindow().GetSize()
        center_xy = (size[0] // 2, size[1] // 2)

        # Update brush preview
        scripted_effect._updateBrushPreview(center_xy, view_widget, eraseMode=False)
        view_widget.sliceView().forceRender()
        slicer.app.processEvents()

        self._record_step(
            ctx,
            "Paint",
            "Click and drag on the slice view to paint. The adaptive brush will "
            "automatically detect edges and segment the region based on intensity similarity.",
            "Brush positioned over tumor ready to paint",
        )

        # =========================================
        # Step 8: Controls Reference
        # =========================================
        self._record_step(
            ctx,
            "Controls",
            "**Keyboard shortcuts:**\n"
            "- **Left-click drag**: Paint (add to segment)\n"
            "- **Ctrl + Left-click drag**: Erase (remove from segment)\n"
            "- **Shift + Scroll wheel**: Adjust brush radius\n"
            "- **Ctrl + Shift + Scroll wheel**: Adjust threshold zone",
            "Adaptive Brush with controls reference",
        )

    def verify(self, ctx: TestContext) -> None:
        """Verify tutorial completed and generate documentation."""
        logger.info("Verifying Getting Started tutorial")

        # Check we captured all steps
        ctx.assert_equal(len(self.steps), 8, "Should have captured 8 tutorial steps")

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
