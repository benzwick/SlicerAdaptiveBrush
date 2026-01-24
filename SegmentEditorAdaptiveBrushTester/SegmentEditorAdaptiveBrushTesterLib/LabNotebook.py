"""Lab notebook for documenting optimization findings.

Creates markdown documents to record observations, metrics, and conclusions
from optimization and testing workflows.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class LabNotebook:
    """Document optimization findings in markdown format.

    Creates structured markdown documents for recording:
    - Optimization experiment setup
    - Trial results and metrics
    - Observations and conclusions
    - Screenshot references

    Usage:
        notebook = LabNotebook("Watershed Parameter Optimization")

        notebook.add_section("Setup", "Testing on MRBrainTumor1 sample...")
        notebook.add_observation("Edge sensitivity > 60 causes over-segmentation")

        notebook.add_metrics_table(
            metrics=[{"trial": 1, "dice": 0.85}, {"trial": 2, "dice": 0.92}],
            columns=["trial", "dice"]
        )

        notebook.add_conclusion("Optimal edge sensitivity is 40-50")
        notebook.save()
    """

    # Lab notebooks directory relative to this file's package
    NOTEBOOKS_DIR = Path(__file__).parent.parent / "LabNotebooks"

    def __init__(self, title: str, filename: str | None = None) -> None:
        """Initialize lab notebook.

        Args:
            title: Title of the notebook document.
            filename: Optional custom filename (without extension).
                      Default: {date}_{title_slug}.md
        """
        self.title = title
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.time = datetime.now().strftime("%H:%M:%S")

        if filename:
            self.filename = f"{filename}.md"
        else:
            title_slug = title.lower().replace(" ", "_")[:50]
            self.filename = f"{self.date}_{title_slug}.md"

        self.entries: list[str] = []

        self.NOTEBOOKS_DIR.mkdir(exist_ok=True)
        self._write_header()

    def _write_header(self) -> None:
        """Write notebook header."""
        self.entries.append(f"# {self.title}")
        self.entries.append("")
        self.entries.append(f"**Date:** {self.date}")
        self.entries.append(f"**Time:** {self.time}")
        self.entries.append("")

    def add_section(self, heading: str, content: str) -> None:
        """Add a section with heading and content.

        Args:
            heading: Section heading.
            content: Section content (can include markdown).
        """
        self.entries.append(f"## {heading}")
        self.entries.append("")
        self.entries.append(content)
        self.entries.append("")

    def add_subsection(self, heading: str, content: str) -> None:
        """Add a subsection with heading and content.

        Args:
            heading: Subsection heading.
            content: Subsection content.
        """
        self.entries.append(f"### {heading}")
        self.entries.append("")
        self.entries.append(content)
        self.entries.append("")

    def add_observation(self, observation: str) -> None:
        """Add a timestamped observation.

        Args:
            observation: Observation text.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.entries.append(f"- **{timestamp}:** {observation}")

    def add_bullet(self, text: str) -> None:
        """Add a bullet point.

        Args:
            text: Bullet point text.
        """
        self.entries.append(f"- {text}")

    def add_numbered(self, number: int, text: str) -> None:
        """Add a numbered item.

        Args:
            number: Item number.
            text: Item text.
        """
        self.entries.append(f"{number}. {text}")

    def add_code_block(self, code: str, language: str = "python") -> None:
        """Add a code block.

        Args:
            code: Code content.
            language: Programming language for syntax highlighting.
        """
        self.entries.append(f"```{language}")
        self.entries.append(code)
        self.entries.append("```")
        self.entries.append("")

    def add_metrics_table(self, metrics: list[dict], columns: list[str]) -> None:
        """Add a markdown table of metrics.

        Args:
            metrics: List of metric dictionaries.
            columns: Column names to include.
        """
        if not metrics:
            return

        # Header
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"

        # Rows
        rows = []
        for m in metrics:
            values = []
            for c in columns:
                v = m.get(c, "")
                if isinstance(v, float):
                    values.append(f"{v:.3f}")
                else:
                    values.append(str(v))
            row = "| " + " | ".join(values) + " |"
            rows.append(row)

        self.entries.extend([header, separator] + rows)
        self.entries.append("")

    def add_optimization_summary(self, summary: dict) -> None:
        """Add an optimization summary section.

        Args:
            summary: Summary dictionary from ParameterOptimizer.get_summary().
        """
        self.add_section("Optimization Summary", "")

        self.entries.append(f"- **Algorithm:** {summary.get('algorithm', 'N/A')}")
        self.entries.append(f"- **Gold Standard:** {summary.get('gold_standard', 'N/A')}")
        self.entries.append(f"- **Total Trials:** {summary.get('total_trials', 0)}")
        self.entries.append("")

        self.add_subsection("Best Results", "")
        self.entries.append(f"- **Best Dice:** {summary.get('best_dice', 0):.3f}")
        self.entries.append(
            f"- **Best Hausdorff 95%:** {summary.get('best_hausdorff_95', float('inf')):.1f}mm"
        )
        self.entries.append("")

        if "best_dice_params" in summary:
            self.add_subsection("Best Parameters (by Dice)", "")
            self.add_code_block(str(summary["best_dice_params"]))

    def add_screenshot_reference(self, path: str, caption: str) -> None:
        """Add a screenshot reference with caption.

        Args:
            path: Relative or absolute path to screenshot.
            caption: Caption for the screenshot.
        """
        self.entries.append(f"![{caption}]({path})")
        self.entries.append(f"*{caption}*")
        self.entries.append("")

    def add_parameter_analysis(self, analysis: dict[str, dict]) -> None:
        """Add parameter sensitivity analysis.

        Args:
            analysis: Analysis from ParameterOptimizer.analyze_parameter_sensitivity().
        """
        self.add_section("Parameter Sensitivity Analysis", "")

        for param, data in analysis.items():
            self.add_subsection(f"Parameter: {param}", "")

            if "correlation" in data:
                self.entries.append(f"- **Correlation with Dice:** {data['correlation']:.3f}")
                self.entries.append(f"- **Value Range:** {data['value_range']}")
                self.entries.append(f"- **Best Value:** {data['best_value']}")
            elif "avg_dice_by_value" in data:
                self.entries.append("- **Type:** Categorical")
                self.entries.append(f"- **Best Value:** {data['best_value']}")
                self.entries.append("- **Average Dice by Value:**")
                for v, d in data["avg_dice_by_value"].items():
                    self.entries.append(f"  - {v}: {d:.3f}")

            self.entries.append("")

    def add_conclusion(self, text: str) -> None:
        """Add a conclusion section.

        Args:
            text: Conclusion text.
        """
        self.add_section("Conclusions", text)

    def add_next_steps(self, steps: list[str]) -> None:
        """Add a next steps section.

        Args:
            steps: List of next steps.
        """
        self.add_section("Next Steps", "")
        for i, step in enumerate(steps, 1):
            self.add_numbered(i, step)
        self.entries.append("")

    def add_raw(self, text: str) -> None:
        """Add raw text without formatting.

        Args:
            text: Raw text to add.
        """
        self.entries.append(text)

    def save(self) -> Path:
        """Save the notebook to disk.

        Returns:
            Path to the saved file.
        """
        filepath = self.NOTEBOOKS_DIR / self.filename

        with open(filepath, "w") as f:
            f.write("\n".join(self.entries))

        logger.info(f"Saved lab notebook to {filepath}")
        return filepath

    def get_content(self) -> str:
        """Get the notebook content as a string.

        Returns:
            Notebook content.
        """
        return "\n".join(self.entries)

    @classmethod
    def list_notebooks(cls) -> list[dict]:
        """List all lab notebooks.

        Returns:
            List of notebook info dictionaries.
        """
        notebooks: list[dict] = []

        if not cls.NOTEBOOKS_DIR.exists():
            return notebooks

        for path in sorted(cls.NOTEBOOKS_DIR.glob("*.md"), reverse=True):
            # Extract title from first line
            try:
                with open(path) as f:
                    first_line = f.readline().strip()
                    title = first_line.lstrip("# ") if first_line.startswith("#") else path.stem

                notebooks.append(
                    {
                        "filename": path.name,
                        "title": title,
                        "path": str(path),
                        "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not read notebook {path}: {e}")

        return notebooks
