"""Markdown report generation for algorithm profiles.

Generates human-readable reports from algorithm characterization data,
including comparison tables, recommendations, and visualizations.

See ADR-011 for architecture decisions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .AlgorithmProfile import AlgorithmComparison, AlgorithmProfile

logger = logging.getLogger(__name__)


class AlgorithmReportGenerator:
    """Generate markdown reports from algorithm profiles.

    Creates detailed reports with:
    - Algorithm comparison tables
    - Individual algorithm profiles
    - Parameter recommendations
    - Use case guidance

    Example:
        comparison = characterizer.create_comparison()
        generator = AlgorithmReportGenerator(comparison)
        generator.save_report("algorithm_report.md")
    """

    def __init__(self, comparison: AlgorithmComparison) -> None:
        """Initialize report generator.

        Args:
            comparison: AlgorithmComparison with profiles to report.
        """
        self.comparison = comparison

    def generate_report(self) -> str:
        """Generate complete markdown report.

        Returns:
            Markdown string.
        """
        sections = [
            self._generate_header(),
            self._generate_summary(),
            self._generate_comparison_table(),
            self._generate_recommendations(),
            self._generate_detailed_profiles(),
            self._generate_parameter_importance(),
            self._generate_footer(),
        ]

        return "\n\n".join(sections)

    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# Algorithm Performance Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Algorithms analyzed: {len(self.comparison.profiles)}
"""

    def _generate_summary(self) -> str:
        """Generate executive summary."""
        if not self.comparison.profiles:
            return "## Summary\n\nNo algorithm data available."

        best_dice = self.comparison.get_best_for_dice()
        fastest = self.comparison.get_fastest()

        lines = [
            "## Summary",
            "",
        ]

        if best_dice:
            lines.append(
                f"**Best Accuracy:** {best_dice.display_name} "
                f"(mean Dice: {best_dice.performance.mean_dice:.3f})"
            )

        if fastest:
            lines.append(
                f"**Fastest:** {fastest.display_name} (~{fastest.performance.mean_time_ms:.0f}ms)"
            )

        if self.comparison.recommended_default:
            rec = next(
                (
                    p
                    for p in self.comparison.profiles
                    if p.algorithm == self.comparison.recommended_default
                ),
                None,
            )
            if rec:
                lines.append(f"**Recommended Default:** {rec.display_name}")

        return "\n".join(lines)

    def _generate_comparison_table(self) -> str:
        """Generate comparison table."""
        if not self.comparison.profiles:
            return ""

        lines = [
            "## Algorithm Comparison",
            "",
            "| Algorithm | Mean Dice | Best Dice | Speed | Efficiency | Tendency |",
            "|-----------|-----------|-----------|-------|------------|----------|",
        ]

        for profile in self.comparison.profiles:
            perf = profile.performance
            lines.append(
                f"| {profile.display_name} | "
                f"{perf.mean_dice:.3f} | "
                f"{perf.best_dice:.3f} | "
                f"{perf.speed_category} | "
                f"{profile.click_efficiency} | "
                f"{profile.segmentation_tendency} |"
            )

        return "\n".join(lines)

    def _generate_recommendations(self) -> str:
        """Generate use case recommendations."""
        if not self.comparison.profiles:
            return ""

        lines = [
            "## Recommendations",
            "",
            "### By Use Case",
            "",
        ]

        # Group by best use case
        use_case_algos: dict[str, list[str]] = {}
        for profile in self.comparison.profiles:
            for use_case in profile.best_for:
                if use_case not in use_case_algos:
                    use_case_algos[use_case] = []
                use_case_algos[use_case].append(profile.display_name)

        for use_case, algos in use_case_algos.items():
            lines.append(f"- **{use_case}:** {', '.join(algos)}")

        # Add avoid cases
        lines.extend(
            [
                "",
                "### What to Avoid",
                "",
            ]
        )

        for profile in self.comparison.profiles:
            if profile.avoid_for:
                lines.append(
                    f"- **{profile.display_name}:** Avoid for {', '.join(profile.avoid_for).lower()}"
                )

        return "\n".join(lines)

    def _generate_detailed_profiles(self) -> str:
        """Generate detailed profile sections."""
        sections = ["## Detailed Algorithm Profiles", ""]

        for profile in self.comparison.profiles:
            sections.append(self._generate_single_profile(profile))

        return "\n".join(sections)

    def _generate_single_profile(self, profile: AlgorithmProfile) -> str:
        """Generate section for single algorithm."""
        perf = profile.performance
        optimal = profile.optimal_preset

        lines = [
            f"### {profile.display_name}",
            "",
            "**Performance:**",
            f"- Mean Dice: {perf.mean_dice:.3f} (std: {perf.std_dice:.3f})",
            f"- Best Dice: {perf.best_dice:.3f}",
            f"- Speed: {perf.mean_time_ms:.0f}ms ({perf.speed_category})",
            f"- Trials: {perf.trial_count} (pruned: {perf.pruned_count})",
            "",
        ]

        # Optimal parameters
        if optimal.params:
            lines.extend(
                [
                    "**Optimal Parameters:**",
                    "```",
                ]
            )
            for key, value in optimal.params.items():
                lines.append(f"{key}: {value}")
            lines.extend(
                [
                    "```",
                    "",
                ]
            )

        # Strengths
        if profile.strengths:
            lines.extend(
                [
                    "**Strengths:**",
                ]
            )
            for strength in profile.strengths:
                lines.append(f"- {strength}")
            lines.append("")

        # Weaknesses
        if profile.weaknesses:
            lines.extend(
                [
                    "**Weaknesses:**",
                ]
            )
            for weakness in profile.weaknesses:
                lines.append(f"- {weakness}")
            lines.append("")

        # Best for
        if profile.best_for:
            lines.extend(
                [
                    "**Best For:**",
                ]
            )
            for use_case in profile.best_for:
                lines.append(f"- {use_case}")
            lines.append("")

        lines.append("---")
        lines.append("")

        return "\n".join(lines)

    def _generate_parameter_importance(self) -> str:
        """Generate parameter importance section."""
        # Aggregate importance across all profiles
        all_importance: dict[str, list[float]] = {}

        for profile in self.comparison.profiles:
            for param, importance in profile.parameter_importance.items():
                if param not in all_importance:
                    all_importance[param] = []
                all_importance[param].append(importance)

        if not all_importance:
            return ""

        # Average importance
        avg_importance = {
            param: sum(values) / len(values) for param, values in all_importance.items()
        }

        # Sort by importance
        sorted_params = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

        lines = [
            "## Parameter Importance",
            "",
            "Parameters ranked by impact on segmentation quality:",
            "",
            "| Parameter | Importance |",
            "|-----------|------------|",
        ]

        for param, importance in sorted_params:
            bar = "█" * int(importance * 20)
            lines.append(f"| {param} | {bar} {importance:.2f} |")

        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """Generate report footer."""
        return f"""---

*Report generated by SlicerAdaptiveBrush Algorithm Characterizer*
*{self.comparison.comparison_notes}*
"""

    def save_report(self, output_path: Path | str) -> None:
        """Save report to file.

        Args:
            output_path: Path to save markdown file.
        """
        output_path = Path(output_path)
        report = self.generate_report()

        output_path.write_text(report)
        logger.info(f"Saved algorithm report to {output_path}")


def generate_algorithm_report(
    comparison: AlgorithmComparison,
    output_path: Path | str,
) -> str:
    """Convenience function to generate and save report.

    Args:
        comparison: AlgorithmComparison to report.
        output_path: Path to save report.

    Returns:
        Generated markdown string.
    """
    generator = AlgorithmReportGenerator(comparison)
    generator.save_report(output_path)
    return generator.generate_report()
