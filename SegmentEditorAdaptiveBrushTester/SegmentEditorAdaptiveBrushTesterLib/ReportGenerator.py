"""Report generator for test runs.

Generates human-readable and machine-parseable reports from test results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .TestCase import TestResult
    from .TestRunner import TestSuiteResult

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports from test results.

    Usage:
        generator = ReportGenerator()

        # Generate markdown report
        report = generator.generate_markdown(suite_result)
        print(report)

        # Save to file
        generator.save_markdown(suite_result, Path("./report.md"))

        # Generate summary for Claude review
        summary = generator.generate_summary(suite_result)
    """

    def generate_markdown(self, suite_result: TestSuiteResult) -> str:
        """Generate a markdown report from test suite results.

        Args:
            suite_result: TestSuiteResult to report on.

        Returns:
            Markdown-formatted report string.
        """
        lines = []

        # Header
        lines.append(f"# Test Report: {suite_result.suite_name}")
        lines.append("")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Duration:** {suite_result.duration_seconds:.2f}s")
        lines.append(f"**Output:** `{suite_result.output_folder}`")
        lines.append("")

        # Summary
        status = "PASSED" if suite_result.passed else "FAILED"
        lines.append("## Summary")
        lines.append("")
        lines.append(f"**Status:** {status}")
        lines.append(
            f"**Tests:** {suite_result.passed_count}/{suite_result.total_count} passed"
        )
        lines.append("")

        # Results table
        lines.append("## Results")
        lines.append("")
        lines.append("| Test | Status | Duration | Assertions |")
        lines.append("|------|--------|----------|------------|")

        for r in suite_result.results:
            status = "PASS" if r.passed else "FAIL"
            failed = len([a for a in r.assertions if not a.passed])
            total = len(r.assertions)
            assertions = f"{total - failed}/{total}"
            lines.append(f"| {r.name} | {status} | {r.duration_seconds:.2f}s | {assertions} |")

        lines.append("")

        # Failed tests detail
        failed_tests = [r for r in suite_result.results if not r.passed]
        if failed_tests:
            lines.append("## Failed Tests")
            lines.append("")

            for r in failed_tests:
                lines.append(f"### {r.name}")
                lines.append("")

                if r.error:
                    lines.append("**Error:**")
                    lines.append("```")
                    lines.append(r.error)
                    lines.append("```")
                    lines.append("")

                failed_assertions = [a for a in r.assertions if not a.passed]
                if failed_assertions:
                    lines.append("**Failed Assertions:**")
                    for a in failed_assertions:
                        lines.append(f"- {a.message}")
                        if a.expected is not None:
                            lines.append(f"  - Expected: {a.expected}")
                            lines.append(f"  - Actual: {a.actual}")
                    lines.append("")

                if r.screenshots:
                    lines.append("**Screenshots:**")
                    for s in r.screenshots:
                        lines.append(f"- `{s}`")
                    lines.append("")

        # Performance metrics
        lines.append("## Performance Metrics")
        lines.append("")

        for r in suite_result.results:
            if r.metrics and r.metrics.get("timings"):
                lines.append(f"### {r.name}")
                lines.append("")
                lines.append("| Operation | Duration |")
                lines.append("|-----------|----------|")
                for t in r.metrics["timings"]:
                    lines.append(f"| {t['operation']} | {t['duration_ms']:.1f}ms |")
                lines.append("")

        return "\n".join(lines)

    def save_markdown(self, suite_result: TestSuiteResult, filepath: Path) -> None:
        """Save markdown report to file.

        Args:
            suite_result: TestSuiteResult to report on.
            filepath: Path to save report.
        """
        report = self.generate_markdown(suite_result)

        with open(filepath, "w") as f:
            f.write(report)

        logger.info(f"Saved markdown report to: {filepath}")

    def generate_summary(self, suite_result: TestSuiteResult) -> dict:
        """Generate a summary for Claude review.

        Returns a structured dictionary suitable for AI analysis.

        Args:
            suite_result: TestSuiteResult to summarize.

        Returns:
            Summary dictionary with key information.
        """
        failed_tests = [r for r in suite_result.results if not r.passed]

        summary = {
            "suite_name": suite_result.suite_name,
            "passed": suite_result.passed,
            "total_tests": suite_result.total_count,
            "passed_count": suite_result.passed_count,
            "failed_count": suite_result.failed_count,
            "duration_seconds": suite_result.duration_seconds,
            "output_folder": str(suite_result.output_folder),
            "failed_tests": [],
            "performance_issues": [],
            "screenshots": [],
        }

        # Detail failed tests
        for r in failed_tests:
            test_summary = {
                "name": r.name,
                "error": r.error,
                "failed_assertions": [
                    {"message": a.message, "expected": str(a.expected), "actual": str(a.actual)}
                    for a in r.assertions
                    if not a.passed
                ],
                "screenshots": r.screenshots,
            }
            summary["failed_tests"].append(test_summary)

        # Check for performance issues (operations > 100ms)
        for r in suite_result.results:
            if r.metrics and r.metrics.get("timings"):
                for t in r.metrics["timings"]:
                    if t["duration_ms"] > 100:
                        summary["performance_issues"].append(
                            {
                                "test": r.name,
                                "operation": t["operation"],
                                "duration_ms": t["duration_ms"],
                            }
                        )

        # Collect all screenshots
        for r in suite_result.results:
            for s in r.screenshots:
                summary["screenshots"].append({"test": r.name, "filename": s})

        return summary

    def save_summary(self, suite_result: TestSuiteResult, filepath: Path) -> None:
        """Save summary to JSON file.

        Args:
            suite_result: TestSuiteResult to summarize.
            filepath: Path to save summary.
        """
        summary = self.generate_summary(suite_result)

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved summary to: {filepath}")
