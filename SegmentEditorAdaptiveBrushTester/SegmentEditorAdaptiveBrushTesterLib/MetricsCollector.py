"""Metrics collection for test runs.

Collects timing and quality metrics during test execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TimingResult:
    """Result of a timing measurement."""

    operation: str
    duration_seconds: float
    start_time: float
    end_time: float

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_seconds * 1000


class TimingContext:
    """Context manager for timing an operation.

    Usage:
        with ctx.timing("watershed_stroke") as t:
            effect.apply(...)
        print(f"Took {t.duration_ms:.1f}ms")
    """

    def __init__(self, operation: str, collector: MetricsCollector) -> None:
        """Initialize timing context.

        Args:
            operation: Name of operation being timed.
            collector: MetricsCollector to record result.
        """
        self.operation = operation
        self._collector = collector
        self._start_time: float = 0
        self._end_time: float = 0
        self._result: TimingResult | None = None

    def __enter__(self) -> TimingContext:
        """Start timing."""
        self._start_time = time.time()
        logger.debug(f"Timing started: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and record result."""
        self._end_time = time.time()
        duration = self._end_time - self._start_time

        self._result = TimingResult(
            operation=self.operation,
            duration_seconds=duration,
            start_time=self._start_time,
            end_time=self._end_time,
        )

        self._collector._record_timing(self._result)
        logger.debug(f"Timing complete: {self.operation} = {self._result.duration_ms:.1f}ms")

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds (available after context exits)."""
        if self._result is None:
            return 0.0
        return self._result.duration_seconds

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds (available after context exits)."""
        if self._result is None:
            return 0.0
        return self._result.duration_ms


@dataclass
class MetricValue:
    """A recorded metric value."""

    name: str
    value: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects timing and quality metrics during tests.

    Usage:
        collector = MetricsCollector()

        # Time an operation
        with collector.timing("watershed_stroke"):
            effect.apply(...)

        # Record a custom metric
        collector.record_metric("voxel_count", 1247, "voxels")

        # Get all metrics
        metrics = collector.get_metrics()
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._timings: list[TimingResult] = []
        self._metrics: list[MetricValue] = []

    def timing(self, operation: str) -> TimingContext:
        """Create a timing context for measuring operation duration.

        Args:
            operation: Name of the operation being timed.

        Returns:
            Context manager that records duration on exit.
        """
        return TimingContext(operation, self)

    def _record_timing(self, result: TimingResult) -> None:
        """Record a timing result (called by TimingContext)."""
        self._timings.append(result)

    def record_metric(self, name: str, value: float, unit: str = "") -> None:
        """Record a custom metric.

        Args:
            name: Metric name (e.g., "voxel_count", "dice_coefficient").
            value: Metric value.
            unit: Optional unit string (e.g., "voxels", "ms").
        """
        metric = MetricValue(name=name, value=value, unit=unit)
        self._metrics.append(metric)
        logger.debug(f"Recorded metric: {name} = {value} {unit}")

    def get_metrics(self) -> dict:
        """Get all collected metrics as a dictionary.

        Returns:
            Dictionary with "timings" and "metrics" keys.
        """
        return {
            "timings": [
                {
                    "operation": t.operation,
                    "duration_seconds": t.duration_seconds,
                    "duration_ms": t.duration_ms,
                }
                for t in self._timings
            ],
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                }
                for m in self._metrics
            ],
        }

    def get_timing(self, operation: str) -> TimingResult | None:
        """Get timing for a specific operation.

        Args:
            operation: Operation name.

        Returns:
            TimingResult or None if not found.
        """
        for t in reversed(self._timings):  # Return most recent
            if t.operation == operation:
                return t
        return None

    def get_metric(self, name: str) -> MetricValue | None:
        """Get a specific metric.

        Args:
            name: Metric name.

        Returns:
            MetricValue or None if not found.
        """
        for m in reversed(self._metrics):  # Return most recent
            if m.name == name:
                return m
        return None

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._timings.clear()
        self._metrics.clear()
