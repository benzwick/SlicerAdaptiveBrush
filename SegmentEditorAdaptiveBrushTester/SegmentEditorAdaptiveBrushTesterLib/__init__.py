"""SegmentEditorAdaptiveBrushTester library.

This package provides the testing framework for SlicerAdaptiveBrush.
"""

from .ActionRecorder import ActionRecorder
from .GoldStandardManager import GoldStandardManager
from .LabNotebook import LabNotebook
from .MetricsCollector import MetricsCollector, TimingContext
from .ParameterOptimizer import OptimizationTrial, ParameterOptimizer, ParameterSpace
from .ReportGenerator import ReportGenerator
from .ScreenshotCapture import ScreenshotCapture, ScreenshotInfo
from .SegmentationMetrics import MetricsResult, SegmentationMetrics, StrokeMetricsTracker
from .TestCase import TestCase
from .TestContext import TestContext
from .TestRegistry import TestRegistry, register_test
from .TestRunFolder import TestRunFolder
from .TestRunner import TestRunner

__all__ = [
    # Core testing
    "TestCase",
    "TestContext",
    "TestRegistry",
    "TestRunner",
    "register_test",
    # Screenshot capture
    "ScreenshotCapture",
    "ScreenshotInfo",
    # Metrics collection
    "MetricsCollector",
    "TimingContext",
    # Test run management
    "TestRunFolder",
    "ActionRecorder",
    "ReportGenerator",
    # Segmentation metrics (Phase 2)
    "SegmentationMetrics",
    "MetricsResult",
    "StrokeMetricsTracker",
    # Gold standards (Phase 2)
    "GoldStandardManager",
    # Parameter optimization (Phase 2)
    "ParameterOptimizer",
    "ParameterSpace",
    "OptimizationTrial",
    # Lab notebooks (Phase 2)
    "LabNotebook",
]
