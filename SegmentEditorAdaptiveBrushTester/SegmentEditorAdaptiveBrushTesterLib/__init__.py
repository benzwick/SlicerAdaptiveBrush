"""SegmentEditorAdaptiveBrushTester library.

This package provides the testing framework for SlicerAdaptiveBrush.
"""

from .ActionRecorder import ActionRecorder
from .MetricsCollector import MetricsCollector, TimingContext
from .ReportGenerator import ReportGenerator
from .ScreenshotCapture import ScreenshotCapture, ScreenshotInfo
from .TestCase import TestCase
from .TestContext import TestContext
from .TestRegistry import TestRegistry, register_test
from .TestRunFolder import TestRunFolder
from .TestRunner import TestRunner

__all__ = [
    "TestCase",
    "TestContext",
    "TestRegistry",
    "TestRunner",
    "register_test",
    "ScreenshotCapture",
    "ScreenshotInfo",
    "MetricsCollector",
    "TimingContext",
    "TestRunFolder",
    "ActionRecorder",
    "ReportGenerator",
]
