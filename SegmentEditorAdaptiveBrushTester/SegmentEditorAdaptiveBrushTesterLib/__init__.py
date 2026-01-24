"""SegmentEditorAdaptiveBrushTester library.

This package provides the testing framework for SlicerAdaptiveBrush.
"""

from .TestCase import TestCase
from .TestContext import TestContext
from .TestRegistry import TestRegistry, register_test
from .TestRunner import TestRunner
from .ScreenshotCapture import ScreenshotCapture, ScreenshotInfo
from .MetricsCollector import MetricsCollector, TimingContext
from .TestRunFolder import TestRunFolder
from .ActionRecorder import ActionRecorder
from .ReportGenerator import ReportGenerator

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
