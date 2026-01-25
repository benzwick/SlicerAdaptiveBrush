"""SegmentEditorAdaptiveBrushTester library.

This package provides the testing framework for SlicerAdaptiveBrush.
"""

from .ActionRecorder import ActionRecorder
from .AlgorithmCharacterizer import AlgorithmCharacterizer
from .AlgorithmProfile import (
    ALGORITHM_DISPLAY_NAMES,
    AlgorithmComparison,
    AlgorithmProfile,
    ExampleScreenshot,
    OptimalPreset,
    PerformanceMetrics,
    get_display_name,
)
from .AlgorithmReportGenerator import AlgorithmReportGenerator, generate_algorithm_report
from .GoldStandardManager import GoldStandardManager
from .LabNotebook import LabNotebook
from .MetricsCollector import MetricsCollector, TimingContext
from .OptimizationConfig import OptimizationConfig, ParameterSpec, RecipeSpec, create_default_config
from .OptunaOptimizer import (
    OPTUNA_AVAILABLE,
    OptimizationResults,
    OptunaOptimizer,
    OptunaTrialResult,
)
from .ParameterOptimizer import OptimizationTrial, ParameterOptimizer, ParameterSpace
from .Recipe import Recipe, list_recipes
from .RecipeRecorder import (
    RecipeRecorder,
    get_global_recorder,
    is_recording,
    start_recording,
    stop_recording,
)
from .RecipeRunner import RecipeResult, RecipeRunner, run_recipe
from .RecipeTestRunner import (
    RecipeTestResult,
    RecipeTestRunner,
    RecipeTestSuiteResult,
    run_recipe_tests,
)
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
    # Recipes (v0.13.0)
    "Recipe",
    "RecipeRunner",
    "RecipeResult",
    "run_recipe",
    "list_recipes",
    "RecipeRecorder",
    "get_global_recorder",
    "start_recording",
    "stop_recording",
    "is_recording",
    # Recipe testing (v0.13.0)
    "RecipeTestRunner",
    "RecipeTestResult",
    "RecipeTestSuiteResult",
    "run_recipe_tests",
    # Optuna optimization (v0.13.0)
    "OptimizationConfig",
    "ParameterSpec",
    "RecipeSpec",
    "create_default_config",
    "OptunaOptimizer",
    "OptunaTrialResult",
    "OptimizationResults",
    "OPTUNA_AVAILABLE",
    # Algorithm characterization (v0.13.0)
    "AlgorithmProfile",
    "AlgorithmComparison",
    "AlgorithmCharacterizer",
    "AlgorithmReportGenerator",
    "PerformanceMetrics",
    "OptimalPreset",
    "ExampleScreenshot",
    "ALGORITHM_DISPLAY_NAMES",
    "get_display_name",
    "generate_algorithm_report",
]
