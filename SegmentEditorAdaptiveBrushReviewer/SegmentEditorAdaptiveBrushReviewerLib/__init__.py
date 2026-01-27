"""SegmentEditorAdaptiveBrushReviewer library.

This package provides components for the Results Review module.

See ADR-012 for architecture decisions.
"""

from .ComparisonMetrics import ComparisonMetrics, SegmentationMetrics, compute_metrics_from_nodes
from .ContourRenderer import ContourRenderer
from .RatingManager import Rating, RatingManager, ReviewRecord
from .ResultsLoader import OptimizationRun, ResultsLoader
from .ScreenshotViewer import ScreenshotViewer
from .VisualizationController import VisualizationController

__all__ = [
    "ResultsLoader",
    "OptimizationRun",
    "VisualizationController",
    "ScreenshotViewer",
    "ContourRenderer",
    "ComparisonMetrics",
    "SegmentationMetrics",
    "compute_metrics_from_nodes",
    "RatingManager",
    "Rating",
    "ReviewRecord",
]
