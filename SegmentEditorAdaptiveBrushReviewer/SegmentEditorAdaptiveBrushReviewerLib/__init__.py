"""SegmentEditorAdaptiveBrushReviewer library.

This package provides components for the Results Review module.

See ADR-012 for architecture decisions.
See ADR-016 for enhanced visualization and navigation.
See ADR-017 for DICOM SEG data format.
See ADR-018 for CrossSegmentationExplorer integration.
"""

from .ComparisonMetrics import ComparisonMetrics, SegmentationMetrics, compute_metrics_from_nodes
from .ContourRenderer import ContourRenderer
from .DicomManager import (
    DicomDatabaseNotAvailable,
    DicomManager,
    DicomManagerError,
    QuantitativeReportingNotAvailable,
)
from .ModelGrouping import (
    ComparisonModel,
    TrialModelMapper,
    quick_compare_algorithms,
    quick_compare_top_trials,
)
from .RatingManager import Rating, RatingManager, ReviewRecord
from .ResultsLoader import OptimizationRun, ResultsLoader
from .ScreenshotViewer import ScreenshotViewer
from .SequenceRecorder import SceneViewBookmarks, SequenceRecorder, ViewGroupManager
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
    # Sequence recording and view management (ADR-016 enhancement)
    "SequenceRecorder",
    "ViewGroupManager",
    "SceneViewBookmarks",
    # DICOM management (ADR-017)
    "DicomManager",
    "DicomManagerError",
    "DicomDatabaseNotAvailable",
    "QuantitativeReportingNotAvailable",
    # Model grouping for cross-comparison (ADR-018)
    "ComparisonModel",
    "TrialModelMapper",
    "quick_compare_algorithms",
    "quick_compare_top_trials",
]
