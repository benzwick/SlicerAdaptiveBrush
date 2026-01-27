"""Mock optimization run factory for testing.

Creates temporary optimization run directory structures that can be
loaded by ResultsLoader for testing the Reviewer module.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MockTrialSpec:
    """Specification for a mock trial."""

    trial_number: int
    dice: float
    duration_ms: float = 100.0
    algorithm: str = "geodesic_distance"
    params: dict[str, Any] | None = None
    pruned: bool = False


class MockOptimizationRunFactory:
    """Factory for creating mock optimization run directories.

    Creates directory structures that match what ResultsLoader expects:
        <run_name>/
            config.yaml
            results.json
            parameter_importance.json
            segmentations/
            screenshots/

    Usage:
        factory = MockOptimizationRunFactory()
        run_path = factory.create_run("test_run", num_trials=5)
        # Use run_path in tests
        factory.cleanup()  # Remove temp directories
    """

    def __init__(self) -> None:
        """Initialize the factory."""
        self._temp_dirs: list[Path] = []

    def create_run(
        self,
        name: str = "test_run",
        num_trials: int = 5,
        trials: list[MockTrialSpec] | None = None,
        best_trial_idx: int | None = None,
    ) -> Path:
        """Create a mock optimization run directory.

        Args:
            name: Name for the run directory.
            num_trials: Number of trials to generate (ignored if trials provided).
            trials: Optional list of trial specifications.
            best_trial_idx: Index of best trial (defaults to highest dice).

        Returns:
            Path to the created run directory.
        """
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="mock_opt_run_"))
        self._temp_dirs.append(temp_dir)

        run_path = temp_dir / name
        run_path.mkdir(parents=True)

        # Generate trial specs if not provided
        if trials is None:
            trials = self._generate_trial_specs(num_trials)

        # Create config.yaml
        config = {
            "name": name,
            "n_trials": len(trials),
            "recipes": [
                {
                    "sample_data": "MRHead",
                    "gold_standard": "test_gold",
                }
            ],
        }
        config_path = run_path / "config.yaml"
        try:
            import yaml  # type: ignore[import-untyped]

            with open(config_path, "w") as f:
                yaml.dump(config, f)
        except ImportError:
            # Fallback to JSON-like format if yaml not available
            with open(config_path, "w") as f:
                f.write(f"name: {name}\n")
                f.write(f"n_trials: {len(trials)}\n")

        # Create results.json
        trial_data = []
        for spec in trials:
            params = spec.params or {}
            params["algorithm"] = spec.algorithm
            trial_data.append(
                {
                    "trial_number": spec.trial_number,
                    "params": params,
                    "value": spec.dice,
                    "duration_ms": spec.duration_ms,
                    "pruned": spec.pruned,
                    "user_attrs": {},
                    "intermediate_values": {},
                }
            )

        # Determine best trial
        completed_trials = [t for t in trial_data if not t["pruned"]]
        if completed_trials:
            if best_trial_idx is not None:
                best = trial_data[best_trial_idx]
            else:
                # Type ignore because mypy doesn't know t["value"] is float
                best = max(completed_trials, key=lambda t: float(t["value"]))  # type: ignore[arg-type]
        else:
            best = None

        results = {
            "config_name": name,
            "trials": trial_data,
            "best_trial": best,
        }

        with open(run_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Create parameter_importance.json
        importance = {
            "algorithm": 0.45,
            "radius": 0.25,
            "threshold_zone": 0.15,
            "std_multiplier": 0.10,
            "closing_radius": 0.05,
        }
        with open(run_path / "parameter_importance.json", "w") as f:
            json.dump(importance, f, indent=2)

        # Create directories for screenshots and segmentations
        (run_path / "screenshots").mkdir()
        (run_path / "segmentations").mkdir()

        logger.info(f"Created mock optimization run at {run_path}")
        return run_path

    def add_segmentation_files(
        self,
        run_path: Path,
        volume_node: Any,
        trial_numbers: list[int] | None = None,
    ) -> None:
        """Add segmentation files for trials.

        Creates sphere segmentations and saves them as .seg.nrrd files
        for the specified trials (or all trials if not specified).

        Args:
            run_path: Path to the optimization run directory.
            volume_node: The volume node for reference geometry.
            trial_numbers: Optional list of trial numbers to create segmentations for.
        """
        import slicer

        # Read results to get trial list
        results_path = run_path / "results.json"
        if not results_path.exists():
            logger.warning(f"No results.json found in {run_path}")
            return

        with open(results_path) as f:
            results = json.load(f)

        trials = results.get("trials", [])
        if not trials:
            logger.warning("No trials in results.json")
            return

        seg_dir = run_path / "segmentations"

        # Import segmentation factory
        from .mock_segmentations import MockSegmentationFactory

        seg_factory = MockSegmentationFactory()

        for trial in trials:
            trial_num = trial.get("trial_number", 0)
            if trial_numbers is not None and trial_num not in trial_numbers:
                continue

            # Create segmentation with a sphere
            seg_node = seg_factory.create_sphere_segmentation(
                center_ijk=(128, 128, 64),  # Roughly center of MRHead
                radius=15,
                volume_node=volume_node,
                name=f"Trial_{trial_num}",
            )

            # Save to file
            seg_file = seg_dir / f"trial_{trial_num}.seg.nrrd"
            slicer.util.saveNode(seg_node, str(seg_file))

            # Update trial data with segmentation path
            trial["segmentation_path"] = str(seg_file)

            # Clean up the node
            slicer.mrmlScene.RemoveNode(seg_node)

        # Re-save results with updated paths
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        seg_factory.cleanup()
        logger.info(f"Added segmentation files to {run_path}")

    def _generate_trial_specs(self, num_trials: int) -> list[MockTrialSpec]:
        """Generate random trial specifications.

        Args:
            num_trials: Number of trials to generate.

        Returns:
            List of MockTrialSpec objects.
        """
        import random

        random.seed(42)  # Reproducible

        algorithms = [
            "geodesic_distance",
            "watershed",
            "connected_threshold",
            "region_growing",
            "level_set",
            "random_walker",
            "threshold_brush",
        ]

        specs = []
        for i in range(num_trials):
            specs.append(
                MockTrialSpec(
                    trial_number=i + 1,
                    dice=0.5 + random.random() * 0.45,  # 0.5 - 0.95
                    duration_ms=50 + random.random() * 200,  # 50 - 250ms
                    algorithm=random.choice(algorithms),
                    pruned=random.random() < 0.1,  # 10% pruned
                )
            )

        return specs

    def create_minimal_run(self, name: str = "minimal_run") -> Path:
        """Create a minimal run with just one trial.

        Args:
            name: Run name.

        Returns:
            Path to run directory.
        """
        return self.create_run(
            name=name,
            trials=[
                MockTrialSpec(trial_number=1, dice=0.85),
            ],
        )

    def create_test_format_run(self, name: str = "test_run") -> Path:
        """Create a run in test runner format (array of test results).

        This tests the alternate parsing path in ResultsLoader.

        Args:
            name: Run name.

        Returns:
            Path to run directory.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="mock_test_run_"))
        self._temp_dirs.append(temp_dir)

        run_path = temp_dir / name
        run_path.mkdir(parents=True)

        # Test runner format is a list of test results
        results = [
            {
                "name": "test_algorithm_watershed",
                "passed": True,
                "duration_seconds": 1.5,
                "screenshots": [],
                "assertions": [{"passed": True, "message": "Should segment tissue"}],
                "metrics": {
                    "metrics": [
                        {"name": "voxel_count", "value": 12500, "unit": "voxels"},
                        {"name": "dice_score", "value": 0.92, "unit": ""},
                    ]
                },
            },
            {
                "name": "test_algorithm_geodesic",
                "passed": True,
                "duration_seconds": 0.8,
                "screenshots": ["001.png", "002.png"],
                "assertions": [{"passed": True, "message": "Should segment"}],
                "metrics": {
                    "metrics": [
                        {"name": "voxel_count", "value": 11000, "unit": "voxels"},
                        {"name": "dice_coefficient", "value": 0.89, "unit": ""},
                    ]
                },
            },
        ]

        with open(run_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        (run_path / "screenshots").mkdir()
        (run_path / "segmentations").mkdir()

        logger.info(f"Created mock test format run at {run_path}")
        return run_path

    def create_real_trials(
        self,
        run_path: Path,
        volume_node: Any,
        paint_locations: list[tuple[float, float, float]] | None = None,
        algorithms: list[str] | None = None,
    ) -> None:
        """Create real trials by actually painting with Adaptive Brush.

        This runs the actual segmentation algorithm at specified RAS locations
        and saves the resulting segmentations as trial data.

        Args:
            run_path: Path to the optimization run directory.
            volume_node: The volume node to segment.
            paint_locations: List of RAS coordinates to paint at.
                            Defaults to several locations in MRHead brain.
            algorithms: List of algorithms to test. Defaults to common ones.
        """
        import time

        import slicer

        # Default paint locations (good for MRHead)
        if paint_locations is None:
            paint_locations = [
                (-2.8, 6.4, 0.0),  # Center brain
                (25.0, -5.0, 10.0),  # Right hemisphere
                (-25.0, -5.0, 10.0),  # Left hemisphere
            ]

        # Default algorithms
        if algorithms is None:
            algorithms = ["watershed", "geodesic_distance", "connected_threshold"]

        # Set up Segment Editor
        slicer.util.selectModule("SegmentEditor")
        slicer.app.processEvents()

        # Create segmentation node
        seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        seg_node.CreateDefaultDisplayNodes()
        seg_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)

        # Get segment editor widget
        seg_editor_module = slicer.modules.segmenteditor.widgetRepresentation().self()
        seg_editor_widget = seg_editor_module.editor
        seg_editor_widget.setSegmentationNode(seg_node)
        seg_editor_widget.setSourceVolumeNode(volume_node)

        # Activate Adaptive Brush
        seg_editor_widget.setActiveEffectByName("Adaptive Brush")
        effect = seg_editor_widget.activeEffect()

        if effect is None:
            logger.error("Failed to activate Adaptive Brush")
            return

        scripted_effect = effect.self()
        scripted_effect.radiusSlider.value = 20.0  # 20mm brush
        slicer.app.processEvents()

        # Results to save
        seg_dir = run_path / "segmentations"
        trial_results = []

        for trial_num, (algo, location) in enumerate(
            zip(algorithms * len(paint_locations), paint_locations * len(algorithms)), start=1
        ):
            if trial_num > len(algorithms) * len(paint_locations):
                break

            logger.info(f"Creating trial {trial_num}: {algo} at {location}")

            # Clear segment for this trial
            seg_node.GetSegmentation().RemoveAllSegments()
            segment_id = seg_node.GetSegmentation().AddEmptySegment(f"Trial_{trial_num}")
            seg_editor_widget.setCurrentSegmentID(segment_id)

            # Re-activate effect
            seg_editor_widget.setActiveEffectByName("Adaptive Brush")
            effect = seg_editor_widget.activeEffect()
            scripted_effect = effect.self()

            # Set algorithm
            combo = scripted_effect.algorithmCombo
            idx = combo.findData(algo)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            slicer.app.processEvents()

            # Paint at location
            start_time = time.time()
            scripted_effect.paintAt(*location)
            duration_ms = (time.time() - start_time) * 1000
            slicer.app.processEvents()

            # Save segmentation
            seg_file = seg_dir / f"trial_{trial_num}.seg.nrrd"
            slicer.util.saveNode(seg_node, str(seg_file))

            # Count voxels for "dice" approximation (just using voxel count as metric)
            import numpy as np

            array = slicer.util.arrayFromSegmentBinaryLabelmap(seg_node, segment_id, volume_node)
            voxel_count = int(np.sum(array > 0)) if array is not None else 0
            # Fake dice as normalized voxel count (for mock purposes)
            fake_dice = min(0.95, voxel_count / 50000.0 + 0.5) if voxel_count > 0 else 0.1

            trial_results.append(
                {
                    "trial_number": trial_num,
                    "params": {"algorithm": algo, "location": list(location)},
                    "value": fake_dice,
                    "duration_ms": duration_ms,
                    "pruned": False,
                    "user_attrs": {"voxel_count": voxel_count},
                    "intermediate_values": {},
                    "segmentation_path": str(seg_file),
                }
            )

            logger.info(f"  Trial {trial_num}: {voxel_count} voxels in {duration_ms:.1f}ms")

        # Update results.json with real trial data
        results_path = run_path / "results.json"
        best_trial = max(trial_results, key=lambda t: t["value"]) if trial_results else None
        results = {
            "config_name": run_path.name,
            "trials": trial_results,
            "best_trial": best_trial,
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Clean up
        seg_editor_widget.setActiveEffect(None)
        slicer.mrmlScene.RemoveNode(seg_node)

        logger.info(f"Created {len(trial_results)} real trials in {run_path}")

    def cleanup(self) -> None:
        """Remove all temporary directories created by this factory."""
        for temp_dir in self._temp_dirs:
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_dir}: {e}")
        self._temp_dirs.clear()
