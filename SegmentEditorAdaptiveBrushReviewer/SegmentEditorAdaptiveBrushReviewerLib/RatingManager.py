"""Rating management for segmentation review.

Provides a rating system with CSV export for tracking review decisions
on trial segmentations.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Rating(IntEnum):
    """Segmentation quality rating (Likert scale).

    Based on SegmentationReview extension's approach:
    https://github.com/zapaishchykova/SegmentationReview
    """

    UNRATED = 0
    ACCEPT = 1  # Acceptable - ready for use
    MINOR = 2  # Minor changes needed - small corrections required
    MAJOR = 3  # Major changes needed - significant rework required
    REJECT = 4  # Unacceptable - unsuitable for use

    @property
    def label(self) -> str:
        """Human-readable label."""
        labels = {
            0: "Unrated",
            1: "Accept",
            2: "Minor changes",
            3: "Major changes",
            4: "Reject",
        }
        return labels.get(self.value, "Unknown")

    @classmethod
    def from_string(cls, s: str) -> Rating:
        """Create Rating from string label."""
        mapping = {
            "accept": cls.ACCEPT,
            "minor": cls.MINOR,
            "minor changes": cls.MINOR,
            "major": cls.MAJOR,
            "major changes": cls.MAJOR,
            "reject": cls.REJECT,
            "unrated": cls.UNRATED,
        }
        return mapping.get(s.lower(), cls.UNRATED)


@dataclass
class ReviewRecord:
    """Record of a single review decision."""

    trial_id: str
    run_name: str
    rating: Rating
    notes: str = ""
    reviewer: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)
    slice_index: int | None = None  # For per-slice reviews

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "run_name": self.run_name,
            "rating": self.rating.value,
            "rating_label": self.rating.label,
            "notes": self.notes,
            "reviewer": self.reviewer,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "slice_index": self.slice_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewRecord:
        """Create from dictionary."""
        return cls(
            trial_id=data["trial_id"],
            run_name=data["run_name"],
            rating=Rating(data["rating"]),
            notes=data.get("notes", ""),
            reviewer=data.get("reviewer", ""),
            timestamp=data.get("timestamp", ""),
            metrics=data.get("metrics", {}),
            slice_index=data.get("slice_index"),
        )


class RatingManager:
    """Manage review ratings with persistence.

    Tracks ratings for trial segmentations and exports to CSV/JSON
    for analysis and audit.
    """

    def __init__(self, reviews_dir: Path | str | None = None):
        """Initialize rating manager.

        Args:
            reviews_dir: Directory for storing review records.
                        Defaults to "reviews" relative to extension root.
        """
        if reviews_dir is None:
            module_dir = Path(__file__).parent.parent.parent
            reviews_dir = module_dir / "reviews"
        self.reviews_dir = Path(reviews_dir)
        self.reviews_dir.mkdir(parents=True, exist_ok=True)

        # In-memory records for current session
        self._records: list[ReviewRecord] = []
        self._session_file: Path | None = None
        self._reviewer: str = ""

    def set_reviewer(self, name: str) -> None:
        """Set the current reviewer name.

        Args:
            name: Reviewer name/ID for attribution.
        """
        self._reviewer = name

    def start_session(self, run_name: str) -> Path:
        """Start a new review session.

        Args:
            run_name: Name of the optimization run being reviewed.

        Returns:
            Path to session file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{timestamp}_{run_name}"

        session_dir = self.reviews_dir / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)

        self._session_file = session_dir / f"{session_name}.json"
        self._records = []

        # Load existing session if it exists (resume capability)
        if self._session_file.exists():
            self._load_session()

        logger.info(f"Started review session: {self._session_file}")
        return self._session_file

    def _load_session(self) -> None:
        """Load existing session records."""
        if self._session_file and self._session_file.exists():
            try:
                with open(self._session_file) as f:
                    data = json.load(f)
                self._records = [ReviewRecord.from_dict(r) for r in data.get("records", [])]
                logger.info(f"Loaded {len(self._records)} existing records")
            except Exception as e:
                logger.warning(f"Could not load session: {e}")

    def _save_session(self) -> None:
        """Save current session to file."""
        if self._session_file:
            data = {
                "reviewer": self._reviewer,
                "records": [r.to_dict() for r in self._records],
            }
            with open(self._session_file, "w") as f:
                json.dump(data, f, indent=2)

    def rate_trial(
        self,
        trial_id: str,
        run_name: str,
        rating: Rating,
        notes: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> ReviewRecord:
        """Record a rating for a trial.

        Args:
            trial_id: Trial identifier (e.g., "trial_079").
            run_name: Name of the optimization run.
            rating: Quality rating.
            notes: Optional reviewer notes.
            metrics: Optional metrics to store with the rating.

        Returns:
            The created ReviewRecord.
        """
        record = ReviewRecord(
            trial_id=trial_id,
            run_name=run_name,
            rating=rating,
            notes=notes,
            reviewer=self._reviewer,
            metrics=metrics or {},
        )

        # Check for existing rating - update instead of duplicate
        for i, existing in enumerate(self._records):
            if existing.trial_id == trial_id and existing.run_name == run_name:
                self._records[i] = record
                self._save_session()
                logger.info(f"Updated rating for {trial_id}: {rating.label}")
                return record

        self._records.append(record)
        self._save_session()
        logger.info(f"Rated {trial_id}: {rating.label}")
        return record

    def get_rating(self, trial_id: str, run_name: str) -> ReviewRecord | None:
        """Get existing rating for a trial.

        Args:
            trial_id: Trial identifier.
            run_name: Run name.

        Returns:
            ReviewRecord if found, None otherwise.
        """
        for record in reversed(self._records):  # Most recent first
            if record.trial_id == trial_id and record.run_name == run_name:
                return record
        return None

    def get_all_ratings(self, run_name: str | None = None) -> list[ReviewRecord]:
        """Get all ratings, optionally filtered by run.

        Args:
            run_name: Optional run name to filter by.

        Returns:
            List of ReviewRecords.
        """
        if run_name is None:
            return list(self._records)
        return [r for r in self._records if r.run_name == run_name]

    def export_csv(self, output_path: Path | str) -> Path:
        """Export ratings to CSV file.

        Args:
            output_path: Path for CSV file.

        Returns:
            Path to created CSV.
        """
        output_path = Path(output_path)

        fieldnames = [
            "trial_id",
            "run_name",
            "rating",
            "rating_label",
            "notes",
            "reviewer",
            "timestamp",
            "dice",
            "hausdorff_mm",
            "volume_trial_mm3",
            "volume_gold_mm3",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for record in self._records:
                row = record.to_dict()
                # Flatten metrics into row
                row.update(record.metrics)
                writer.writerow(row)

        logger.info(f"Exported {len(self._records)} ratings to {output_path}")
        return output_path

    def export_json(self, output_path: Path | str) -> Path:
        """Export ratings to JSON file.

        Args:
            output_path: Path for JSON file.

        Returns:
            Path to created JSON.
        """
        output_path = Path(output_path)

        data = {
            "exported": datetime.now().isoformat(),
            "reviewer": self._reviewer,
            "total_records": len(self._records),
            "summary": self._compute_summary(),
            "records": [r.to_dict() for r in self._records],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self._records)} ratings to {output_path}")
        return output_path

    def _compute_summary(self) -> dict[str, Any]:
        """Compute summary statistics of ratings."""
        if not self._records:
            return {"total": 0}

        ratings_count = {r.label: 0 for r in Rating}
        for record in self._records:
            ratings_count[record.rating.label] += 1

        return {
            "total": len(self._records),
            "by_rating": ratings_count,
            "accept_rate": ratings_count.get("Accept", 0) / len(self._records),
            "reject_rate": ratings_count.get("Reject", 0) / len(self._records),
        }

    def get_summary(self) -> dict[str, Any]:
        """Get summary of current session ratings."""
        return self._compute_summary()

    def clear_session(self) -> None:
        """Clear current session records."""
        self._records = []
        if self._session_file and self._session_file.exists():
            self._session_file.unlink()
        self._session_file = None
