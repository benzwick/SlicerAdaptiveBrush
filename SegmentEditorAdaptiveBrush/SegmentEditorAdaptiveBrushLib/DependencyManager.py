"""Dependency Manager for optional Python packages.

This module provides centralized management for optional dependencies
(scikit-learn, scikit-image) with user confirmation before installation.

Follows 3D Slicer best practices:
- Uses slicer.util.pip_install() for installation
- Uses slicer.util.confirmOkCancelDisplay() to ask user first
- Installs at runtime when needed, NOT at startup
"""

import importlib.util
import logging
from dataclasses import dataclass
from typing import Optional

# Check if we're running inside Slicer
try:
    import slicer

    HAS_SLICER = True
except ImportError:
    HAS_SLICER = False


@dataclass
class DependencySpec:
    """Specification for an optional dependency.

    Attributes:
        name: Human-readable package name (e.g., "scikit-learn")
        pip_name: Name for pip install (e.g., "scikit-learn")
        version_constraint: Version constraint (e.g., ">=1.0")
        feature_description: Description of feature requiring this package
        import_check: Module path to check availability (e.g., "sklearn.mixture")
    """

    name: str
    pip_name: str
    version_constraint: str
    feature_description: str
    import_check: str


class DependencyManager:
    """Centralized manager for optional Python package dependencies.

    This class follows the singleton pattern to maintain session state
    (user choices, install failures) across the application.

    Usage:
        from DependencyManager import dependency_manager

        # Check availability without prompting
        if dependency_manager.is_available("sklearn"):
            from sklearn.mixture import GaussianMixture
            # use GMM

        # Prompt and install if needed
        if dependency_manager.ensure_available("sklearn"):
            from sklearn.mixture import GaussianMixture
            # use GMM
        else:
            # use fallback
    """

    _instance: Optional["DependencyManager"] = None
    _initialized: bool

    def __new__(cls) -> "DependencyManager":
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _init_once(self) -> None:
        """Initialize instance state (called once per singleton)."""
        if self._initialized:
            return

        self._dependencies: dict[str, DependencySpec] = {
            "sklearn": DependencySpec(
                name="scikit-learn",
                pip_name="scikit-learn",
                version_constraint=">=1.0",
                feature_description="GMM-based intensity analysis",
                import_check="sklearn.mixture",
            ),
            "skimage": DependencySpec(
                name="scikit-image",
                pip_name="scikit-image",
                version_constraint=">=0.19",
                feature_description="Random Walker algorithm",
                import_check="skimage.segmentation",
            ),
        }

        # Session state - tracks user choices, failures, and in-progress installs
        self._user_declined: set[str] = set()
        self._install_failed: set[str] = set()
        self._install_in_progress: set[str] = set()

        self._initialized = True

    def __init__(self) -> None:
        """Initialize the dependency manager."""
        self._init_once()

    def is_available(self, key: str) -> bool:
        """Check if a dependency is available without prompting.

        Args:
            key: Dependency key (e.g., "sklearn", "skimage")

        Returns:
            True if the package is installed and importable
        """
        if key not in self._dependencies:
            logging.debug(f"Unknown dependency key: {key}")
            return False

        spec = self._dependencies[key]
        return importlib.util.find_spec(spec.import_check.split(".")[0]) is not None

    def ensure_available(self, key: str) -> bool:
        """Ensure a dependency is available, prompting to install if needed.

        This method:
        1. Returns True immediately if already installed
        2. Returns False if user previously declined or install failed
        3. Shows a confirmation dialog if not installed
        4. Installs the package if user confirms
        5. Returns True if installation succeeds, False otherwise

        Args:
            key: Dependency key (e.g., "sklearn", "skimage")

        Returns:
            True if the package is available (was installed or just installed)
        """
        # Unknown key
        if key not in self._dependencies:
            logging.warning(f"Unknown dependency key: {key}")
            return False

        # Already available
        if self.is_available(key):
            return True

        # Previously declined or failed
        if self._should_skip_prompt(key):
            logging.debug(f"Skipping prompt for {key} (previously declined or failed)")
            return False

        # Can only install inside Slicer
        if not HAS_SLICER:
            logging.info(
                f"Cannot install {key} outside Slicer. "
                f"Install manually: pip install {self._dependencies[key].pip_name}"
            )
            return False

        # Prompt and install
        return self._prompt_and_install(key)

    def _should_skip_prompt(self, key: str) -> bool:
        """Check if we should skip prompting for this dependency.

        Args:
            key: Dependency key

        Returns:
            True if user declined, install failed, or install in progress
        """
        return (
            key in self._user_declined
            or key in self._install_failed
            or key in self._install_in_progress
        )

    def _get_install_message(self, key: str) -> Optional[str]:
        """Get the installation confirmation message.

        Args:
            key: Dependency key

        Returns:
            Formatted message string, or None if key unknown
        """
        if key not in self._dependencies:
            return None

        spec = self._dependencies[key]
        return (
            f"This feature requires '{spec.name}'.\n\n"
            f"Feature: {spec.feature_description}\n\n"
            f"Would you like to install it now?\n\n"
            f"(Package: {spec.pip_name}{spec.version_constraint})"
        )

    def _is_headless(self) -> bool:
        """Check if running in headless/CI mode where dialogs would block.

        Returns:
            True if dialogs should be skipped
        """
        import os

        # Check for CI environment variables
        ci_vars = ["CI", "GITHUB_ACTIONS", "JENKINS_HOME", "TRAVIS", "GITLAB_CI"]
        if any(os.environ.get(var) for var in ci_vars):
            return True

        # Check if Slicer main window is available
        if HAS_SLICER:
            main_window = slicer.util.mainWindow()
            if main_window is None:
                return True

        return False

    def _prompt_and_install(self, key: str) -> bool:
        """Show confirmation dialog and install if user accepts.

        Args:
            key: Dependency key

        Returns:
            True if installation succeeded
        """
        # Skip dialogs in headless/CI mode to prevent hanging
        if self._is_headless():
            logging.info(f"Headless mode: skipping install prompt for {key}")
            return False

        spec = self._dependencies[key]
        message = self._get_install_message(key)

        if message is None:
            return False

        # Mark as in-progress to prevent duplicate prompts from concurrent events
        self._install_in_progress.add(key)

        try:
            # Show confirmation dialog
            user_accepted = slicer.util.confirmOkCancelDisplay(
                message, windowTitle=f"Install {spec.name}?"
            )

            if not user_accepted:
                logging.info(f"User declined to install {spec.name}")
                self._user_declined.add(key)
                return False

            # Attempt installation
            try:
                logging.info(f"Installing {spec.pip_name}{spec.version_constraint}...")
                slicer.util.pip_install(f"{spec.pip_name}{spec.version_constraint}")
                logging.info(f"Successfully installed {spec.name}")
                return True

            except Exception as e:
                logging.error(f"Failed to install {spec.name}: {e}")
                self._install_failed.add(key)

                # Show error to user
                slicer.util.errorDisplay(
                    f"Failed to install {spec.name}.\n\n"
                    f"Error: {e}\n\n"
                    f"You can try installing manually:\n"
                    f"  pip install {spec.pip_name}",
                    windowTitle="Installation Failed",
                )
                return False
        finally:
            # Always clear in-progress flag
            self._install_in_progress.discard(key)

    def clear_session(self) -> None:
        """Clear session state (declined choices, failures, and in-progress).

        Call this if you want to allow re-prompting for dependencies
        that were previously declined or failed.
        """
        self._user_declined.clear()
        self._install_failed.clear()
        self._install_in_progress.clear()
        logging.debug("Cleared dependency manager session state")


# Singleton instance
dependency_manager = DependencyManager()


def get_dependency_manager() -> DependencyManager:
    """Get the singleton DependencyManager instance.

    Returns:
        The global DependencyManager instance
    """
    return dependency_manager
