"""Tests for DependencyManager.

Tests the centralized dependency management system for optional packages.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestDependencySpec:
    """Tests for DependencySpec dataclass."""

    def test_dependency_spec_creation(self):
        """Test creating a DependencySpec with all fields."""
        from DependencyManager import DependencySpec

        spec = DependencySpec(
            name="scikit-learn",
            pip_name="scikit-learn",
            version_constraint=">=1.0",
            feature_description="GMM-based intensity analysis",
            import_check="sklearn.mixture",
        )

        assert spec.name == "scikit-learn"
        assert spec.pip_name == "scikit-learn"
        assert spec.version_constraint == ">=1.0"
        assert spec.feature_description == "GMM-based intensity analysis"
        assert spec.import_check == "sklearn.mixture"


class TestDependencyManager:
    """Tests for DependencyManager singleton."""

    @pytest.fixture
    def manager(self):
        """Get a fresh DependencyManager instance for testing."""
        from DependencyManager import DependencyManager

        # Create a fresh instance (not the singleton)
        mgr = DependencyManager.__new__(DependencyManager)
        mgr._initialized = False
        mgr._init_once()
        # Clear session state
        mgr._user_declined = set()
        mgr._install_failed = set()
        return mgr

    def test_singleton_pattern(self):
        """Test that DependencyManager follows singleton pattern."""
        from DependencyManager import dependency_manager, get_dependency_manager

        # Both should return same instance
        assert dependency_manager is get_dependency_manager()

    def test_has_sklearn_in_registry(self, manager):
        """Test that sklearn is registered."""
        assert "sklearn" in manager._dependencies
        spec = manager._dependencies["sklearn"]
        assert spec.pip_name == "scikit-learn"
        assert "sklearn.mixture" in spec.import_check

    def test_has_skimage_in_registry(self, manager):
        """Test that skimage is registered."""
        assert "skimage" in manager._dependencies
        spec = manager._dependencies["skimage"]
        assert spec.pip_name == "scikit-image"
        assert "skimage.segmentation" in spec.import_check

    def test_is_available_returns_bool(self, manager):
        """Test is_available returns a boolean without prompting."""
        # Should return bool, not prompt
        result = manager.is_available("sklearn")
        assert isinstance(result, bool)

    def test_is_available_unknown_key_returns_false(self, manager):
        """Test is_available returns False for unknown dependency key."""
        result = manager.is_available("nonexistent_package")
        assert result is False

    @patch("importlib.util.find_spec")
    def test_is_available_checks_import(self, mock_find_spec, manager):
        """Test is_available uses importlib to check availability."""
        mock_find_spec.return_value = None
        result = manager.is_available("sklearn")
        assert result is False
        mock_find_spec.assert_called()

    @patch("importlib.util.find_spec")
    def test_is_available_true_when_installed(self, mock_find_spec, manager):
        """Test is_available returns True when package is installed."""
        mock_find_spec.return_value = MagicMock()  # Non-None means available
        result = manager.is_available("sklearn")
        assert result is True


class TestEnsureAvailable:
    """Tests for ensure_available method."""

    @pytest.fixture
    def manager(self):
        """Get a fresh DependencyManager instance for testing."""
        from DependencyManager import DependencyManager

        mgr = DependencyManager.__new__(DependencyManager)
        mgr._initialized = False
        mgr._init_once()
        mgr._user_declined = set()
        mgr._install_failed = set()
        return mgr

    @patch("importlib.util.find_spec")
    def test_ensure_returns_true_if_already_available(self, mock_find_spec, manager):
        """Test ensure_available returns True if package already installed."""
        mock_find_spec.return_value = MagicMock()  # Package available
        result = manager.ensure_available("sklearn")
        assert result is True

    @patch("importlib.util.find_spec")
    def test_ensure_returns_false_if_declined(self, mock_find_spec, manager):
        """Test ensure_available returns False if user declined this session."""
        mock_find_spec.return_value = None  # Not available
        manager._user_declined.add("sklearn")

        result = manager.ensure_available("sklearn")
        assert result is False

    @patch("importlib.util.find_spec")
    def test_ensure_returns_false_if_install_failed(self, mock_find_spec, manager):
        """Test ensure_available returns False if install already failed."""
        mock_find_spec.return_value = None
        manager._install_failed.add("sklearn")

        result = manager.ensure_available("sklearn")
        assert result is False

    @patch("importlib.util.find_spec")
    @patch("DependencyManager.HAS_SLICER", False)
    def test_ensure_returns_false_outside_slicer(self, mock_find_spec, manager):
        """Test ensure_available returns False when running outside Slicer."""
        mock_find_spec.return_value = None

        result = manager.ensure_available("sklearn")
        assert result is False

    def test_ensure_records_decline_when_prompt_returns_false(self, manager):
        """Test that declining via _prompt_and_install records the decline."""
        # Simulate what happens when user declines
        # We test this by directly manipulating state since we can't mock slicer easily
        manager._user_declined.add("sklearn")

        # Verify it's remembered
        assert manager._should_skip_prompt("sklearn") is True
        assert "sklearn" in manager._user_declined

    @patch("importlib.util.find_spec")
    def test_ensure_unknown_key_returns_false(self, mock_find_spec, manager):
        """Test ensure_available returns False for unknown key."""
        mock_find_spec.return_value = None
        result = manager.ensure_available("nonexistent")
        assert result is False


class TestInstallFlow:
    """Tests for the installation flow."""

    @pytest.fixture
    def manager(self):
        """Get a fresh DependencyManager instance for testing."""
        from DependencyManager import DependencyManager

        mgr = DependencyManager.__new__(DependencyManager)
        mgr._initialized = False
        mgr._init_once()
        mgr._user_declined = set()
        mgr._install_failed = set()
        return mgr

    def test_get_install_message_sklearn(self, manager):
        """Test installation message formatting for sklearn."""
        message = manager._get_install_message("sklearn")

        assert "scikit-learn" in message
        assert "GMM-based intensity analysis" in message

    def test_get_install_message_skimage(self, manager):
        """Test installation message formatting for skimage."""
        message = manager._get_install_message("skimage")

        assert "scikit-image" in message
        assert "Random Walker" in message

    def test_get_install_message_unknown_returns_none(self, manager):
        """Test get_install_message returns None for unknown key."""
        message = manager._get_install_message("nonexistent")
        assert message is None


class TestSessionMemory:
    """Tests for session-level memory of user choices."""

    @pytest.fixture
    def manager(self):
        """Get a fresh DependencyManager instance."""
        from DependencyManager import DependencyManager

        mgr = DependencyManager.__new__(DependencyManager)
        mgr._initialized = False
        mgr._init_once()
        mgr._user_declined = set()
        mgr._install_failed = set()
        return mgr

    def test_declined_remembered_in_session(self, manager):
        """Test that user decline is remembered within session."""
        manager._user_declined.add("sklearn")

        # Should not re-prompt, just return False
        assert manager._should_skip_prompt("sklearn") is True

    def test_failed_remembered_in_session(self, manager):
        """Test that install failure is remembered within session."""
        manager._install_failed.add("sklearn")

        # Should not re-prompt, just return False
        assert manager._should_skip_prompt("sklearn") is True

    def test_clear_session_resets_memory(self, manager):
        """Test that clear_session resets all memory."""
        manager._user_declined.add("sklearn")
        manager._install_failed.add("skimage")

        manager.clear_session()

        assert len(manager._user_declined) == 0
        assert len(manager._install_failed) == 0


class TestIntegration:
    """Integration tests for DependencyManager with real imports."""

    def test_sklearn_availability_matches_import(self):
        """Test that is_available matches actual import success."""
        from DependencyManager import dependency_manager

        available = dependency_manager.is_available("sklearn")

        # Try actual import
        try:
            from sklearn.mixture import GaussianMixture  # noqa: F401

            can_import = True
        except ImportError:
            can_import = False

        assert available == can_import

    def test_skimage_availability_matches_import(self):
        """Test that is_available matches actual import success for skimage."""
        from DependencyManager import dependency_manager

        available = dependency_manager.is_available("skimage")

        # Try actual import
        try:
            from skimage.segmentation import random_walker  # noqa: F401

            can_import = True
        except ImportError:
            can_import = False

        assert available == can_import
