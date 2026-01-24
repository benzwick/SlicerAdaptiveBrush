"""Test case registry for discovering and managing test cases.

Test cases are registered via the @register_test decorator or by
calling TestRegistry.register() directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .TestCase import TestCase, TestCaseInfo

logger = logging.getLogger(__name__)


class TestRegistry:
    """Registry of available test cases.

    Test cases register themselves using the @register_test decorator
    or by calling TestRegistry.register() directly.

    Example:
        @register_test(category="algorithm")
        class TestAlgorithmWatershed(TestCase):
            name = "algorithm_watershed"
            description = "Test watershed algorithm"
            ...

        # Or manually:
        TestRegistry.register(TestAlgorithmWatershed)
    """

    _tests: dict[str, TestCaseInfo] = {}

    @classmethod
    def register(cls, test_cls: type[TestCase], category: str | None = None) -> None:
        """Register a test case class.

        Args:
            test_cls: TestCase subclass to register.
            category: Optional category override.
        """
        from .TestCase import TestCaseInfo

        # Create a temporary instance to get name/description
        # (they're class attributes, but we need to handle defaults)
        name = getattr(test_cls, "name", "") or test_cls.__name__
        description = getattr(test_cls, "description", "")
        cat = category or getattr(test_cls, "category", "general")

        info = TestCaseInfo(
            name=name,
            description=description,
            category=cat,
            cls=test_cls,
        )
        cls._tests[name] = info
        logger.debug(f"Registered test: {name} (category: {cat})")

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a test case by name.

        Args:
            name: Name of test to unregister.

        Returns:
            True if test was found and removed.
        """
        if name in cls._tests:
            del cls._tests[name]
            logger.debug(f"Unregistered test: {name}")
            return True
        return False

    @classmethod
    def get(cls, name: str) -> TestCaseInfo | None:
        """Get a test case by name.

        Args:
            name: Name of test to get.

        Returns:
            TestCaseInfo or None if not found.
        """
        return cls._tests.get(name)

    @classmethod
    def list_tests(cls, category: str | None = None) -> list[TestCaseInfo]:
        """List all registered test cases.

        Args:
            category: Optional category filter.

        Returns:
            List of TestCaseInfo objects.
        """
        tests = list(cls._tests.values())
        if category:
            tests = [t for t in tests if t.category == category]
        return sorted(tests, key=lambda t: t.name)

    @classmethod
    def list_categories(cls) -> list[str]:
        """List all unique categories.

        Returns:
            Sorted list of category names.
        """
        categories = set(t.category for t in cls._tests.values())
        return sorted(categories)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tests. Mainly for testing."""
        cls._tests.clear()
        logger.debug("Cleared all registered tests")


def register_test(
    category: str = "general",
) -> callable:
    """Decorator to register a test case class.

    Usage:
        @register_test(category="algorithm")
        class TestAlgorithmWatershed(TestCase):
            name = "algorithm_watershed"
            description = "Test watershed algorithm"
            ...

    Args:
        category: Test category for filtering.

    Returns:
        Class decorator that registers the test.
    """

    def decorator(cls: type) -> type:
        TestRegistry.register(cls, category=category)
        return cls

    return decorator
