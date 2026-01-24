"""Test cases for AdaptiveBrush testing framework.

Test cases are auto-discovered and registered via the @register_test decorator.
Import test modules here to ensure they are loaded.
"""

# Import test modules to register them
# Using explicit re-exports to satisfy linter
from . import test_algorithm_watershed as test_algorithm_watershed
from . import test_ui_options_panel as test_ui_options_panel
from . import test_workflow_basic as test_workflow_basic
