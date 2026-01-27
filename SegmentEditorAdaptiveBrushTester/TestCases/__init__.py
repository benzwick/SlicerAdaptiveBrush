"""Test cases for AdaptiveBrush testing framework.

Test cases are auto-discovered and registered via the @register_test decorator.
Import test modules here to ensure they are loaded.
"""

# Import test modules to register them
# Using explicit re-exports to satisfy linter

# Algorithm tests
from . import test_algorithm_all as test_algorithm_all
from . import test_algorithm_watershed as test_algorithm_watershed

# Visualization tests
from . import test_brush_visualization as test_brush_visualization

# Optimization tests
from . import test_optimization_tumor as test_optimization_tumor

# Painting tests
from . import test_painting_operations as test_painting_operations

# Regression tests
from . import test_regression_gold as test_regression_gold

# Reviewer integration tests
from . import test_reviewer_integration as test_reviewer_integration
from . import test_reviewer_ui_bookmarks as test_reviewer_ui_bookmarks
from . import test_reviewer_ui_keyboard as test_reviewer_ui_keyboard
from . import test_reviewer_ui_rating as test_reviewer_ui_rating

# Reviewer UI tests
from . import test_reviewer_ui_slice_navigation as test_reviewer_ui_slice_navigation
from . import test_reviewer_ui_visualization as test_reviewer_ui_visualization
from . import test_reviewer_ui_workflow_playback as test_reviewer_ui_workflow_playback
from . import test_reviewer_unit_bookmarks as test_reviewer_unit_bookmarks

# Reviewer unit tests
from . import test_reviewer_unit_sequence as test_reviewer_unit_sequence
from . import test_reviewer_unit_viewgroup as test_reviewer_unit_viewgroup

# UI tests
from . import test_ui_options_panel as test_ui_options_panel

# Wizard tests
from . import test_wizard_workflow as test_wizard_workflow

# Workflow tests
from . import test_workflow_basic as test_workflow_basic
