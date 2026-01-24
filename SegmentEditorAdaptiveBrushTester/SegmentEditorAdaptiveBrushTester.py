"""Slicer testing framework for AdaptiveBrush.

This module provides:
1. TestRunner for executing registered test cases
2. Interactive testing panel for manual testing
3. Screenshot capture and metrics collection
4. Action recording for reproducibility

Usage in Slicer Python console:
    import SegmentEditorAdaptiveBrushTester as tester
    runner = tester.TestRunner()
    result = runner.run_suite("all")
"""

from __future__ import annotations

import logging
from pathlib import Path

import qt
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
)

# Configure logging for the testing framework
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
)

logger = logging.getLogger(__name__)


class SegmentEditorAdaptiveBrushTester(ScriptedLoadableModule):
    """Slicer module for testing AdaptiveBrush.

    Provides an interactive panel for running tests and manual testing.
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Adaptive Brush Tester")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Developer Tools")]
        self.parent.dependencies = ["SegmentEditorAdaptiveBrush"]
        self.parent.contributors = ["Ben Zwick"]
        self.parent.helpText = _(
            """
Testing framework for the Adaptive Brush segment editor effect.

Provides:
- Automated test execution
- Screenshot capture
- Metrics collection
- Manual testing with action recording

For more information, see the <a href="https://github.com/benzwick/SlicerAdaptiveBrush">
project documentation</a>.
"""
        )
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = _(
            """
Testing framework for SlicerAdaptiveBrush development.
"""
        )
        self.parent.hidden = False


class SegmentEditorAdaptiveBrushTesterWidget(ScriptedLoadableModuleWidget):
    """Interactive testing panel widget."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self._test_runner = None
        self._test_run_folder = None
        self._action_recorder = None
        self._screenshot_capture = None
        self._screenshot_counter = 0

    def setup(self):
        """Set up the widget UI."""
        ScriptedLoadableModuleWidget.setup(self)

        # Import library classes
        from SegmentEditorAdaptiveBrushTesterLib import (
            ScreenshotCapture,
            TestRunner,
        )

        self._test_runner = TestRunner()
        self._screenshot_capture = ScreenshotCapture()

        # Discover tests
        self._test_runner.discover_tests()

        # --- Test Run Section ---
        testRunCollapsible = ctk.ctkCollapsibleButton()
        testRunCollapsible.text = _("Test Run")
        self.layout.addWidget(testRunCollapsible)
        testRunLayout = qt.QFormLayout(testRunCollapsible)

        # Status label
        self.statusLabel = qt.QLabel(_("Status: Not started"))
        testRunLayout.addRow(self.statusLabel)

        # Test run folder label
        self.folderLabel = qt.QLabel(_("Folder: -"))
        self.folderLabel.setWordWrap(True)
        testRunLayout.addRow(self.folderLabel)

        # Suite selector
        self.suiteSelector = qt.QComboBox()
        self.suiteSelector.addItem("all", "all")
        for category in self._test_runner.list_categories():
            self.suiteSelector.addItem(category, category)
        testRunLayout.addRow(_("Suite:"), self.suiteSelector)

        # Run tests button
        self.runTestsButton = qt.QPushButton(_("Run Tests"))
        self.runTestsButton.connect("clicked(bool)", self.onRunTests)
        testRunLayout.addRow(self.runTestsButton)

        # --- Manual Testing Section ---
        manualCollapsible = ctk.ctkCollapsibleButton()
        manualCollapsible.text = _("Manual Testing")
        manualCollapsible.collapsed = True
        self.layout.addWidget(manualCollapsible)
        manualLayout = qt.QFormLayout(manualCollapsible)

        # Recording status
        self.recordingLabel = qt.QLabel(_("Recording: Off"))
        manualLayout.addRow(self.recordingLabel)

        # Start/Stop recording
        recordingButtonsLayout = qt.QHBoxLayout()

        self.startRecordingButton = qt.QPushButton(_("Start Recording"))
        self.startRecordingButton.connect("clicked(bool)", self.onStartRecording)
        recordingButtonsLayout.addWidget(self.startRecordingButton)

        self.stopRecordingButton = qt.QPushButton(_("Stop Recording"))
        self.stopRecordingButton.enabled = False
        self.stopRecordingButton.connect("clicked(bool)", self.onStopRecording)
        recordingButtonsLayout.addWidget(self.stopRecordingButton)

        manualLayout.addRow(recordingButtonsLayout)

        # New Group button and current group label
        groupLayout = qt.QHBoxLayout()

        self.newGroupButton = qt.QPushButton(_("New Group"))
        self.newGroupButton.enabled = False
        self.newGroupButton.connect("clicked(bool)", self.onNewGroup)
        groupLayout.addWidget(self.newGroupButton)

        self.currentGroupLabel = qt.QLabel(_("Group: (none)"))
        groupLayout.addWidget(self.currentGroupLabel)

        manualLayout.addRow(groupLayout)

        # Screenshot button
        self.screenshotButton = qt.QPushButton(_("Take Screenshot"))
        self.screenshotButton.enabled = False
        self.screenshotButton.connect("clicked(bool)", self.onTakeScreenshot)
        manualLayout.addRow(self.screenshotButton)

        # Note input
        self.noteEdit = qt.QLineEdit()
        self.noteEdit.setPlaceholderText(_("Enter observation..."))
        manualLayout.addRow(_("Note:"), self.noteEdit)

        self.addNoteButton = qt.QPushButton(_("Add Note"))
        self.addNoteButton.enabled = False
        self.addNoteButton.connect("clicked(bool)", self.onAddNote)
        manualLayout.addRow(self.addNoteButton)

        # Pass/Fail buttons
        passFailLayout = qt.QHBoxLayout()

        self.markPassButton = qt.QPushButton(_("Mark Pass"))
        self.markPassButton.enabled = False
        self.markPassButton.setStyleSheet("background-color: #4CAF50; color: white;")
        self.markPassButton.connect("clicked(bool)", self.onMarkPass)
        passFailLayout.addWidget(self.markPassButton)

        self.markFailButton = qt.QPushButton(_("Mark Fail"))
        self.markFailButton.enabled = False
        self.markFailButton.setStyleSheet("background-color: #f44336; color: white;")
        self.markFailButton.connect("clicked(bool)", self.onMarkFail)
        passFailLayout.addWidget(self.markFailButton)

        manualLayout.addRow(passFailLayout)

        # Actions log
        self.actionsLog = qt.QTextEdit()
        self.actionsLog.setReadOnly(True)
        self.actionsLog.setMaximumHeight(150)
        manualLayout.addRow(_("Recent Actions:"), self.actionsLog)

        # --- Results Section ---
        resultsCollapsible = ctk.ctkCollapsibleButton()
        resultsCollapsible.text = _("Results")
        resultsCollapsible.collapsed = True
        self.layout.addWidget(resultsCollapsible)
        resultsLayout = qt.QFormLayout(resultsCollapsible)

        # Results text
        self.resultsText = qt.QTextEdit()
        self.resultsText.setReadOnly(True)
        resultsLayout.addRow(self.resultsText)

        # Open folder button
        self.openFolderButton = qt.QPushButton(_("Open Output Folder"))
        self.openFolderButton.enabled = False
        self.openFolderButton.connect("clicked(bool)", self.onOpenFolder)
        resultsLayout.addRow(self.openFolderButton)

        # Add spacer
        self.layout.addStretch(1)

    def onRunTests(self):
        """Run the selected test suite."""
        suite = self.suiteSelector.currentData

        self.statusLabel.text = _(f"Status: Running {suite}...")
        self.runTestsButton.enabled = False

        # Force UI update
        slicer.app.processEvents()

        try:
            result = self._test_runner.run_suite(suite)

            # Update status
            status = "Passed" if result.passed else "Failed"
            self.statusLabel.text = _(
                f"Status: {status} ({result.passed_count}/{result.total_count})"
            )
            self.folderLabel.text = _(f"Folder: {result.output_folder}")

            # Save test run folder for manual testing
            from SegmentEditorAdaptiveBrushTesterLib import TestRunFolder

            self._test_run_folder = TestRunFolder(result.output_folder)

            # Update results text
            from SegmentEditorAdaptiveBrushTesterLib import ReportGenerator

            generator = ReportGenerator()
            report = generator.generate_markdown(result)
            self.resultsText.setMarkdown(report)

            # Copy Slicer log
            self._test_run_folder.copy_slicer_log()

            # Enable output folder button
            self.openFolderButton.enabled = True

        except Exception as e:
            self.statusLabel.text = _(f"Status: Error - {e}")
            logger.exception("Error running tests")

        finally:
            self.runTestsButton.enabled = True

    def onStartRecording(self):
        """Start recording manual actions."""
        if self._test_run_folder is None:
            # Create a new test run folder for manual testing
            from SegmentEditorAdaptiveBrushTesterLib import TestRunFolder

            base_path = Path(__file__).parent.parent / "test_runs"
            self._test_run_folder = TestRunFolder.create(base_path, "manual")
            self.folderLabel.text = _(f"Folder: {self._test_run_folder.path}")
            self.openFolderButton.enabled = True

        # Set up screenshot capture with base folder
        self._screenshot_capture.set_base_folder(self._test_run_folder.screenshots_folder)

        # Create initial group (prompt for name)
        group_name = qt.QInputDialog.getText(
            slicer.util.mainWindow(),
            _("New Screenshot Group"),
            _("Enter group name:"),
            qt.QLineEdit.Normal,
            "manual_test",
        )
        if group_name:
            self._screenshot_capture.new_group(group_name)
            self.currentGroupLabel.text = _(f"Group: {group_name}")
        else:
            self._screenshot_capture.new_group("manual_test")
            self.currentGroupLabel.text = _("Group: manual_test")

        from SegmentEditorAdaptiveBrushTesterLib import ActionRecorder

        self._action_recorder = ActionRecorder(self._test_run_folder)
        self._action_recorder.start()

        self._updateRecordingUI(True)
        self._logAction("Recording started (auto-capture enabled)")
        self._logAction("Paint events will be recorded automatically")

    def onStopRecording(self):
        """Stop recording manual actions."""
        if self._action_recorder:
            action_count = self._action_recorder.action_count
            self._action_recorder.stop()
            self._logAction(f"Recording stopped ({action_count} actions captured)")
            self._action_recorder = None

        # Save screenshot manifest
        self._screenshot_capture.save_manifest()
        self._logAction("Screenshot manifest saved")

        self._updateRecordingUI(False)
        self.currentGroupLabel.text = _("Group: (none)")

    def onNewGroup(self):
        """Create a new screenshot group."""
        group_name = qt.QInputDialog.getText(
            slicer.util.mainWindow(),
            _("New Screenshot Group"),
            _("Enter group name:"),
            qt.QLineEdit.Normal,
            "",
        )
        if group_name:
            self._screenshot_capture.new_group(group_name)
            self.currentGroupLabel.text = _(f"Group: {group_name}")
            self._logAction(f"Created new group: {group_name}")

    def _updateRecordingUI(self, recording: bool):
        """Update UI for recording state."""
        if recording:
            self.recordingLabel.text = _("Recording: ● On (auto-capture)")
            self.recordingLabel.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.recordingLabel.text = _("Recording: Off")
            self.recordingLabel.setStyleSheet("")
        self.startRecordingButton.enabled = not recording
        self.stopRecordingButton.enabled = recording
        self.newGroupButton.enabled = recording
        self.screenshotButton.enabled = recording
        self.addNoteButton.enabled = recording
        self.markPassButton.enabled = recording
        self.markFailButton.enabled = recording

    def onTakeScreenshot(self):
        """Take a screenshot (auto-numbered within current group)."""
        if not self._test_run_folder:
            return

        # Prompt for description (optional, Slicer's PythonQt returns just the text)
        description = qt.QInputDialog.getText(
            slicer.util.mainWindow(),
            _("Screenshot Description"),
            _("Enter description (optional):"),
            qt.QLineEdit.Normal,
            "",
        )

        # Take screenshot with auto-numbering
        info = self._screenshot_capture.screenshot(description or "")

        if self._action_recorder:
            self._action_recorder.record_screenshot(
                f"{info.group}/{info.number:03d}", description or ""
            )

        self._logAction(f"Screenshot: {info.group}/{info.filename}")

    def onAddNote(self):
        """Add a note."""
        note = self.noteEdit.text.strip()
        if not note:
            return

        if self._action_recorder:
            self._action_recorder.record_note(note)

        self._logAction(f"Note: {note}")
        self.noteEdit.clear()

    def onMarkPass(self):
        """Mark current test as pass."""
        # Slicer's PythonQt returns just the text, empty string if cancelled
        reason = qt.QInputDialog.getText(
            slicer.util.mainWindow(),
            _("Pass Reason"),
            _("Enter reason (optional):"),
            qt.QLineEdit.Normal,
            "",
        )

        # Empty string means cancelled or no reason (both ok for pass)
        if self._action_recorder:
            self._action_recorder.record_pass(reason if reason else "")

        self._logAction(f"PASS: {reason}" if reason else "PASS")

    def onMarkFail(self):
        """Mark current test as fail."""
        # Slicer's PythonQt returns just the text, empty string if cancelled
        reason = qt.QInputDialog.getText(
            slicer.util.mainWindow(),
            _("Fail Reason"),
            _("Enter reason:"),
            qt.QLineEdit.Normal,
            "",
        )

        if not reason:
            return

        if self._action_recorder:
            self._action_recorder.record_fail(reason)

        self._logAction(f"FAIL: {reason}")

    def onOpenFolder(self):
        """Open the output folder in file manager."""
        if self._test_run_folder:
            qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(self._test_run_folder.path)))

    def _logAction(self, message: str):
        """Log an action to the UI."""
        self.actionsLog.append(message)


# Import ctk for collapsible buttons
try:
    import ctk
except ImportError:
    # ctk not available outside Slicer
    pass
