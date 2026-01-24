#-----------------------------------------------------------------------------
set(MODULE_NAME SegmentEditorAdaptiveBrushTester)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/TestCase.py
  ${MODULE_NAME}Lib/TestContext.py
  ${MODULE_NAME}Lib/TestRegistry.py
  ${MODULE_NAME}Lib/TestRunner.py
  ${MODULE_NAME}Lib/ScreenshotCapture.py
  ${MODULE_NAME}Lib/MetricsCollector.py
  ${MODULE_NAME}Lib/TestRunFolder.py
  ${MODULE_NAME}Lib/ActionRecorder.py
  ${MODULE_NAME}Lib/ReportGenerator.py
  TestCases/__init__.py
  TestCases/test_workflow_basic.py
  TestCases/test_algorithm_watershed.py
  TestCases/test_ui_options_panel.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  )
