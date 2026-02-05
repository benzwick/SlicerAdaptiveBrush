#-----------------------------------------------------------------------------
set(MODULE_NAME SegmentEditorAdaptiveBrush)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/DependencyManager.py
  ${MODULE_NAME}Lib/EmbeddedWizardUI.py
  ${MODULE_NAME}Lib/IntensityAnalyzer.py
  ${MODULE_NAME}Lib/ParameterRecommender.py
  ${MODULE_NAME}Lib/ParameterWizard.py
  ${MODULE_NAME}Lib/PerformanceCache.py
  ${MODULE_NAME}Lib/SegmentEditorEffect.py
  ${MODULE_NAME}Lib/WizardAnalyzer.py
  ${MODULE_NAME}Lib/WizardDataStructures.py
  ${MODULE_NAME}Lib/WizardSampler.py
  )

set(MODULE_PYTHON_RESOURCES
  ${MODULE_NAME}Lib/SegmentEditorEffect.png
  Resources/Icons/${MODULE_NAME}.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
