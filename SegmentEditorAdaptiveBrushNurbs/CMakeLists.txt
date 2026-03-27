#-----------------------------------------------------------------------------
set(MODULE_NAME SegmentEditorAdaptiveBrushNurbs)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/StructureDetector.py
  ${MODULE_NAME}Lib/HexMeshGenerator.py
  ${MODULE_NAME}Lib/NurbsVolumeBuilder.py
  ${MODULE_NAME}Lib/ContainmentValidator.py
  ${MODULE_NAME}Lib/QualityMetrics.py
  ${MODULE_NAME}Lib/Visualization.py
  ${MODULE_NAME}Lib/Exporters/__init__.py
  ${MODULE_NAME}Lib/Exporters/MfemExporter.py
  ${MODULE_NAME}Lib/Exporters/MeshExporter.py
  )

set(MODULE_PYTHON_RESOURCES
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
  add_subdirectory(Testing)
endif()
