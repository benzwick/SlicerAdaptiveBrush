cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(SlicerAdaptiveBrush)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/benzwick/SlicerAdaptiveBrush")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Ben Zwick")
set(EXTENSION_DESCRIPTION "Adaptive brush segment editor effect that segments based on image intensity similarity, stopping at edges and boundaries.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/benzwick/SlicerAdaptiveBrush/main/SlicerAdaptiveBrush.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/benzwick/SlicerAdaptiveBrush/main/Screenshots/main-ui.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SegmentEditorAdaptiveBrush)
add_subdirectory(SegmentEditorAdaptiveBrushTester)
add_subdirectory(SegmentEditorAdaptiveBrushReviewer)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
