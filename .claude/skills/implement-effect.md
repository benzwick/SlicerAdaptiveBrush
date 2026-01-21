# Skill: Implement Segment Editor Effect

## Base Class

Inherit from `AbstractScriptedSegmentEditorEffect`:

```python
from slicer.ScriptedLoadableModule import *

class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
    """Adaptive brush segment editor effect."""

    def __init__(self, scriptedEffect):
        scriptedEffect.name = 'Adaptive Brush'
        scriptedEffect.perSegment = True
        AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)
```

## Required Methods

### 1. `__init__(self, scriptedEffect)`
Set effect name and properties.

### 2. `clone(self)`
Return new instance for registration:
```python
def clone(self):
    import qSlicerSegmentationsEditorEffectsPythonQt as effects
    clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
    clonedEffect.setPythonSource(__file__)
    return clonedEffect
```

### 3. `icon(self)`
Return QIcon for toolbar:
```python
def icon(self):
    iconPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons/AdaptiveBrush.png')
    if os.path.exists(iconPath):
        return qt.QIcon(iconPath)
    return qt.QIcon()
```

### 4. `helpText(self)`
Return HTML help string:
```python
def helpText(self):
    return """<html>
    <p>Paint with a brush that adapts to image intensity boundaries.</p>
    <p><b>Usage:</b> Click and drag to paint regions with similar intensity.</p>
    </html>"""
```

### 5. `setupOptionsFrame(self)`
Create UI widgets:
```python
def setupOptionsFrame(self):
    # Radius slider
    self.radiusSlider = ctk.ctkSliderWidget()
    self.radiusSlider.setToolTip("Brush radius in mm")
    self.radiusSlider.minimum = 1
    self.radiusSlider.maximum = 50
    self.radiusSlider.value = 5
    self.scriptedEffect.addLabeledOptionsWidget("Radius (mm):", self.radiusSlider)

    # Edge sensitivity
    self.sensitivitySlider = ctk.ctkSliderWidget()
    self.sensitivitySlider.minimum = 0
    self.sensitivitySlider.maximum = 100
    self.sensitivitySlider.value = 50
    self.scriptedEffect.addLabeledOptionsWidget("Edge Sensitivity:", self.sensitivitySlider)
```

### 6. `processInteractionEvents(self, callerInteractor, eventId, viewWidget)`
Handle mouse events:
```python
def processInteractionEvents(self, callerInteractor, eventId, viewWidget):
    import vtk

    if eventId == vtk.vtkCommand.LeftButtonPressEvent:
        xy = callerInteractor.GetEventPosition()
        ijk = self.xyToIjk(xy, viewWidget)
        if ijk is not None:
            self.applyBrush(ijk, viewWidget)
        return True  # Consume event

    elif eventId == vtk.vtkCommand.MouseMoveEvent:
        if callerInteractor.GetLeftButtonDown():
            xy = callerInteractor.GetEventPosition()
            ijk = self.xyToIjk(xy, viewWidget)
            if ijk is not None:
                self.applyBrush(ijk, viewWidget)
            return True

    elif eventId == vtk.vtkCommand.LeftButtonReleaseEvent:
        self.onMouseRelease()
        return True

    return False  # Let other handlers process
```

## Coordinate Conversion

```python
def xyToIjk(self, xy, viewWidget):
    """Convert screen coordinates to volume IJK."""
    # Get the slice logic
    sliceLogic = viewWidget.sliceLogic()
    sliceNode = sliceLogic.GetSliceNode()

    # Convert XY to RAS
    ras = [0, 0, 0]
    sliceNode.GetXYToRAS().MultiplyPoint([xy[0], xy[1], 0, 1], ras)

    # Get source volume
    sourceVolumeNode = self.scriptedEffect.parameterSetNode().GetSourceVolumeNode()
    if not sourceVolumeNode:
        return None

    # Convert RAS to IJK
    rasToIjk = vtk.vtkMatrix4x4()
    sourceVolumeNode.GetRASToIJKMatrix(rasToIjk)
    ijk = [0, 0, 0, 1]
    rasToIjk.MultiplyPoint([ras[0], ras[1], ras[2], 1], ijk)

    return (int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2])))
```

## Applying Mask to Segment

```python
def applyMaskToSegment(self, maskArray, sourceVolumeNode):
    """Apply numpy mask array to the current segment."""
    import slicer
    import vtk.util.numpy_support as vtknp

    # Get modifier labelmap
    modifierLabelmap = self.scriptedEffect.defaultModifierLabelmap()

    # Get the array from the labelmap
    modifierArray = slicer.util.arrayFromVolume(modifierLabelmap)

    # Apply mask (OR operation to add to existing)
    modifierArray[:] = np.logical_or(modifierArray, maskArray).astype(np.uint8)

    # Mark as modified
    slicer.util.arrayFromVolumeModified(modifierLabelmap)

    # Apply to segment
    self.scriptedEffect.modifySelectedSegmentByLabelmap(
        modifierLabelmap,
        slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd
    )
```

## Effect Registration

In the module's `__init__.py` or main file:

```python
def registerEffect():
    import qSlicerSegmentationsEditorEffectsPythonQt as effects
    scriptedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
    scriptedEffect.setPythonSource(effectFilePath)
    scriptedEffect.self().register()
```
