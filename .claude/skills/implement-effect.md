# Skill: Implement Segment Editor Effect

## Base Class

Inherit from `AbstractScriptedSegmentEditorEffect`:

```python
from SegmentEditorEffects import AbstractScriptedSegmentEditorEffect

class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):
    """Adaptive brush segment editor effect."""

    def __init__(self, scriptedEffect):
        scriptedEffect.name = 'Adaptive Brush'
        scriptedEffect.perSegment = True
        AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)
```

## Required Methods

### 1. `__init__(self, scriptedEffect)`
Set effect name and properties, call parent init.

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
Return QIcon for toolbar (icon in same directory as effect):
```python
def icon(self):
    iconPath = os.path.join(os.path.dirname(__file__), 'SegmentEditorEffect.png')
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

    # Connect signals
    self.radiusSlider.valueChanged.connect(self.onRadiusChanged)
```

### 6. `cleanup(self)` (REQUIRED - Added Jan 2025)
Clean up resources to prevent memory leaks. Called by Slicer before effect deletion.

```python
def cleanup(self):
    """Clean up resources to prevent memory leaks.

    See: https://github.com/Slicer/Slicer/issues/7392
    """
    # Disconnect all signal/slot connections
    widgets_to_disconnect = [
        "radiusSlider",
        "sensitivitySlider",
        # ... all widgets with connections
    ]
    for widget_name in widgets_to_disconnect:
        try:
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.disconnect()
        except Exception:
            pass

    # Clean up VTK pipelines, actors, etc.
    # ...

    # Call parent cleanup
    AbstractScriptedSegmentEditorEffect.cleanup(self)
```

### 7. `processInteractionEvents(self, callerInteractor, eventId, viewWidget)`
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

In the module's main file:

```python
def registerEffect(self):
    import qSlicerSegmentationsEditorEffectsPythonQt as effects
    effectPath = os.path.join(
        os.path.dirname(__file__),
        self.__class__.__name__ + "Lib",
        "SegmentEditorEffect.py",
    )
    scriptedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
    scriptedEffect.setPythonSource(effectPath.replace("\\", "/"))
    scriptedEffect.self().register()
```

## Memory Management Notes

- Always disconnect signals in `cleanup()` to prevent memory leaks
- VTK actors and pipelines should be removed from renderers
- Clear any caches or large data structures
- The `cleanup()` method is called when:
  - The segment editor widget is destroyed
  - The effect is being replaced by another effect
  - Slicer is shutting down
