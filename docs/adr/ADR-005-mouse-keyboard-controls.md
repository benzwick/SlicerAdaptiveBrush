# ADR-005: Mouse and Keyboard Controls

## Status

Accepted

## Context

The adaptive brush needs to support both **Add** and **Erase** modes to allow users to remove voxels from segments using the same intelligent edge-aware algorithms.

3D Slicer has reserved mouse and keyboard shortcuts that we must not conflict with:

| Action | Slicer Operation |
|--------|------------------|
| Right-click + drag | Zoom slice view |
| Middle-click + drag | Pan slice view |
| Shift + left-click + drag | Pan slice view |
| Ctrl + mouse wheel | Zoom |
| Shift + mouse wheel | Adjust brush size (segment editor) |

We need input methods that:
- Are discoverable for new users
- Allow quick toggling for power users
- Don't conflict with standard Slicer navigation
- Provide clear visual feedback

## Decision

Implement **three complementary input methods** for controlling Add/Erase mode:

### 1. UI Radio Buttons (Primary)

Add/Erase radio buttons in the Brush Settings section:
```
Mode:  ○ Add    ○ Erase
```

- Most discoverable option
- Clear, explicit state
- Works for all users regardless of input device

### 2. Ctrl + Left-Click (Modifier)

Holding Ctrl while painting **temporarily inverts** the current mode:
- If UI is set to "Add", Ctrl+paint erases
- If UI is set to "Erase", Ctrl+paint adds

This is a **temporary** inversion - releasing Ctrl returns to the UI-selected mode.

**Rationale:** Ctrl+left-click is available in slice views (not used by Slicer).

### 3. Middle + Left-Click (Modifier)

Holding middle button while left-clicking **temporarily inverts** the current mode (same as Ctrl):
- If UI is set to "Add", Middle+left-click erases
- If UI is set to "Erase", Middle+left-click adds

**Rationale:** Middle button alone still allows pan. Middle+left provides a one-handed alternative to Ctrl+left for users who prefer it.

### 4. Scroll Wheel Controls

| Action | Operation |
|--------|-----------|
| Shift + scroll | Adjust brush radius (±20% per notch) |
| Ctrl + Shift + scroll | Adjust threshold zone (±5% per notch) |

**Note:** Ctrl + scroll is reserved by Slicer for zoom. Slicer's built-in Shift+scroll
only works for the standard brush effect, so we implement our own for Adaptive Brush.

### Visual Feedback

| Mode | Outer Circle | Inner Circle |
|------|--------------|--------------|
| Add | Yellow (1.0, 0.9, 0.1) | Cyan (0.2, 0.9, 1.0) |
| Erase | Red/Orange (1.0, 0.3, 0.1) | Light Red (1.0, 0.5, 0.3) |

### Mode Lock During Stroke

The mode is **locked at stroke start** and cannot change mid-stroke:
- Pressing/releasing Ctrl during a stroke has no effect
- This prevents accidental mode changes that would confuse users
- Each new stroke can use a different mode

## Consequences

### Positive

- **Multiple input options**: UI for discoverability, keyboard for speed, middle-click for quick toggle
- **No conflicts**: All input methods are available in Slicer slice views
- **Clear visual feedback**: Brush color immediately shows current mode
- **Predictable behavior**: Mode lock prevents mid-stroke confusion

### Negative

- **Three ways to do one thing**: Could be confusing, but each serves different use cases
- **Middle+left requires coordination**: Users must hold middle before clicking left

### Mitigation

- Tooltips explain all input methods
- Help text documents available shortcuts
- Visual feedback makes current mode obvious

## Implementation

### State Variables

```python
self.eraseMode = False  # UI-selected mode
self._currentStrokeEraseMode = False  # Locked mode for current stroke
self._isMiddleButtonHeld = False  # Track middle button for erase modifier
```

### Event Processing

```python
def processInteractionEvents(self, callerInteractor, eventId, viewWidget):
    # Track middle button state
    if eventId == vtk.vtkCommand.MiddleButtonPressEvent:
        self._isMiddleButtonHeld = True
        return False  # Don't consume - let pan work

    elif eventId == vtk.vtkCommand.MiddleButtonReleaseEvent:
        self._isMiddleButtonHeld = False
        return False

    isCtrlPressed = callerInteractor.GetControlKey()
    isModifierActive = bool(isCtrlPressed) or self._isMiddleButtonHeld
    effectiveEraseMode = self.eraseMode != isModifierActive  # XOR

    if eventId == vtk.vtkCommand.LeftButtonPressEvent:
        self._currentStrokeEraseMode = effectiveEraseMode  # Lock for stroke
        ...
```

### Mask Application

```python
def applyMaskToSegment(self, mask, erase=False):
    if erase:
        mode = slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeRemove
    else:
        mode = slicer.qSlicerSegmentEditorAbstractEffect.ModificationModeAdd
    self.scriptedEffect.modifySelectedSegmentByLabelmap(modifierLabelmap, mode)
```

## Alternatives Considered

### Alt + Left-Click

Could use Alt as modifier instead of Ctrl. Rejected because:
- Alt+click is sometimes used by window managers
- Ctrl is more standard for modifier operations

### Right-Click for Erase

Rejected because right-click is reserved for zoom in Slicer slice views.

### Scroll Wheel Toggle

Could use scroll wheel to toggle modes. Rejected because:
- Scroll during drawing would be awkward
- Shift+scroll already adjusts brush size
- Could save for future "adjust sensitivity while drawing" feature

## References

- [Slicer Mouse Keyboard Shortcuts](https://slicer.readthedocs.io/en/latest/user_guide/user_interface.html#mouse-keyboard-shortcuts)
- VTK Events: `MiddleButtonPressEvent`, `MouseWheelForwardEvent`, etc.
