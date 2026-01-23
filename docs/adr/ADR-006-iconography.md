# ADR-006: Iconography

## Status

Accepted

## Context

The project uses placeholder icons ("Scripted Loadable Template" and generic extension "E") that don't represent the Adaptive Brush functionality. Good icons help users:

- Quickly identify the effect in the Segment Editor toolbar
- Recognize the extension in the Extension Manager
- Understand the tool's purpose at a glance

Segment Editor effect icons follow conventions:

- 21x21 PNG with RGBA (transparency)
- Simple, recognizable at small sizes
- Green color scheme for painting/drawing tools
- Visual metaphor for the tool's function

### Icons Required

| Location | Size | Purpose |
|----------|------|---------|
| `SlicerAdaptiveBrush.png` | ~128x128 | Extension manager |
| `SegmentEditorAdaptiveBrush/Resources/Icons/SegmentEditorAdaptiveBrush.png` | 21x21 | Effect toolbar |
| `SegmentEditorAdaptiveBrushLib/SegmentEditorEffect.png` | 21x21 | Alt effect icon path |

## Decision

**Create "Magic Brush" icons** by modifying Slicer's existing Paint.png:

1. **Effect Icon (21x21)**: Green brush from Paint.png + yellow/white sparkle dots
2. **Extension Icon (128x128)**: Scaled-up version with same concept, more detail

### Why This Approach

- Leverages familiar Slicer visual language (green = paint)
- Sparkles convey "smart/adaptive" without complexity
- Maintains consistency with other segment editor effects
- Distinguishable from standard Paint effect

### Visual Design

**Effect Icon (21x21):**
- Base: Slicer Paint.png (green brush with checkmark)
- Addition: Small sparkle/star pattern in upper-right area
- Colors: Yellow (#FFD700) and white sparkle dots

**Extension Icon (128x128):**
- Larger brush graphic
- More detailed sparkle effects
- Clear at Extension Manager display size

## Alternatives Considered

### LevelTracing Icon

Use existing LevelTracing.png (green dotted circle following intensity boundaries).

**Rejected because:**
- Conceptually accurate but doesn't convey "brush"
- Not unique enough - users might confuse with existing effect

### Custom Wand Design

Create entirely new icon with magic wand concept.

**Rejected because:**
- More complex to create
- Harder to recognize at 21x21 pixels
- Doesn't convey "painting/drawing" operation

### Brain Imagery

Use brain MRI imagery like some SlicerSegmentEditorExtraEffects icons.

**Rejected because:**
- Too detailed for 21x21 pixels
- Medical imagery not necessary for generic tool

## Consequences

### Positive

- Clear visual identity distinct from standard Paint
- Follows Slicer design conventions
- Works at all display sizes
- Simple to maintain/update
- Users can quickly identify "smart brush" vs regular brush

### Negative

- Requires manual image creation (not code-generatable)
- May need adjustment for different themes/backgrounds

## Implementation

Icons created programmatically using Python PIL/Pillow:

1. Load Paint.png as base
2. Add sparkle overlay (yellow/white dots)
3. Export at 21x21 for effect icon
4. Create scaled 128x128 version for extension

## References

- Slicer Paint.png: `Slicer/Modules/Loadable/Segmentations/EditorEffects/Resources/Icons/Paint.png`
- [Slicer icon conventions](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html)
