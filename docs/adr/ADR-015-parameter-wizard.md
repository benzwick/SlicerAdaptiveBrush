# ADR-015: Quick Select Parameters Wizard

## Status

Implemented

The Quick Select Parameters Wizard is fully functional with:
- Interactive foreground/background sampling in slice views
- Optional boundary tracing for shape analysis
- Intensity distribution analysis with separation scoring
- Shape analysis (diameter, circularity, boundary roughness)
- Algorithm recommendation with decision tree
- Modality and structure-type awareness
- Confidence scoring and warnings
- Alternative algorithm suggestions

## Context

Users new to the Adaptive Brush must manually configure algorithm and parameters
without knowing which settings work best for their specific image and target.
The current presets help but require domain knowledge to select appropriately.

### Challenges

- Users don't know which algorithm suits their imaging modality
- Parameter tuning requires understanding of intensity distributions
- No guidance on brush size relative to structure size
- Edge sensitivity requires trial-and-error to optimize
- Different structure types (tumors, vessels, organs) have different requirements

## Decision

Add an interactive wizard that helps users configure optimal parameters through:

1. **Interactive sampling** - Paint foreground, background, and optionally trace boundary
2. **Automatic analysis** - Intensity statistics, separation scores, shape metrics
3. **Smart recommendations** - Algorithm and parameter suggestions with explanations
4. **Modality awareness** - Questions about imaging type and target structure

### Wizard Workflow

```
Step 1: Sample Foreground  → Paint inside target structure
Step 2: Sample Background  → Paint outside/around structure
Step 3: Trace Boundary     → Optional: draw along edge
Step 4: Questions          → Modality, structure type, priority
Step 5: Results            → Recommended parameters with explanations
```

### Module Structure

```
SegmentEditorAdaptiveBrushLib/
├── WizardDataStructures.py    # Dataclasses for samples and results
├── WizardAnalyzer.py          # Intensity and shape analysis
├── ParameterRecommender.py    # Algorithm selection logic
├── WizardSampler.py           # Interactive sampling handler
├── WizardUI.py                # Qt wizard dialog
└── ParameterWizard.py         # Main coordinator
```

### Data Structures

```python
@dataclass
class WizardSamples:
    foreground_points: list[tuple[int, int, int]]
    foreground_intensities: np.ndarray
    background_points: list[tuple[int, int, int]]
    background_intensities: np.ndarray
    boundary_points: list[tuple[int, int, int]]
    volume_node: Any

@dataclass
class IntensityAnalysisResult:
    foreground_mean: float
    foreground_std: float
    background_mean: float
    background_std: float
    separation_score: float  # 0-1, how well-separated
    overlap_percentage: float
    suggested_threshold_lower: float
    suggested_threshold_upper: float

@dataclass
class ShapeAnalysisResult:
    estimated_diameter_mm: float
    circularity: float
    boundary_roughness: float
    suggested_brush_radius_mm: float
    is_3d_structure: bool

@dataclass
class WizardRecommendation:
    algorithm: str
    algorithm_reason: str
    brush_radius_mm: float
    edge_sensitivity: int
    confidence: float
    warnings: list[str]
    alternative_algorithms: list[tuple[str, str]]
```

### Algorithm Recommendation Logic

The recommender uses a decision tree based on:

1. **Intensity Separation**
   - High separation (>0.8): Prefer Connected Threshold, Region Growing
   - Low separation (<0.4): Prefer Watershed, Level Set

2. **Boundary Characteristics**
   - Smooth boundary: Connected Threshold works well
   - Rough boundary: Level Set handles complexity better

3. **Structure Size**
   - Small (<10mm): Higher precision algorithms, smaller brush
   - Large (>50mm): Faster algorithms, may need multiple strokes

4. **Modality Adjustments**
   - CT: Threshold-based approaches work well
   - MRI T1: Watershed preferred for tissue boundaries
   - Ultrasound: Account for noise, prefer region growing

5. **Structure Type Adjustments**
   - Tumor: Level Set for irregular boundaries
   - Vessel: Geodesic Distance for tubular structures
   - Bone: Connected Threshold for high contrast

### Edge Sensitivity Calculation

```python
def calculate_edge_sensitivity(shape, intensity):
    sensitivity = 50  # Base
    sensitivity += int(shape.boundary_roughness * 30)
    if intensity.separation_score < 0.5:
        sensitivity += 15
    if shape.is_small_structure:
        sensitivity += 10
    return clamp(sensitivity, 0, 100)
```

### User Interface

```
┌─────────────────────────────────────────────────────────────────┐
│  Quick Select Parameters Wizard                                 │
├─────────────────────────────────────────────────────────────────┤
│  Step 1 of 5: Sample Foreground                                │
│  ─────────────────────────────────────────────────────────────  │
│  Paint a few strokes on the INSIDE of the structure you        │
│  want to segment.                                               │
│                                                                 │
│  [Paint Mode Active - Click and drag on image]                 │
│                                                                 │
│  Samples collected: 3,247 voxels                               │
│                                                                 │
│  [Clear]                                 [◄ Back] [Next ►]     │
└─────────────────────────────────────────────────────────────────┘
```

### Integration

A "Quick Select Parameters..." button is added to the Adaptive Brush UI
after the algorithm dropdown:

```python
def setupOptionsFrame(self):
    # ... algorithm combo ...

    self.wizardButton = qt.QPushButton("Quick Select Parameters...")
    self.wizardButton.clicked.connect(self.onWizardClicked)
    brushLayout.addRow(self.wizardButton)
```

## Consequences

### Positive

- **Lower barrier to entry**: New users get optimal parameters through interaction
- **Educational**: Explanations help users understand why parameters matter
- **Modality-aware**: Recommendations adapt to imaging type
- **Confidence scoring**: Users know when results may need manual adjustment
- **Alternative suggestions**: Users can explore other valid approaches

### Negative

- **Adds complexity**: Multiple new modules to maintain
- **Requires interaction**: Users must paint samples (not instant)
- **Analysis may be slow**: Large samples take time to analyze
- **Slicer dependency**: Full functionality only in Slicer environment

### Trade-offs

| Aspect | Alternative | Chosen | Reason |
|--------|-------------|--------|--------|
| Input method | Auto-detect from image | Interactive sampling | More accurate, user controls ROI |
| Modality detection | DICOM headers | User question | Not all images have headers |
| Boundary analysis | Required | Optional | Some structures have clear intensity separation |
| Recommendations | Single best | Primary + alternatives | Users may have preferences |

## Alternatives Considered

### Automatic Detection Only

**Rejected**: Can't know user's target structure or ROI without input.

### Simple Parameter Sliders

**Rejected**: Doesn't solve the "which values are good" problem.

### Machine Learning Model

**Rejected**: Requires training data, less interpretable, harder to maintain.

### Copy from Preset

**Rejected**: Presets are generic, don't adapt to specific images.

## Implementation Notes

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `WizardDataStructures.py` | Dataclasses | ~180 |
| `WizardAnalyzer.py` | Analysis algorithms | ~330 |
| `ParameterRecommender.py` | Recommendation engine | ~500 |
| `WizardSampler.py` | Interactive sampling | ~180 |
| `WizardUI.py` | Qt wizard dialog | ~400 |
| `ParameterWizard.py` | Main coordinator | ~380 |

### Test Coverage

- `test_wizard_data_structures.py`: 24 tests
- `test_wizard_analyzer.py`: 22 tests
- `test_parameter_recommender.py`: 23 tests

### Usage

```python
# From SegmentEditorEffect.onWizardClicked():
from ParameterWizard import ParameterWizard

wizard = ParameterWizard(self)
wizard.start()

# Wizard handles:
# 1. Opening dialog
# 2. Activating samplers for each page
# 3. Running analysis when user reaches results page
# 4. Applying parameters when user clicks Finish
```

## References

- [ADR-001](ADR-001-algorithm-selection.md): Algorithm Selection
- [ADR-005](ADR-005-mouse-keyboard-controls.md): Mouse and Keyboard Controls
- [IntensityAnalyzer.py](../../SegmentEditorAdaptiveBrush/SegmentEditorAdaptiveBrushLib/IntensityAnalyzer.py): Existing intensity analysis
