# ADR-014: Enhanced Testing and Recipe Replay

## Status

Proposed

## Context

The existing recipe system (ADR-013) uses Python functions for segmentation workflows.
While powerful, this approach has limitations:
- Cannot step through execution (all-or-nothing)
- Cannot rewind to a previous state
- Cannot branch and record new actions mid-workflow
- No checkpoint/restore capability

Users need to:
1. Review segmentation steps one at a time
2. Understand parameters used at each step
3. Rewind and try different approaches
4. Branch from a known-good point and experiment

Additionally, the testing framework captures layout screenshots but not dedicated 3D view
screenshots, and the Review Module lacks view layout controls.

## Decision

### 1. Enhanced Screenshots

Add `capture_layout_with_3d()` method to ScreenshotCapture that captures both the layout
and a dedicated 3D view screenshot as separate files. This ensures 3D visualization is
always captured for review.

### 2. View Layout Controls

Add layout switching buttons to the Review Module:
- Conventional (3) - Default Slicer layout
- Four-Up (4) - Four-up grid view
- 3D Only (6) - Single 3D view
- Dual 3D (15) - Two 3D views

### 3. Action-Based Recipe Format

Replace function-based recipes with JSON action lists:

```python
@dataclass
class RecipeAction:
    type: str  # "paint", "erase", "set_param", "set_algorithm"
    ras: tuple[float, float, float] | None = None
    params: dict = field(default_factory=dict)
    timestamp: float = 0.0
    description: str = ""

@dataclass
class ActionRecipe:
    name: str
    sample_data: str
    segment_name: str
    actions: list[RecipeAction]
    gold_standard: str | None = None
```

Benefits:
- Each action is a discrete, serializable step
- Supports stepping, rewinding, branching
- Convertible from existing Python recipes via execution recording

### 4. Segmentation Checkpointing

Save labelmap as numpy array at each step:
- Fast creation and restore (<50ms for typical segmentations)
- Enables instant rewind without re-execution
- Memory-efficient using numpy compression

```python
@dataclass
class SegmentationCheckpoint:
    step_index: int
    labelmap_array: np.ndarray
    timestamp: float
```

### 5. Stepping Recipe Runner

New `SteppingRecipeRunner` class that:
- Executes recipes one action at a time
- Creates checkpoints before each action
- Supports step_forward(), step_backward(), goto_step()
- Enables branching via start_branch(), add_manual_action(), save_branch()

### 6. Branch Workflow

Copy + Continue approach:
- Never modify original recipe during replay
- Create copy with actions up to current step
- Record new manual actions to copy
- Save branched recipe as new file

This enables safe experimentation without risk to the original workflow.

### 7. Recipe Replay UI

Add new collapsible section to Review Module:

```
+- Recipe Replay -------------------------------------------+
| Recipe: [brain_tumor_1 v] [Load] [Convert from .py]      |
|                                                           |
| Step: [|<] [<] [3/5] [>] [>|]   [Auto-play >] Speed: [1x]|
|                                                           |
|  *-----*-----@-----o-----o   (timeline slider)            |
|  1     2     3     4     5                                |
|                                                           |
| +- Current Action ------------------------------------+ |
| | Step 3: paint                                       | |
| | Position: (-5.31, 20.70, 22.17) RAS                | |
| | Algorithm: watershed                                | |
| | Radius: 20.0 mm                                    | |
| | Edge Sensitivity: 45                               | |
| +-----------------------------------------------------+ |
|                                                           |
| [Start Branch] [Stop Branch] [Save Branch As...]          |
+-----------------------------------------------------------+
```

### 8. Action Recorder Callback Mode

Extend ActionRecorder with optional callback for branch recording:

```python
def __init__(self, test_run_folder=None, action_callback=None):
    self._action_callback = action_callback

def _record_action(self, action_type, details):
    # ... existing code ...
    if self._action_callback:
        self._action_callback(action)
```

## Consequences

### Positive

- **Step-by-step debugging**: Review each action's effect on segmentation
- **Safe experimentation**: Branch from any point without risk
- **Visual parameter inspection**: See parameters used at each step
- **Instant rewind**: Checkpoints enable fast state restoration
- **Better screenshots**: 3D view always captured alongside layout
- **Flexible viewing**: Layout buttons for different review needs

### Negative

- **Two recipe formats**: Python functions (ADR-013) and JSON actions (ADR-014)
- **Checkpoint memory**: Long workflows accumulate checkpoint data
- **Added complexity**: Recorder and runner have more responsibilities

### Mitigations

- Python recipes remain canonical; JSON is execution trace
- Checkpoints stored as compressed numpy arrays
- Clear separation between recording and replay code paths

## Implementation

### New Files

| File | Purpose |
|------|---------|
| `ActionRecipe.py` | RecipeAction, ActionRecipe dataclasses |
| `SteppingRecipeRunner.py` | Step-by-step execution with checkpoints |

### Modified Files

| File | Changes |
|------|---------|
| `ScreenshotCapture.py` | Add `capture_layout_with_3d()` |
| `SegmentEditorAdaptiveBrushReviewer.py` | Layout buttons, replay section |
| `ActionRecorder.py` | Add callback mode |
| `ROADMAP.md` | Add v0.14.0 section |

## References

- [ADR-010](ADR-010-testing-framework.md): Slicer Testing Framework
- [ADR-012](ADR-012-review-module.md): Results Review Module
- [ADR-013](ADR-013-segmentation-recipes.md): Segmentation Recipes
