# Algorithm Profiles

This document describes the algorithm profiling system used to optimize segmentation parameters.

## Overview

Algorithm profiles provide data-driven recommendations for algorithm and parameter selection
based on:

1. **Automated optimization** - Optuna-based parameter search against gold standards
2. **Modality characteristics** - Known imaging physics and typical intensity patterns
3. **Structure properties** - Shape, size, and boundary characteristics

## Profile Data Structure

Algorithm profiles are stored in `algorithm_profiles.json` at the project root.

### Parameter Importance

From optimization studies, we found:

| Parameter | Importance |
|-----------|------------|
| Algorithm choice | 73.1% |
| Brush radius | 18.6% |
| Threshold zone | 5.1% |
| Edge sensitivity | 3.2% |

**Key insight**: Algorithm selection is by far the most important parameter.

### Algorithm Characteristics

Each algorithm has profiled characteristics:

```json
{
  "watershed": {
    "speed": "medium",
    "precision": "high",
    "boundary_adherence": "excellent",
    "recommended_for": ["tumor", "lesion", "brain_tissue"],
    "parameters": {
      "gradient_scale": {"range": [0.5, 3.0], "optimal": 1.1},
      "smoothing": {"range": [0.1, 1.0], "optimal": 0.78}
    }
  }
}
```

### Modality Recommendations

Recommendations are organized by imaging modality and structure:

```json
{
  "ct": {
    "bone": {
      "primary": "threshold_brush",
      "alternatives": ["connected_threshold"],
      "preset": "ct_bone"
    }
  }
}
```

## Using Profiles Programmatically

### Loading Profiles

```python
import json
from pathlib import Path

profiles_path = Path(__file__).parent / "algorithm_profiles.json"
with open(profiles_path) as f:
    profiles = json.load(f)
```

### Getting Recommendations

```python
def get_recommendation(modality: str, structure: str) -> dict:
    """Get algorithm recommendation for modality/structure pair."""
    mod_recs = profiles.get("modality_recommendations", {})
    if modality in mod_recs and structure in mod_recs[modality]:
        return mod_recs[modality][structure]
    return profiles.get("structure_recommendations", {}).get(structure, {})

# Example usage
rec = get_recommendation("ct", "bone")
print(f"Primary algorithm: {rec['primary']}")  # threshold_brush
print(f"Preset to apply: {rec['preset']}")      # ct_bone
```

### Applying Recommendations

```python
def apply_recommendation(effect, modality: str, structure: str):
    """Apply algorithm recommendation to effect."""
    rec = get_recommendation(modality, structure)

    # Apply the preset (sets common parameters)
    if "preset" in rec:
        effect.applyPreset(rec["preset"])

    # Algorithm is selected via UI - log the recommendation
    print(f"Recommended algorithm: {rec.get('primary', 'watershed')}")
```

## Generating New Profiles

### Running Optimization

Use the optimization script to generate benchmark data:

```bash
Slicer --python-script scripts/run_optimization.py
```

This runs Optuna optimization against gold standards and outputs:
- `results.json` - Full trial data
- `parameter_importance.json` - Relative parameter importance
- Screenshots and segmentations for each trial

### Adding Benchmarks

After running optimization, add benchmark results to the profile:

```json
{
  "algorithms": {
    "watershed": {
      "benchmarks": {
        "MRBrainTumor1_tumor": {
          "dice": 0.9991,
          "edge_sensitivity": 70,
          "threshold_zone": 60,
          "brush_radius_mm": 25.0
        }
      }
    }
  }
}
```

## Profile Evolution

### Adding New Modalities

1. Create recipes for the new modality
2. Run optimization with gold standards
3. Analyze results and add to `modality_recommendations`
4. Update documentation

### Refining Recommendations

1. Collect user feedback on segmentation quality
2. Run A/B tests with alternative algorithms
3. Update primary/alternative recommendations based on results

## Integration with Presets

Presets define **common parameters** (edge sensitivity, sampling method, etc.)
that work well for a modality/structure combination.

Algorithm selection is **independent** of presets - users choose the algorithm
via the UI dropdown based on:

1. **Speed requirements** - Fast vs. precise
2. **Boundary characteristics** - Sharp vs. diffuse
3. **Algorithm profiles** - Data-driven recommendations

### Preset to Algorithm Mapping

While presets don't force an algorithm, there are natural pairings:

| Preset | Recommended Algorithm |
|--------|----------------------|
| `ct_bone` | Threshold Brush |
| `ct_soft_tissue` | Watershed |
| `ct_lung` | Connected Threshold |
| `ct_vessel_contrast` | Geodesic Distance |
| `mri_t1gd_tumor` | Watershed |
| `mri_t2_lesion` | Watershed |

## Future Work

1. **Auto-selection** - Automatically suggest algorithm based on modality detection
2. **Online learning** - Update profiles based on user corrections
3. **Per-structure profiles** - Build profiles for specific anatomical structures
4. **Multi-algorithm fusion** - Combine results from multiple algorithms
