# ADR-011: Smart Optimization Framework

## Status

**Implemented** (2026-01-26)

**Components:**
- `scripts/run_optimization.py` - Entry point script for running optimization in Slicer
- `OptunaOptimizer` class with TPE sampler and HyperbandPruner
- `OptimizationConfig` for YAML configuration loading
- FAnova parameter importance analysis
- SQLite persistence for study resumption
- Example configuration files (`quick_test.yaml`, `default.yaml`, `tumor_optimization.yaml`)
- Lab notebook generation with parameter importance and trial history
- `/run-optimization` skill documentation

**Usage:**
```bash
Slicer --python-script scripts/run_optimization.py configs/quick_test.yaml
```

## Context

The project needs a systematic way to:

1. Find optimal parameters for each segmentation algorithm
2. Use smart sampling (not random/grid) to explore parameter space efficiently
3. Support early stopping of poor trials to save time
4. Analyze parameter importance to understand which settings matter most
5. Store results persistently for resumption and analysis

### Challenges

- Parameter spaces are large (5+ dimensions per algorithm)
- Trials are expensive (each requires running segmentation in Slicer)
- Some parameter combinations fail early - detecting this saves time
- Different algorithms have different parameter sets (hierarchical)
- Users want to understand *why* certain parameters work better

### Previous Approach

A basic random/grid search approach was considered but lacks:

- Smart sampling that learns from previous trials
- Early stopping/pruning of bad trials
- Hierarchical parameter suggestion (algorithm-specific params)
- Parameter importance analysis
- Persistent storage for study resumption

## Decision

Integrate **Optuna** as the optimization backend, providing:

### Why Optuna?

| Library | Hierarchical Params | Pruning | Visualization | Importance | Verdict |
|---------|---------------------|---------|---------------|------------|---------|
| **Optuna** | Excellent (Python if/else) | Excellent (Hyperband) | Excellent (Dashboard) | FAnova | **Recommended** |
| Hyperopt | Good | Limited | Poor | None | Deprecating |
| Ray Tune | Good | Excellent | Good | Via callbacks | Overkill |
| SMAC3 | Excellent | Good | Minimal | Built-in | Good alternative |
| Ax/BoTorch | Excellent | Good | Limited | Limited | Research-focused |

Optuna wins for: easiest API, best visualization, FAnova importance, HyperbandPruner efficiency.

### Core Components

```
SegmentEditorAdaptiveBrushTester/
├── SegmentEditorAdaptiveBrushTesterLib/
│   ├── OptunaOptimizer.py       # Optuna integration
│   └── OptimizationConfig.py    # YAML config loader
├── configs/                      # Optimization configs
│   ├── default.yaml
│   ├── tumor_optimization.yaml
│   └── quick_test.yaml
└── optimization_results/         # Results storage
    └── {timestamp}_{name}/
        ├── config.yaml           # Copy of input config
        ├── optuna_study.db       # SQLite for resumption
        ├── results.json          # Complete results
        └── parameter_importance.json
```

### OptunaOptimizer Class

```python
class OptunaOptimizer:
    """Optuna-powered parameter optimization."""

    def __init__(self, config_path: Path):
        self.config = OptimizationConfig.load(config_path)
        self.study = None

    def create_study(self) -> optuna.Study:
        """Create study with TPE sampler and HyperbandPruner."""
        return optuna.create_study(
            storage=f"sqlite:///{self.results_path}/optuna_study.db",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner(),
            direction="maximize",
            load_if_exists=True
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Optimization objective with pruning support."""
        # Suggest algorithm
        algo = trial.suggest_categorical("algorithm", self.config.algorithms)

        # Suggest algorithm-specific params (hierarchical)
        params = self._suggest_params(trial, algo)

        # Run segmentation via recipe
        for i, stroke_dice in enumerate(metrics["stroke_dices"]):
            trial.report(stroke_dice, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return metrics["dice"]

    def get_importance(self) -> dict:
        """FAnova parameter importance analysis."""
        return optuna.importance.get_param_importances(
            self.study,
            evaluator=optuna.importance.FanovaImportanceEvaluator()
        )
```

### YAML Configuration

```yaml
# configs/tumor_optimization.yaml
version: "1.0"
name: "MRBrainTumor1 Optimization"
description: "Find optimal parameters for brain tumor segmentation"

settings:
  n_trials: 100
  timeout_minutes: 60
  pruning: true
  pruner: "hyperband"      # hyperband, median, none
  sampler: "tpe"           # tpe, random, cmaes
  primary_metric: "dice"   # dice, hausdorff_95
  save_segmentations: true
  save_screenshots: true

# Recipes to optimize (Python files)
recipes:
  - path: "recipes/brain_tumor_1.py"
    gold_standard: "gold_standards/MRBrainTumor1_tumor.seg.nrrd"

# Parameter search space
parameter_space:
  global:
    edge_sensitivity: {type: int, range: [20, 80], step: 10}
    threshold_zone: {type: int, range: [30, 70], step: 10}

  algorithm_substitution:
    enabled: true
    candidates: ["watershed", "level_set_cpu", "connected_threshold"]

  algorithms:
    watershed:
      watershedGradientScale: {type: float, range: [0.5, 2.5]}
      watershedSmoothing: {type: float, range: [0.2, 1.0]}

    level_set_cpu:
      levelSetIterations: {type: int, range: [30, 150], step: 20}
      levelSetPropagation: {type: float, range: [0.5, 2.0]}
```

### Key Features

1. **TPE Sampler**: Tree-structured Parzen Estimator learns from past trials to suggest promising parameters.

2. **HyperbandPruner**: Prunes trials early if intermediate metrics (per-stroke Dice) are poor. Estimated 4x speedup over full evaluation.

3. **Hierarchical Parameters**: Algorithm-specific parameters only suggested when that algorithm is selected:
   ```python
   algo = trial.suggest_categorical("algorithm", ["watershed", "level_set_cpu"])
   if algo == "watershed":
       grad_scale = trial.suggest_float("watershed_gradient_scale", 0.5, 2.5)
   elif algo == "level_set_cpu":
       iterations = trial.suggest_int("level_set_iterations", 30, 150, step=20)
   ```

4. **FAnova Importance**: Identifies which parameters most affect the objective:
   ```json
   {
     "edge_sensitivity": 0.45,
     "algorithm": 0.30,
     "watershedGradientScale": 0.15,
     "brush_radius_mm": 0.10
   }
   ```

5. **SQLite Persistence**: Studies resume from checkpoint after crashes:
   ```python
   study = optuna.create_study(
       storage="sqlite:///results/study.db",
       load_if_exists=True
   )
   ```

### Optimization Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `global_override` | Same parameter values for all recipe actions | Quick optimization |
| `algorithm_swap` | Try different algorithms keeping click positions | Algorithm comparison |
| `per_action` | Vary parameters per action | Maximum flexibility |
| `action_subset` | Find minimal actions needed | Click efficiency |

### Integration with Recipes

The optimizer works with **segmentation recipes** (see ADR-013):

```python
# Load recipe
recipe = Recipe.load("recipes/brain_tumor_1.py")

# Apply parameter overrides from Optuna
params = optimizer.suggest_params(trial)
modified_recipe = recipe.with_overrides(params)

# Execute and measure
result = RecipeRunner(modified_recipe).run()
dice = SegmentationMetrics.compute(result, gold).dice
```

## Consequences

### Positive

- **Faster convergence**: TPE finds good parameters in fewer trials than random search
- **Time savings**: HyperbandPruner stops bad trials early (~4x speedup)
- **Insights**: FAnova explains which parameters matter
- **Resumable**: SQLite storage allows long studies across sessions
- **Configurable**: YAML files enable reproducible optimization runs

### Negative

- **New dependency**: Optuna must be installed (`pip install optuna`)
- **Complexity**: More moving parts than simple grid search
- **Learning curve**: TPE/Hyperband concepts may be unfamiliar

### Trade-offs

| Aspect | Alternative | Chosen | Reason |
|--------|-------------|--------|--------|
| Backend | Grid search | Optuna TPE | Much faster convergence |
| Storage | JSON files | SQLite | Native resumption support |
| Config | Python code | YAML files | Easier editing, shareable |

## Alternatives Considered

### Pure Grid Search

**Rejected**: Combinatorial explosion. 5 parameters with 5 values each = 3,125 trials.

### Hyperopt

**Rejected**: Less active development, worse visualization, no HyperbandPruner.

### Custom Bayesian Optimization

**Rejected**: Reinventing the wheel. Optuna is mature and well-tested.

### No Configuration Files

**Rejected**: Hardcoded parameters make studies non-reproducible.

## Implementation Notes

### Dependencies

Add to `pyproject.toml` dev dependencies:
```toml
[project.optional-dependencies]
dev = [
    "optuna>=4.0.0",
    "pyyaml>=6.0",
]
```

### Visualization

Optuna provides built-in visualization:
```python
import optuna.visualization as vis
fig = vis.plot_optimization_history(study)
fig = vis.plot_param_importances(study)
```

Optional: `optuna-dashboard` for web-based monitoring.

## References

- [ADR-010](ADR-010-testing-framework.md): Slicer Testing Framework
- [ADR-013](ADR-013-segmentation-recipes.md): Segmentation Recipes
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [TPE Paper](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
- [Hyperband Paper](https://arxiv.org/abs/1603.06560)
