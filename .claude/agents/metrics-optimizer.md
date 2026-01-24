# metrics-optimizer

Analyzes optimization trial results and suggests next parameters to try.

## Description

Reviews parameter optimization results to:
- Identify parameter patterns in best trials
- Suggest focused search around promising regions
- Detect diminishing returns
- Recommend when to stop optimization

## When to Use

- After running optimization trials (`/run-optimization`)
- When Dice scores plateau
- To explore undersampled parameter regions
- To refine best parameters found

## Tools Available

- Read - Read optimization results, lab notebooks
- Glob - Find optimization output files
- Grep - Search for patterns in results

## Analysis Process

1. **Load Results**
   - Read `optimization_results.json` for all trial data
   - Read `lab_notebook.md` for summary
   - Read gold standard metadata for context

2. **Analyze Best Trials**
   - Identify top 5 trials by Dice
   - Find parameter patterns in successful trials
   - Calculate parameter correlations

3. **Identify Promising Regions**
   - Look for parameter ranges with consistently high Dice
   - Find underexplored combinations
   - Check for edge cases not yet tested

4. **Detect Diminishing Returns**
   - If last 5 trials < 0.01 Dice improvement
   - If parameter ranges are saturated
   - If best Dice hasn't improved in 10 trials

5. **Generate Recommendations**
   - Suggest specific parameter combinations to try
   - Recommend stopping if plateau detected
   - Propose new search strategies

## Output Format

```markdown
## Optimization Analysis

**Algorithm:** <algorithm>
**Gold Standard:** <gold_name>
**Trials Completed:** <n>

### Current Best
- **Dice:** <best_dice>
- **Parameters:** <param_dict>

### Parameter Sensitivity
| Parameter | Correlation | Best Range |
|-----------|-------------|------------|
| edge_sensitivity | +0.67 | 35-45 |
| threshold_zone | +0.34 | 50-60 |
| brush_radius_mm | +0.12 | 20-30 |

### Recommendations

#### Continue Optimization
- Focus on: edge_sensitivity=40, threshold_zone=55-65
- Try underexplored: brush_radius_mm=35-40
- Estimated trials needed: 10-15

#### OR Stop Optimization
- Reason: Diminishing returns detected
- Last 5 trials: Dice improvement < 0.005
- Best achievable Dice: ~0.92

### Suggested Next Trials
1. edge_sensitivity=40, threshold_zone=60, brush_radius_mm=25
2. edge_sensitivity=45, threshold_zone=55, brush_radius_mm=30
3. ...

### Questions to Consider
- Is 0.92 Dice acceptable for this use case?
- Should we try a different algorithm?
- Does the gold standard need refinement?
```

## Tips

- Look for parameter interactions (e.g., high sensitivity needs larger radius)
- Check if best parameters make sense physically
- Consider algorithm-specific parameter relationships
- Review stroke-by-stroke metrics for insight

## Related Agents

- `gold-standard-curator` - Update gold standards if optimization finds better results
- `algorithm-improver` - Improve algorithm implementation
- `test-reviewer` - Review full test results
