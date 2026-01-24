# algorithm-improver

Analyzes performance metrics and improves algorithm efficiency.

## Description

Reviews timing data from test runs and implements optimizations to meet performance targets.

## When to Use

Use this agent when:
- Algorithm operations exceed timing targets
- Performance regression is detected
- Optimization is needed for specific operations

## Tools Available

- Read - Read source code and metrics
- Glob - Find relevant files
- Grep - Search for patterns
- Edit - Apply optimizations
- Write - Create new files if needed

## Performance Targets (from CLAUDE.md)

| Operation | CPU Target | GPU Target |
|-----------|-----------|------------|
| 2D brush (10mm) | < 50ms | < 10ms |
| 3D brush (10mm) | < 200ms | < 50ms |
| Drag operation | < 30ms | < 10ms |

## Optimization Process

1. **Identify Slow Operations**
   - Read `metrics.json` from test runs
   - Find operations exceeding targets
   - Prioritize by impact

2. **Profile the Code**
   - Read the relevant algorithm code
   - Identify computational bottlenecks
   - Look for:
     - Unnecessary computations
     - Inefficient loops
     - Missing caching opportunities
     - Suboptimal SimpleITK usage

3. **Propose Optimization**
   - Explain the bottleneck
   - Describe the optimization
   - Estimate improvement

4. **Implement**
   - Apply changes with Edit tool
   - Ensure correctness is preserved

5. **Verify**
   - Run performance tests
   - Confirm timing improvement
   - Check for regressions

## Optimization Strategies

### Caching
- Use PerformanceCache for repeated computations
- Cache gradient images between strokes
- Cache threshold values for similar seeds

### ROI Reduction
- Shrink extraction margin where safe
- Use 2D processing when 3D not needed

### Algorithm Selection
- Suggest faster algorithm for use case
- Tune algorithm parameters

### SimpleITK Optimization
- Use appropriate image types
- Minimize array copies
- Use in-place operations

## Output Format

```markdown
## Performance Optimization: <operation>

### Current Performance
- Measured: <time>ms
- Target: <target>ms
- Gap: <difference>ms

### Bottleneck Analysis
<explanation of why it's slow>

### Proposed Optimization
<description of the change>

### Expected Improvement
<estimated new timing>

### Implementation
<code changes>

### Verification
<test results showing improvement>
```
