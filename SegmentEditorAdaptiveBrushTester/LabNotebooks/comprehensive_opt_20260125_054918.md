# Comprehensive Algorithm Optimization

**Date:** 2026-01-25
**Time:** 05:49:18

## Overview


Comprehensive parameter optimization across all algorithms.

- **Total trials:** 146
- **Algorithms tested:** 7
- **Click configurations:** 5
- **Radius configurations:** 6
- **Bugs found:** 0


## Best Results per Algorithm



### watershed


- **Dice:** 1.000
- **Hausdorff 95%:** 0.0mm
- **Click config:** 5_standard
- **Parameters:** `{'brush_radius_mm': 25.0, 'thresholdZone': 50, 'edge_sensitivity': 40, 'watershedGradientScale': 1.5, 'watershedSmoothing': 0.5}`


### geodesic_distance


- **Dice:** 0.947
- **Hausdorff 95%:** 1.3mm
- **Click config:** 5_dense_center
- **Parameters:** `{'brush_radius_mm': 25.0, 'thresholdZone': 50, 'edge_sensitivity': 60, 'geodesicEdgeWeight': 15.0, 'geodesicDistanceScale': 1.5, 'geodesicSmoothing': 0.8}`


### connected_threshold


- **Dice:** 0.959
- **Hausdorff 95%:** 0.9mm
- **Click config:** 1_center
- **Parameters:** `{'brush_radius_mm': 25.0, 'thresholdZone': 50, 'edge_sensitivity': 40}`


### region_growing


- **Dice:** 0.950
- **Hausdorff 95%:** 0.9mm
- **Click config:** 3_spread
- **Parameters:** `{'brush_radius_mm': 25.0, 'thresholdZone': 50, 'edge_sensitivity': 60, 'regionGrowingMultiplier': 3.5, 'regionGrowingIterations': 6}`


### threshold_brush


- **Dice:** 0.924
- **Hausdorff 95%:** 6.6mm
- **Click config:** 1_center
- **Parameters:** `{'brush_radius_mm': 25.0, 'thresholdZone': 50, 'autoThreshold': True, 'edge_sensitivity': 40, 'thresholdMethod': 'otsu'}`


### level_set_cpu


- **Dice:** 0.930
- **Hausdorff 95%:** 1.4mm
- **Click config:** 3_spread
- **Parameters:** `{'brush_radius_mm': 25.0, 'thresholdZone': 50, 'edge_sensitivity': 50, 'levelSetPropagation': 1.2, 'levelSetCurvature': 1.2, 'levelSetIterations': 80}`


### random_walker


- **Dice:** 0.712
- **Hausdorff 95%:** 6.4mm
- **Click config:** 7_thorough
- **Parameters:** `{'brush_radius_mm': 25.0, 'thresholdZone': 50, 'edge_sensitivity': 60, 'randomWalkerBeta': 300}`


## Overall Best


- **Algorithm:** watershed
- **Dice:** 1.000
- **Hausdorff 95%:** 0.0mm
- **Trial ID:** 8
- **Click config:** 5_standard


## Files


- Results: `test_runs/comprehensive_opt_20260125_053405/summary.json`
- Analysis: `test_runs/comprehensive_opt_20260125_053405/analysis.json`
- Screenshots: `test_runs/comprehensive_opt_20260125_053405/trial_*/`
