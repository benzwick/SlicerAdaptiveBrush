# SlicerAdaptiveBrush Development Roadmap

## Version 0.1.0 - Foundation ✓

- [x] Extension scaffold from Slicer Extension Wizard
- [x] Project documentation (CLAUDE.md, ADRs, etc.)
- [x] Test infrastructure setup (pytest + unittest scaffolding)
- [x] Convert generic template to Segment Editor Effect
- [x] Basic mouse click handling
- [x] Simple connected threshold (proof of concept)

## Version 0.2.0 - Core Algorithm ✓

- [x] ROI extraction around cursor
- [x] Intensity analysis with GMM (sklearn with fallback)
- [x] Automatic threshold estimation
- [x] Connected threshold segmentation
- [x] Basic UI controls (radius slider, algorithm selector)

## Version 0.3.0 - Watershed Refinement ✓

- [x] Gradient magnitude computation
- [x] Marker-based watershed (MorphologicalWatershedFromMarkers)
- [x] Boundary refinement via erosion/dilation markers
- [x] Edge sensitivity parameter (0-100%)

## Version 0.4.0 - 3D Support ✓

- [x] 3D sphere brush mode
- [x] Volumetric watershed
- [x] Memory optimization via ROI extraction
- [ ] Slice-by-slice preview

## Version 0.5.0 - Threshold Brush Enhancement ✓

- [x] Threshold Brush algorithm
- [x] Auto-threshold methods (Otsu, Huang, Triangle, Maximum Entropy, IsoData, Li)
- [x] Auto-detection of foreground/background based on seed intensity
- [x] Manual threshold mode with lower/upper sliders
- [x] Set-from-seed button with tolerance control
- [x] Brush outline visualization (VTK pipeline)

## Version 0.6.0 - Additional Algorithms ✓

- [x] Level Set algorithm (Geodesic Active Contour via SimpleITK)
- [x] Region Growing algorithm (ConfidenceConnected)
- [x] Geodesic Distance algorithm (Fast Marching with gradient+intensity speed)
- [x] Random Walker algorithm (scikit-image with fallback)
- [x] Algorithm dropdown selector with 8 options
- [x] Backend selector UI (Auto/CPU/GPU preparation)

## Version 0.7.0 - Performance Infrastructure ✓

- [x] PerformanceCache class structure
- [x] Cache invalidation on parameter changes
- [x] Undo/redo support (single save per stroke)
- [x] Threshold caching (reuse when seed intensity similar)
- [x] Gradient caching between strokes on same slice
- [x] Cache statistics and hit rate logging
- [ ] ROI result caching for nearby brush positions
- [ ] Performance benchmarks

## Version 0.8.0 - Advanced UI ✓

- [x] Dual-circle brush visualization (outer=extent, inner=threshold zone)
- [x] Preview mode (semi-transparent segmentation preview before clicking)
- [x] Dynamic threshold ranges based on image intensity percentiles
- [x] Parameter presets for common tissue types (Bone, Soft Tissue, Lung, Brain, Tumor, Vessels, Fat)
- [x] Advanced parameters UI section (collapsible)
- [x] Gaussian distance weighting for intensity sampling
- [x] Comprehensive tooltips for all parameters
- [x] User-visible warnings for missing dependencies

## Version 0.9.0 - Usability Enhancements ✓

- [x] Erase mode (Ctrl+click or Middle+click to invert)
- [x] Scroll wheel controls
  - [x] Shift+scroll for brush radius
  - [x] Ctrl+Shift+scroll for threshold zone
- [x] Custom icons (brush with sparkles)
- [x] Configurable crosshair display (styles, size, color)
- [x] Auto-install optional Python dependencies with user prompt
- [ ] Keyboard shortcuts (`[`/`]` for radius, number keys for algorithms)
- [ ] Status bar feedback (algorithm name, computation time)
- [ ] Save/load custom user presets

## Version 0.10.0 - GUI Polish ✓

- [x] Collapsible help section (like Paint effect's "Show details")
- [x] Reorder widgets: most impactful parameters first, progressive disclosure
  - [x] Basic controls visible by default (algorithm, radius, sensitivity)
  - [x] Advanced parameters in collapsed sections
  - [x] Consistent parameter positions across algorithms (muscle memory)
- [x] Algorithm-specific parameter visibility (hide irrelevant params)
- [ ] ~~Add link to documentation~~ (deferred - no docs yet)

## Version 0.11.0 - CI/CD Pipeline ✓

- [x] GitHub Actions: run pytest on all commits
- [x] CI status badge in README
- [x] Extension packaging workflow

## Version 0.12.0 - Slicer Testing Framework ✓

- [x] SegmentEditorAdaptiveBrushTester module
  - [x] TestRunner with registered test cases
  - [x] TestCase base class and TestContext utilities
  - [x] TestRegistry for test case discovery
- [x] Screenshot capture utilities (slice views, 3D, widgets)
- [x] Metrics collection (timing, quality metrics)
- [x] Interactive testing panel (manual recording, notes)
- [x] Test run output organization (results, screenshots, logs)
- [x] Claude Code skills for test execution and review
  - [x] run-slicer-tests: launch Slicer and run test suite
  - [x] review-test-results: analyze test output with agents
  - [ ] add-test-case: create new test cases from template
- [ ] Claude Code agents for improvement workflows
  - [ ] test-reviewer: analyze results and suggest improvements
  - [ ] bug-fixer: diagnose failures and propose fixes
  - [ ] algorithm-improver: optimize based on metrics
  - [ ] ui-improver: review screenshots for UI issues
- [x] Initial test cases
  - [x] test_workflow_basic.py
  - [x] test_algorithm_watershed.py
  - [x] test_ui_options_panel.py
  - [x] test_optimization_tumor.py
  - [x] test_regression_gold.py

## Version 0.13.0 - Smart Optimization Framework

- [x] Segmentation Recipes (ADR-013)
  - [x] Recipe class for complete segmentation workflows
  - [x] Action class for individual operations (adaptive_brush, paint, threshold, etc.)
  - [x] RecipeRunner for executing recipes in Slicer
  - [x] RecipeRecorder for capturing manual sessions
  - [x] Example recipes (brain_tumor_1.py, template.py)

- [x] Optuna Integration (ADR-011)
  - [x] OptunaOptimizer with TPE sampler
  - [x] HyperbandPruner for early stopping (~4x speedup)
  - [x] Hierarchical parameter suggestion (algorithm-specific)
  - [x] FAnova parameter importance analysis
  - [x] SQLite persistence for study resumption

- [x] YAML Configuration
  - [x] OptimizationConfig for reproducible runs
  - [x] Parameter space definition
  - [x] Algorithm substitution support
  - [x] Example configs (default.yaml, tumor_optimization.yaml, quick_test.yaml)

- [x] Algorithm Characterization
  - [x] AlgorithmProfile with performance metrics
  - [x] AlgorithmCharacterizer for profile generation
  - [x] AlgorithmReportGenerator for markdown reports
  - [x] Strengths, weaknesses, and use case recommendations

- [ ] Results Review Module (ADR-012)
  - [ ] SegmentEditorAdaptiveBrushReviewer Slicer module
  - [ ] Dual segmentation display (gold vs test)
  - [ ] Screenshot thumbnail viewer
  - [ ] Save-as-gold-standard functionality
  - [ ] Visual comparison modes (outline, transparent, fill)

- [ ] Documentation
  - [x] ADR-011: Optimization Framework
  - [x] ADR-012: Results Review Module
  - [x] ADR-013: Segmentation Recipes
  - [ ] User guide for recipe creation
  - [ ] Optimization tutorial

## Version 0.15.0 - Living Documentation

- [ ] Test scripts generate screenshots automatically
- [ ] Documentation written to match test coverage
- [ ] Auto-generated website (GitHub Pages)
- [ ] Documentation updates dynamically with code changes
- [ ] User guide with auto-captured screenshots

## Version 1.0.0 - Production Ready

- [ ] All tests passing (unit + integration)
- [ ] Complete documentation
- [ ] Extension submission to Slicer Extensions Index

## Future (v2.0+)

- [ ] GPU acceleration (OpenCL/CUDA for Level Set)
- [ ] GPU-accelerated gradient computation
- [ ] Deep learning edge detection
- [ ] Multi-modality support
- [ ] C++ optimization of critical paths
- [ ] Adaptive radius based on local features
- [ ] Background computation thread
- [ ] Auto-parameter suggestion based on image characteristics
