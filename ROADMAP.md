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

## Version 0.10.0 - GUI Polish (Current)

- [ ] Collapsible help section (like Paint effect's "Show details")
- [ ] Reorder widgets: most impactful parameters first, progressive disclosure
  - [ ] Basic controls visible by default (algorithm, radius, sensitivity)
  - [ ] Advanced parameters in collapsed sections
  - [ ] Consistent parameter positions across algorithms (muscle memory)
- [ ] Algorithm-specific parameter visibility (hide irrelevant params)
- [ ] Add link to documentation

## Version 0.11.0 - CI/CD Pipeline

- [ ] GitHub Actions: run pytest on all commits
- [ ] CI status badge in README
- [ ] Extension packaging workflow

## Version 0.12.0 - Slicer Integration Testing

- [ ] GitHub Action: install extension into Slicer
- [ ] Run effect on Slicer sample data (MRHead, CTChest, etc.)
- [ ] Verify extension installs and all Slicer tests pass
- [ ] Record-and-replay: convert recorded Slicer sessions to test scripts

## Version 0.13.0 - Living Documentation

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
