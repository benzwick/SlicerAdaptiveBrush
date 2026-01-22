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

## Version 0.9.0 - Usability Enhancements (Current)

- [x] Erase mode (Ctrl+click or Middle+click to invert)
- [x] Scroll wheel controls
  - [x] Shift+scroll for brush radius
  - [x] Ctrl+Shift+scroll for threshold zone
- [ ] Keyboard shortcuts
  - [ ] `[` / `]` to decrease/increase brush radius
  - [ ] Number keys (1-5) to switch algorithms quickly
- [ ] Algorithm-specific parameter visibility (hide irrelevant params)
- [ ] Status bar feedback (show algorithm and computation time)
- [ ] Save/load custom user presets
- [ ] Speed up Level Set with early stopping convergence check

## Version 1.0.0 - Production Ready

- [ ] Complete UI polish
- [ ] Comprehensive documentation
- [ ] User guide with examples
- [x] Test implementations (41 tests passing)
- [ ] Auto-parameter suggestion based on image characteristics
- [ ] Extension submission to Slicer Extensions Index

## Future (v2.0+)

- [ ] GPU acceleration (OpenCL/CUDA for Level Set)
- [ ] GPU-accelerated gradient computation
- [ ] Deep learning edge detection
- [ ] Multi-modality support
- [ ] C++ optimization of critical paths
- [ ] Adaptive radius based on local features
- [ ] Background computation thread
