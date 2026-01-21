# SlicerAdaptiveBrush Development Roadmap

## Version 0.1.0 - Foundation (Current)

- [x] Extension scaffold from Slicer Extension Wizard
- [x] Project documentation (CLAUDE.md, ADRs, etc.)
- [ ] Test infrastructure setup
- [ ] Convert generic template to Segment Editor Effect
- [ ] Basic mouse click handling
- [ ] Simple connected threshold (proof of concept)

## Version 0.2.0 - Core Algorithm

- [ ] ROI extraction around cursor
- [ ] Intensity analysis with GMM
- [ ] Automatic threshold estimation
- [ ] Connected threshold segmentation
- [ ] Basic UI controls (radius slider)

## Version 0.3.0 - Watershed Refinement

- [ ] Gradient magnitude computation
- [ ] Marker-based watershed
- [ ] Boundary refinement
- [ ] Edge sensitivity parameter

## Version 0.4.0 - 3D Support

- [ ] 3D sphere brush mode
- [ ] Volumetric watershed
- [ ] Memory optimization for 3D
- [ ] Slice-by-slice preview

## Version 0.5.0 - Performance

- [ ] Drag operation caching
- [ ] Progressive resolution refinement
- [ ] Background computation thread
- [ ] Performance benchmarks

## Version 0.6.0 - Edge Enhancement

- [ ] Geodesic active contour (optional)
- [ ] Refinement level selector (Fast/Balanced/Precise)
- [ ] Sub-voxel accuracy mode

## Version 1.0.0 - Production Ready

- [ ] Complete UI polish
- [ ] Comprehensive documentation
- [ ] User guide with examples
- [ ] Extension submission to Slicer Extensions Index

## Future (v2.0+)

- [ ] GPU acceleration (VTK-OpenGL)
- [ ] Deep learning edge detection
- [ ] Multi-modality support
- [ ] C++ optimization of critical paths
- [ ] Adaptive radius based on local features
