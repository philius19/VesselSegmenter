# Methods Section for Nature Communications

## 3D Vascular Network Segmentation and Mesh Generation

Confocal z-stack images were segmented using a custom regional adaptive pipeline to address intensity heterogeneity between anatomical regions. Raw images underwent gamma correction (γ=1.8) and morphological background subtraction, followed by region-specific intensity thresholding (top: 5th percentile, middle: 3rd percentile, bottom: 1.5th percentile for damaged tissue; 2nd percentile uniformly for controls). Vessel enhancement was performed using multi-scale Frangi filtering (σ = 1.0-3.0 pixels, γ=15) to detect tubular structures across different vessel diameters. Binary masks were refined through connected component analysis, retaining the largest 1-6 components based on network complexity. For 3D mesh generation, smooth grayscale masks were created using distance transform gradients (weight=0.4) combined with Gaussian smoothing (σ=1.5) to ensure high-quality isosurface extraction. Final 3D meshes were generated using the u-shape3D algorithm with optimized parameters for vascular data (scaleOtsu=0.35, smoothMeshIterations=10). The complete pipeline preserved vessel connectivity while maintaining single-component topology required for volumetric analysis.

---

## Key Technical Points Covered (for reference):

1. **Regional adaptive thresholding** - addresses the core innovation
2. **Multi-scale Frangi filtering** - standard vessel detection method  
3. **Component selection and refinement** - ensures quality
4. **Distance transform smoothing** - critical for mesh quality
5. **u-shape3D integration** - final 3D mesh generation
6. **Parameter specifications** - reproducibility requirements

**Word count**: ~165 words (well under typical 200-word limit for methods subsections)
**Line count**: 7 lines (within the <8 line requirement)