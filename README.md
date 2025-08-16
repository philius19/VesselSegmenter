# VesselSegmenter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.1234%2Fexample-blue)](https://doi.org/10.1234/example)

A production-ready computational pipeline for 3D vascular network segmentation from confocal microscopy data. Designed for scientific research applications with regional adaptive processing to address intensity heterogeneity in biological imaging.

## Overview

VesselSegmenter provides accurate segmentation of vascular networks from confocal z-stack images. The pipeline addresses key challenges in biological vascular imaging: low signal-to-noise ratios, sparse vessel signals, and intensity heterogeneity across tissue regions.

### Key Capabilities

**Scientific-Grade Processing**
- Regional adaptive thresholding for heterogeneous tissue regions
- Multi-scale vessel enhancement using Frangi, Sato, and Meijering filters
- Optimized algorithms for confocal microscopy vascular data

**Production-Ready Architecture**
- Object-oriented design with comprehensive error handling
- Configuration template system for reproducible research
- Batch processing capabilities for high-throughput analysis

**MATLAB Integration**
- Direct compatibility with u-shape3D mesh generation
- Automated configuration file generation
- Seamless workflow from segmentation to 3D analysis

**Quality Control**
- Parameter preview and validation system
- Comprehensive quality metrics and recommendations
- Interactive workflows for parameter optimization

## Installation

```bash
git clone https://github.com/philius19/VesselSegmenter.git
cd VesselSegmenter
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.vessel_segmentation_pipeline import VesselSegmentationPipeline, SegmentationConfig

# Create configuration
config = SegmentationConfig()

# Initialize pipeline
pipeline = VesselSegmentationPipeline(config, verbose=True)

# Segment vessels
mask, metadata = pipeline.segment_vessels(
    input_path="path/to/confocal_stack.tif",
    output_dir="path/to/output",
    sample_type="control"
)

print(f"Vessel coverage: {metadata['vessel_coverage']:.2%}")
```

### Parameter Validation

```python
# Quick validation before full processing
preview = pipeline.preview_segmentation(
    input_path="path/to/confocal_stack.tif",
    sample_type="control"
)

print(f"Preview coverage: {preview['coverage']:.2%}")
print(f"Recommendation: {preview['recommendation']}")
```

## Scientific Methodology

### Algorithm Design

The pipeline implements a four-stage approach specifically designed for vascular network segmentation:

1. **Preprocessing**: Gamma correction (γ = 1.8), morphological background subtraction, and intensity normalization

2. **Regional Adaptive Segmentation**: Image division into anatomical regions with region-specific percentile thresholding and overlap zone processing for vessel continuity

3. **Multi-Scale Vessel Enhancement**: Frangi vesselness filtering (σ = 1.0-3.0 pixels) with adaptive thresholding and component selection

4. **Smooth Mask Generation**: Distance transform gradient creation, Gaussian smoothing for mesh optimization, and single-component enforcement

### Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `gamma` | Contrast enhancement strength | 1.8 | 1.0-3.0 |
| `frangi_scales` | Vessel detection scales (pixels) | [1.0, 1.5, 2.0, 2.5, 3.0] | 0.5-5.0 |
| `max_components` | Maximum vessel components | 6 | 1-10 |
| `sigma_smooth` | Mesh smoothing parameter | 1.5 | 1.0-3.0 |
| `distance_weight` | Gradient blending weight | 0.4 | 0.1-0.8 |

## Advanced Usage

### Configuration Templates

```python
# Save configuration for reproducibility
template_path = pipeline.save_config_template(
    name="high_sensitivity_vessels",
    output_dir="./templates",
    description="High sensitivity config for fine capillaries",
    sample_type="control"
)

# Load from template
pipeline = VesselSegmentationPipeline.from_template(template_path)
```

### Batch Processing

```python
from pathlib import Path
import glob

tiff_files = glob.glob("data/*.tif")

for file_path in tiff_files:
    sample_name = Path(file_path).stem
    sample_type = "control" if "control" in sample_name.lower() else "ko_rescue"
    
    mask, metadata = pipeline.segment_vessels(
        input_path=file_path,
        output_dir=f"output/{sample_name}",
        sample_type=sample_type,
        show_progress=False
    )
    
    print(f"{sample_name}: {metadata['vessel_coverage']:.2%} coverage")
```

### Standard Configurations

```python
from src.vessel_segmentation_pipeline import create_standard_configs

configs = create_standard_configs()

# Pre-optimized configurations for different scenarios
control_pipeline = VesselSegmentationPipeline(configs['control'])      # Single vessel network
ko_pipeline = VesselSegmentationPipeline(configs['ko_rescue'])         # Fragmented networks
detail_pipeline = VesselSegmentationPipeline(configs['high_detail'])   # Fine capillary detection
large_pipeline = VesselSegmentationPipeline(configs['large_vessels'])  # Major vessel branches
```

## MATLAB Integration

### Automated Configuration Generation

```python
# Generate MATLAB configs during processing
mask, metadata = pipeline.segment_vessels(
    input_path="sample.tif",
    output_dir="output",
    generate_configs=True  # Creates .m configuration file
)
```

### Generated MATLAB Configuration

```matlab
% Auto-generated u-shape3D configuration
p.meshMode = 'loadMask';
p.maskDir = 'path/to/output/sample_slices';
p.maskName = 'slice_';
p.smoothImageSize = 1.0;  % Pre-smoothed masks
p.scaleOtsu = 0.35;       % Optimized for vessel masks
p.smoothMeshIterations = 10;
p.curvatureMedianFilterRadius = 2;
p.removeSmallComponents = 1;
```

### MATLAB Workflow

```matlab
% Load generated configuration
run('output/sample_config.m');

% Execute u-shape3D mesh generation
morphology3D(MD, p);
plotMeshMD(MD, 'surfaceMode', 'intensity');
```

## Quality Control

### Parameter Optimization

```python
# Test different configurations
configs_to_test = {
    'conservative': SegmentationConfig(gamma=1.5, max_components=1),
    'standard': SegmentationConfig(gamma=1.8, max_components=3),
    'aggressive': SegmentationConfig(gamma=2.2, max_components=6)
}

for config_name, config in configs_to_test.items():
    pipeline = VesselSegmentationPipeline(config)
    preview = pipeline.preview_segmentation("sample.tif", "control")
    
    print(f"{config_name}: {preview['coverage']:.2%} coverage, "
          f"{preview['components']} components, {preview['recommendation']}")
```

### Quality Metrics

The pipeline provides comprehensive assessment metrics:

- **Vessel Coverage**: Percentage of volume occupied by vessels
- **Component Count**: Number of disconnected vessel networks  
- **Recommendation**: Automated parameter assessment
- **Processing Metadata**: Complete parameter documentation

### Expected Results

For typical confocal vascular data:
- **Coverage**: 2-15% (depending on vessel density)
- **Components**: 1-6 (varies with network connectivity)
- **Recommendation**: "good" indicates suitable parameters

## Examples and Documentation

### Available Examples

The `examples/` directory contains seven comprehensive tutorials:

1. **Basic Usage**: Simple workflow demonstration
2. **Template Management**: Configuration saving and loading
3. **Parameter Preview**: Quick parameter validation
4. **Batch Processing**: Multiple file processing
5. **Interactive Workflow**: User-guided processing
6. **Scientific Workflow**: Publication-ready processing
7. **Advanced Customization**: Custom parameter sets

### Running Examples

```python
# Run individual examples
from examples.vessel_segmentation_examples import basic_usage_example
basic_usage_example()

# Run complete tutorial
from examples.vessel_segmentation_examples import main
main()  # Interactive tutorial covering all features
```

## Testing and Validation

### Test Suite

```bash
# Run comprehensive test suite
python tests/test_vessel_segmentation.py

# Test examples functionality
python tests/test_examples.py

# Validate installation
python scripts/validate_installation.py
```

### Test Coverage

- **18 Unit Tests**: Core functionality validation
- **7 Example Tests**: Tutorial and workflow verification
- **MATLAB Integration**: Configuration generation testing  
- **Real Data Validation**: Confocal microscopy data testing

## Publication and Citation

### Methods Section

For scientific publications, see `docs/methods_section.md` for a ready-to-use methods description optimized for journals like Nature Communications.

### Citation

If you use VesselSegmenter in your research, please cite:

```bibtex
@software{kaintoch2025vesselsegmenter,
  title={VesselSegmenter: 3D Vascular Network Segmentation Pipeline},
  author={Kaintoch, Philipp},
  year={2025},
  url={https://github.com/philius19/VesselSegmenter},
  version={1.0.0}
}
```

## Requirements

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: Variable (depends on image sizes)

### Dependencies

**Required**
- `numpy` ≥ 1.21.0
- `scipy` ≥ 1.7.0
- `scikit-image` ≥ 0.18.0
- `tifffile` ≥ 2021.7.2

**Optional**
- `matplotlib` ≥ 3.3.0 (visualization examples)
- `jupyter` ≥ 1.0.0 (notebook examples)

**MATLAB Integration**
- MATLAB R2019b or later
- u-shape3D toolbox for mesh generation
- Image Processing Toolbox

## Troubleshooting

### Common Issues

**Memory Errors**
```python
# Reduce chunk size for large images
config = SegmentationConfig(chunk_size=5)  # Default: 15
```

**Low Vessel Detection**
```python
# Lower thresholds for dim vessels
config = SegmentationConfig(
    threshold_percentiles={'control': {'top': 1.0, 'middle': 0.8, 'bottom': 1.0}}
)
```

**High Background Noise**
```python
# Increase minimum object size
config = SegmentationConfig(min_object_size=100)  # Default: 50
```

### Getting Help

1. Review the `examples/` directory for usage patterns
2. Run parameter preview to validate settings
3. Examine the test suite for implementation examples
4. Open an issue on GitHub for specific problems

## Contributing

We welcome contributions to VesselSegmenter. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- u-shape3D development team for mesh generation framework
- scikit-image community for image processing algorithms
- Contributors to Frangi, Sato, and Meijering vessel enhancement methods

## Contact

**Philipp Kaintoch**  
GitHub: [@philius19](https://github.com/philius19)
E-Mail: p.kaintoch@uni-muenster.de

For questions, suggestions, or collaborations, please open an issue or contact the maintainer.

---

*Developed for scientific research applications. Tested with confocal microscopy vascular imaging data.*
