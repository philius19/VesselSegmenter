# Installation Guide

This guide provides detailed installation instructions for VesselSegmenter on different platforms.

## Prerequisites

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: Version 3.8 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended for large datasets
- **Storage**: At least 2GB free space for installation and temporary files

### Python Environment

We recommend using conda or virtualenv to manage dependencies:

```bash
# Using conda (recommended)
conda create -n vesselsegmenter python=3.9
conda activate vesselsegmenter

# Using virtualenv
python -m venv vesselsegmenter
source vesselsegmenter/bin/activate  # On Windows: vesselsegmenter\Scripts\activate
```

## Installation Methods

### Method 1: From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/philius19/VesselSegmenter.git
cd VesselSegmenter

# Install in development mode
pip install -e .

# Install optional dependencies
pip install -e ".[examples]"  # For visualization examples
pip install -e ".[dev]"       # For development tools
```

### Method 2: Direct Installation

```bash
# Install required dependencies
pip install numpy scipy scikit-image tifffile

# Download and extract the package
# Copy vessel_segmentation_pipeline.py to your project directory
```

## Verification

Test your installation:

```python
# Test basic imports
from src.vessel_segmentation_pipeline import SegmentationConfig, VesselSegmentationPipeline

# Create a simple pipeline
config = SegmentationConfig()
pipeline = VesselSegmentationPipeline(config)

print("✓ VesselSegmenter installed successfully!")
```

## Optional Components

### MATLAB Integration (Optional)

For 3D mesh generation capabilities:

1. **Install MATLAB** (R2019b or later)
2. **Install u-shape3D toolbox** (available separately)
3. **Add to MATLAB path**:
   ```matlab
   addpath('path/to/VesselSegmenter/matlab')
   ```

### Jupyter Notebooks (Optional)

For interactive examples:

```bash
pip install jupyter matplotlib
jupyter notebook examples/
```

## Troubleshooting

### Common Installation Issues

**ImportError: No module named 'skimage'**
```bash
pip install scikit-image
```

**Memory errors during installation**
```bash
pip install --no-cache-dir -e .
```

**Permission errors (macOS/Linux)**
```bash
pip install --user -e .
```

### Platform-Specific Notes

**Windows**
- Use Anaconda for easier scientific package management
- Ensure Visual C++ Build Tools are installed

**macOS**
- Xcode command line tools may be required: `xcode-select --install`

**Linux**
- Ensure development packages are installed: `sudo apt-get install python3-dev`

## Performance Optimization

### For Large Datasets

1. **Increase available memory**:
   ```python
   config = SegmentationConfig(chunk_size=5)  # Reduce from default 15
   ```

2. **Use SSD storage** for temporary files

3. **Close other applications** during processing

### For Batch Processing

1. **Use silent mode**:
   ```python
   pipeline = VesselSegmentationPipeline(config, verbose=False)
   ```

2. **Process overnight** for large datasets

3. **Monitor disk space** for output files

## Next Steps

After installation:

1. **Run examples**: `python examples/vessel_segmentation_examples.py`
2. **Read documentation**: Review the main README.md
3. **Test with your data**: Start with parameter preview
4. **Configure MATLAB**: Set up u-shape3D integration if needed

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review the main README.md for usage examples
3. Run the test suite: `python tests/test_vessel_segmentation.py`
4. Open an issue on GitHub with detailed error messages