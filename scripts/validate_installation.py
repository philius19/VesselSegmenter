#!/usr/bin/env python3
"""
Installation Validation Script for VesselSegmenter

This script validates that VesselSegmenter is properly installed and all
dependencies are available.

Usage:
    python scripts/validate_installation.py
"""

import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_dependencies():
    """Check required dependencies."""
    print("\nChecking dependencies...")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('scikit-image', 'skimage'),
        ('tifffile', 'tifffile')
    ]
    
    optional_packages = [
        ('matplotlib', 'matplotlib'),
        ('jupyter', 'jupyter')
    ]
    
    all_required_available = True
    
    # Check required packages
    for display_name, import_name in required_packages:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {display_name} ({version})")
        except ImportError:
            print(f"  ✗ {display_name} (missing)")
            all_required_available = False
    
    # Check optional packages
    print("\nChecking optional dependencies...")
    for display_name, import_name in optional_packages:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {display_name} ({version})")
        except ImportError:
            print(f"  ○ {display_name} (optional, not installed)")
    
    return all_required_available

def check_vesselsegmenter():
    """Check VesselSegmenter package."""
    print("\nChecking VesselSegmenter package...")
    
    try:
        # Add src to path if needed
        src_path = Path(__file__).parent.parent / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        from vessel_segmentation_pipeline import (
            SegmentationConfig,
            VesselSegmentationPipeline,
            ConfigTemplateManager,
            create_standard_configs
        )
        
        print("  ✓ Core imports successful")
        
        # Test basic functionality
        config = SegmentationConfig()
        pipeline = VesselSegmentationPipeline(config)
        print("  ✓ Pipeline creation successful")
        
        # Test standard configs
        standard_configs = create_standard_configs()
        print(f"  ✓ Standard configurations available ({len(standard_configs)} configs)")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        return False

def check_examples():
    """Check examples availability."""
    print("\nChecking examples...")
    
    try:
        examples_path = Path(__file__).parent.parent / "examples"
        if examples_path.exists():
            sys.path.insert(0, str(examples_path))
        
        from vessel_segmentation_examples import main
        print("  ✓ Examples module available")
        return True
        
    except ImportError as e:
        print(f"  ○ Examples not available: {e}")
        return False

def check_matlab_integration():
    """Check MATLAB integration files."""
    print("\nChecking MATLAB integration...")
    
    matlab_path = Path(__file__).parent.parent / "matlab" / "VesselMeshGeneration.m"
    
    if matlab_path.exists():
        print("  ✓ MATLAB integration script available")
        return True
    else:
        print("  ○ MATLAB integration script not found")
        return False

def run_simple_test():
    """Run a simple functionality test."""
    print("\nRunning simple functionality test...")
    
    try:
        import tempfile
        import numpy as np
        
        # Add src to path if needed
        src_path = Path(__file__).parent.parent / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        from vessel_segmentation_pipeline import SegmentationConfig, VesselSegmentationPipeline
        
        # Create synthetic test data
        test_image = np.random.rand(10, 50, 50).astype(np.float32)
        
        # Create pipeline
        config = SegmentationConfig()
        pipeline = VesselSegmentationPipeline(config, verbose=False)
        
        # Test preprocessing
        processed = pipeline._preprocess_image(test_image)
        assert processed.shape == test_image.shape
        
        print("  ✓ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        return False

def main():
    """Run complete validation."""
    print("=" * 60)
    print("VESSELSEGMENTER INSTALLATION VALIDATION")
    print("=" * 60)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_vesselsegmenter,
        check_examples,
        check_matlab_integration,
        run_simple_test
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Check failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    if results[0] and results[1] and results[2]:  # Critical checks
        print("🎉 INSTALLATION SUCCESSFUL!")
        print("VesselSegmenter is ready to use.")
        
        if results[3]:
            print("📖 Examples are available for learning.")
        
        if results[4]:
            print("🔧 MATLAB integration is available.")
        
        if results[5]:
            print("✅ All functionality tests passed.")
        
    else:
        print("❌ INSTALLATION INCOMPLETE")
        print("Please address the issues above.")
        
        if not results[0]:
            print("  → Upgrade Python to 3.8 or higher")
        
        if not results[1]:
            print("  → Install missing dependencies: pip install -r requirements.txt")
        
        if not results[2]:
            print("  → Check VesselSegmenter package installation")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)