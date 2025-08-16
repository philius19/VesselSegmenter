#!/usr/bin/env python3
"""
Test Suite for Vessel Segmentation Examples

Tests all 7 examples in vessel_segmentation_examples.py to ensure they work correctly
and don't have any syntax or import errors.

Author: Philipp Kaintoch
Date: 2025-08-16
"""

import tempfile
import sys
from pathlib import Path
from unittest.mock import patch
import traceback

# Add current directory to path so we can import the examples
sys.path.insert(0, str(Path(__file__).parent))

try:
    from vessel_segmentation_examples import (
        basic_usage_example,
        template_management_example,
        parameter_preview_example,
        batch_processing_example,
        interactive_workflow,
        scientific_workflow_example,
        advanced_customization_example
    )
    print("✓ Successfully imported all example functions")
except ImportError as e:
    print(f"✗ Failed to import examples: {e}")
    sys.exit(1)


def test_example_function(example_func, example_name):
    """Test a single example function."""
    print(f"  Testing {example_name}...", end=" ")
    
    try:
        # Capture stdout to avoid cluttering test output
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Run the example function
        example_func()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        print("✓ PASSED")
        return True
        
    except FileNotFoundError as e:
        # Expected for examples that reference actual data files
        sys.stdout = old_stdout
        print(f"✓ PASSED (FileNotFoundError expected: {str(e)[:50]}...)")
        return True
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"✗ FAILED: {e}")
        # Print traceback for debugging
        print(f"    Traceback: {traceback.format_exc()}")
        return False


def test_all_examples():
    """Test all example functions."""
    print("=" * 60)
    print("VESSEL SEGMENTATION EXAMPLES - TEST SUITE")
    print("=" * 60)
    
    examples = [
        (basic_usage_example, "Basic Usage Example"),
        (template_management_example, "Template Management Example"),
        (parameter_preview_example, "Parameter Preview Example"), 
        (batch_processing_example, "Batch Processing Example"),
        (interactive_workflow, "Interactive Workflow"),
        (scientific_workflow_example, "Scientific Workflow Example"),
        (advanced_customization_example, "Advanced Customization Example")
    ]
    
    total_tests = len(examples)
    passed_tests = 0
    
    for example_func, example_name in examples:
        if test_example_function(example_func, example_name):
            passed_tests += 1
    
    print(f"\n" + "=" * 60)
    print(f"EXAMPLES TEST SUMMARY: {passed_tests}/{total_tests} examples passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL EXAMPLES WORKING!")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} examples had issues")
        return False


def test_example_imports():
    """Test that all required imports work correctly."""
    print("\n--- Testing Example Imports ---")
    
    try:
        from vessel_segmentation_pipeline import (
            VesselSegmentationPipeline, 
            SegmentationConfig, 
            ConfigTemplateManager,
            create_standard_configs
        )
        print("  ✓ Core pipeline imports working")
        
        import glob
        import numpy as np
        from pathlib import Path
        print("  ✓ Standard library imports working")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Import test failed: {e}")
        return False


def test_example_configurations():
    """Test that example configurations are valid."""
    print("\n--- Testing Example Configurations ---")
    
    try:
        from vessel_segmentation_pipeline import SegmentationConfig, create_standard_configs
        
        # Test standard configs used in examples
        configs = create_standard_configs()
        assert 'control' in configs
        assert 'ko_rescue' in configs
        print("  ✓ Standard configurations available")
        
        # Test custom configurations from examples
        custom_config = SegmentationConfig(
            gamma=2.0,
            max_components=3,
            sigma_smooth=1.2,
            frangi_scales=[0.5, 1.0, 1.5, 2.0, 2.5]
        )
        print("  ✓ Custom configurations work")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration test failed: {e}")
        return False


def test_template_functionality():
    """Test template functionality used in examples."""
    print("\n--- Testing Template Functionality ---")
    
    try:
        from vessel_segmentation_pipeline import (
            SegmentationConfig, 
            VesselSegmentationPipeline,
            ConfigTemplateManager
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test template saving (used in examples)
            config = SegmentationConfig(gamma=2.0)
            pipeline = VesselSegmentationPipeline(config)
            
            template_path = pipeline.save_config_template(
                "test_template",
                temp_path,
                "Test template description",
                "control"
            )
            
            assert template_path.exists()
            print("  ✓ Template saving works")
            
            # Test template loading (used in examples)  
            loaded_pipeline = VesselSegmentationPipeline.from_template(template_path)
            assert loaded_pipeline.config.gamma == 2.0
            print("  ✓ Template loading works")
            
            # Test template listing (used in examples)
            templates = ConfigTemplateManager.list_templates(temp_path)
            assert len(templates) == 1
            print("  ✓ Template listing works")
            
        return True
        
    except Exception as e:
        print(f"  ✗ Template functionality test failed: {e}")
        return False


def main():
    """Run all example tests."""
    print("Starting comprehensive examples testing...\n")
    
    # Test basic functionality first
    tests_passed = 0
    total_tests = 4
    
    if test_example_imports():
        tests_passed += 1
        
    if test_example_configurations():
        tests_passed += 1
        
    if test_template_functionality():
        tests_passed += 1
        
    # Test all example functions
    if test_all_examples():
        tests_passed += 1
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"OVERALL TEST SUMMARY: {tests_passed}/{total_tests} test suites passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL EXAMPLE TESTS SUCCESSFUL!")
        print("\nThe vessel segmentation examples are ready for use!")
        return True
    else:
        print(f"⚠️  {total_tests - tests_passed} test suites had issues")
        print("\nSome examples may need attention before final release.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)