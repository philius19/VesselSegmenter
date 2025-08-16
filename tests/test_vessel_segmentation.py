#!/usr/bin/env python3
"""
Comprehensive Test Suite for Vessel Segmentation Pipeline

Tests all core functionality of the vessel segmentation pipeline including:
- Configuration management
- Image loading and validation  
- Preprocessing operations
- Vessel detection algorithms
- Output generation
- MATLAB integration
- Template management

Author: Philipp Kaintoch
Date: 2025-08-16
"""

import numpy as np
import tempfile
import json
from pathlib import Path
import tifffile
import traceback

# Import the vessel segmentation components
from vessel_segmentation_pipeline import (
    SegmentationConfig,
    VesselSegmentationPipeline,
    ConfigTemplateManager,
    create_standard_configs
)

class TestSegmentationConfig:
    """Test configuration management and validation."""
    
    def test_default_config_creation(self):
        """Test default configuration initialization."""
        config = SegmentationConfig()
        
        assert config.gamma == 1.8
        assert config.bg_removal_radius == 8
        assert config.max_components == 6
        assert config.sigma_smooth == 1.5
        assert len(config.frangi_scales) == 5
        
    def test_custom_config_creation(self):
        """Test custom configuration parameters."""
        config = SegmentationConfig(
            gamma=2.0,
            max_components=3,
            sigma_smooth=1.2
        )
        
        assert config.gamma == 2.0
        assert config.max_components == 3
        assert config.sigma_smooth == 1.2
        
    def test_config_validation_in_pipeline(self):
        """Test configuration validation during pipeline initialization."""
        # Valid configuration
        valid_config = SegmentationConfig(gamma=2.0, distance_weight=0.5)
        pipeline = VesselSegmentationPipeline(valid_config)
        assert pipeline.config.gamma == 2.0
        
        # Invalid gamma
        try:
            invalid_config = SegmentationConfig(gamma=5.0)
            VesselSegmentationPipeline(invalid_config)
            assert False, "Should have raised ValueError for invalid gamma"
        except ValueError as e:
            assert "Gamma must be 1.0-3.0" in str(e)
            
        # Invalid distance weight
        try:
            invalid_config = SegmentationConfig(distance_weight=1.5)
            VesselSegmentationPipeline(invalid_config)
            assert False, "Should have raised ValueError for invalid distance weight"
        except ValueError as e:
            assert "Distance weight must be 0.1-0.8" in str(e)
            
        # Invalid max components
        try:
            invalid_config = SegmentationConfig(max_components=0)
            VesselSegmentationPipeline(invalid_config)
            assert False, "Should have raised ValueError for invalid max components"
        except ValueError as e:
            assert "Max components must be >= 1" in str(e)


class TestVesselSegmentationPipeline:
    """Test core pipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SegmentationConfig()
        self.pipeline = VesselSegmentationPipeline(self.config, verbose=False)
        
        # Create synthetic test image
        self.test_image = self._create_synthetic_vessel_image()
        
    def _create_synthetic_vessel_image(self):
        """Create a synthetic 3D vessel image for testing."""
        # Create a 3D image with vessel-like structures
        shape = (20, 100, 100)  # Small for fast testing
        image = np.zeros(shape, dtype=np.float32)
        
        # Add some vessel-like tubular structures
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    # Vertical vessel
                    if (x - 30)**2 + (y - 50)**2 < 25:
                        image[z, y, x] = 0.8
                    
                    # Horizontal vessel in middle region
                    if 40 <= y <= 60 and (x - 50)**2 + (z - 10)**2 < 16:
                        image[z, y, x] = 0.6
        
        # Add some noise
        image += np.random.normal(0, 0.05, shape).astype(np.float32)
        image = np.clip(image, 0, 1)
        
        return image
        
    def test_image_loading_and_validation(self):
        """Test image loading with various formats and error conditions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test valid TIFF loading
            valid_tiff = temp_path / "test_valid.tif"
            tifffile.imwrite(str(valid_tiff), (self.test_image * 255).astype(np.uint8))
            
            loaded_image = self.pipeline._load_and_validate_image(valid_tiff)
            assert loaded_image.shape == self.test_image.shape
            assert loaded_image.dtype == np.float32
            assert 0 <= loaded_image.max() <= 1
            
            # Test file not found
            try:
                self.pipeline._load_and_validate_image(temp_path / "nonexistent.tif")
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                pass  # Expected
                
            # Test invalid format
            invalid_format = temp_path / "test.png"
            invalid_format.write_text("dummy")
            try:
                self.pipeline._load_and_validate_image(invalid_format)
                assert False, "Should have raised ValueError for invalid format"
            except ValueError as e:
                assert "Input must be TIFF format" in str(e)
                
            # Test invalid dimensions (2D image)
            invalid_2d = temp_path / "test_2d.tif"
            tifffile.imwrite(str(invalid_2d), np.zeros((100, 100), dtype=np.uint8))
            try:
                self.pipeline._load_and_validate_image(invalid_2d)
                assert False, "Should have raised ValueError for 2D image"
            except ValueError as e:
                assert "Expected 3D image" in str(e)
                
    def test_preprocessing_pipeline(self):
        """Test image preprocessing operations."""
        processed = self.pipeline._preprocess_image(self.test_image)
        
        # Check output properties
        assert processed.shape == self.test_image.shape
        assert processed.dtype == np.float32
        assert 0 <= processed.min()
        assert processed.max() <= 1
        
        # Check that gamma correction increases contrast
        mean_original = np.mean(self.test_image)
        mean_processed = np.mean(processed)
        assert mean_processed != mean_original  # Should be different due to gamma
        
    def test_regional_vessel_detection(self):
        """Test regional adaptive vessel detection."""
        enhanced = self.pipeline._preprocess_image(self.test_image)
        vessel_mask = self.pipeline._detect_vessels_regional(enhanced, 'control')
        
        # Check output properties
        assert vessel_mask.shape == enhanced.shape
        assert vessel_mask.dtype == bool
        
        # Should detect some vessels
        assert np.sum(vessel_mask) > 0
        
        # Test different sample types
        ko_mask = self.pipeline._detect_vessels_regional(enhanced, 'ko_rescue')
        assert ko_mask.shape == enhanced.shape
        assert ko_mask.dtype == bool
        
    def test_frangi_multiscale_filter(self):
        """Test Frangi filter application."""
        enhanced = self.pipeline._preprocess_image(self.test_image)
        frangi_mask = self.pipeline._apply_frangi_multiscale(enhanced)
        
        # Check output properties
        assert frangi_mask.shape == enhanced.shape
        assert frangi_mask.dtype == bool
        
    def test_segmentation_refinement(self):
        """Test segmentation refinement operations."""
        # Create a simple binary mask with small and large components
        mask = np.zeros((20, 100, 100), dtype=bool)
        
        # Large component
        mask[5:15, 30:70, 30:70] = True
        
        # Small noise component
        mask[2:4, 5:8, 5:8] = True
        
        refined = self.pipeline._refine_segmentation(mask)
        
        # Check that small components are removed
        assert np.sum(refined) < np.sum(mask)
        
        # Check that we still have the main component
        assert np.sum(refined) > 0
        
    def test_smooth_mask_creation(self):
        """Test smooth mask generation for u-shape3D."""
        # Create binary vessel mask
        mask = np.zeros((10, 50, 50), dtype=bool)
        mask[2:8, 20:30, 20:30] = True
        
        smooth_mask = self.pipeline._create_smooth_mask(mask)
        
        # Check output properties
        assert smooth_mask.shape == mask.shape
        assert smooth_mask.dtype == np.float64
        assert 0 <= smooth_mask.min()
        assert smooth_mask.max() <= 1
        
        # Should have smooth gradients (non-binary)
        unique_values = len(np.unique(smooth_mask))
        assert unique_values > 2  # More than just 0 and 1
        
    def test_preview_segmentation(self):
        """Test parameter preview functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test TIFF file
            test_tiff = temp_path / "test_preview.tif"
            tifffile.imwrite(str(test_tiff), (self.test_image * 255).astype(np.uint8))
            
            # Test preview with different parameters
            preview_result = self.pipeline.preview_segmentation(
                test_tiff, 
                sample_type='control',
                preview_slices=5
            )
            
            # Check return structure
            assert 'coverage' in preview_result
            assert 'components' in preview_result
            assert 'recommendation' in preview_result
            assert 'advice' in preview_result
            
            # Check value ranges
            assert 0 <= preview_result['coverage'] <= 1
            assert preview_result['components'] >= 0
            assert preview_result['recommendation'] in ['good', 'check_parameters']
            
    def test_full_pipeline_integration(self):
        """Test complete end-to-end pipeline execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test input
            input_file = temp_path / "test_input.tif"
            tifffile.imwrite(str(input_file), (self.test_image * 255).astype(np.uint8))
            
            # Create output directory
            output_dir = temp_path / "output"
            
            # Run full pipeline
            mask, metadata = self.pipeline.segment_vessels(
                input_path=input_file,
                output_dir=output_dir,
                sample_type='control',
                generate_configs=True,
                show_progress=False
            )
            
            # Check outputs
            assert mask.shape == self.test_image.shape
            assert mask.dtype == np.float64
            
            # Check metadata structure
            assert 'vessel_coverage' in metadata
            assert 'components' in metadata
            assert 'output_files' in metadata
            assert 'sample_type' in metadata
            
            # Check output files exist
            assert output_dir.exists()
            assert len(list(output_dir.glob("*.tif"))) > 0  # Should have mask files
            
            if metadata['output_files'].get('matlab_config'):
                matlab_config = Path(metadata['output_files']['matlab_config'])
                assert matlab_config.exists()
                
    def test_verbose_mode(self):
        """Test verbose vs silent mode operation."""
        # Test verbose mode
        verbose_pipeline = VesselSegmentationPipeline(self.config, verbose=True)
        assert verbose_pipeline.verbose == True
        
        # Test silent mode (default)
        silent_pipeline = VesselSegmentationPipeline(self.config, verbose=False)
        assert silent_pipeline.verbose == False
        
        # Test that verbose mode can be overridden in methods
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_tiff = temp_path / "test.tif"
            tifffile.imwrite(str(test_tiff), (self.test_image * 255).astype(np.uint8))
            
            # Should work without raising errors regardless of verbosity
            result = silent_pipeline.preview_segmentation(
                test_tiff, show_progress=False
            )
            assert 'coverage' in result


class TestConfigTemplateManager:
    """Test configuration template management."""
    
    def test_template_save_and_load(self):
        """Test saving and loading configuration templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create config and save as template
            config = SegmentationConfig(gamma=2.0, max_components=3)
            template_path = ConfigTemplateManager.save_template(
                config, 
                "test_template",
                temp_path,
                "Test description",
                "test_type"
            )
            
            # Check template file exists
            assert template_path.exists()
            assert template_path.name == "test_template_template.json"
            
            # Load template
            loaded_config = ConfigTemplateManager.load_template(template_path)
            
            # Check loaded config matches original
            assert loaded_config.gamma == 2.0
            assert loaded_config.max_components == 3
            
    def test_template_listing(self):
        """Test listing available templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple templates
            config1 = SegmentationConfig(gamma=1.5)
            config2 = SegmentationConfig(gamma=2.0)
            
            ConfigTemplateManager.save_template(config1, "template1", temp_path, "First template", "control")
            ConfigTemplateManager.save_template(config2, "template2", temp_path, "Second template", "ko_rescue")
            
            # List templates
            templates = ConfigTemplateManager.list_templates(temp_path)
            
            assert len(templates) == 2
            template_names = [t['name'] for t in templates]
            assert "template1" in template_names
            assert "template2" in template_names
            
    def test_pipeline_template_integration(self):
        """Test template functionality integrated with pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create pipeline and save template
            config = SegmentationConfig(gamma=2.2)
            pipeline = VesselSegmentationPipeline(config)
            
            template_path = pipeline.save_config_template(
                "pipeline_test",
                temp_path,
                "Pipeline test template",
                "control"
            )
            
            # Create new pipeline from template
            new_pipeline = VesselSegmentationPipeline.from_template(template_path)
            
            assert new_pipeline.config.gamma == 2.2


class TestStandardConfigs:
    """Test standard configuration presets."""
    
    def test_standard_configs_creation(self):
        """Test creation of standard configuration presets."""
        configs = create_standard_configs()
        
        # Check all expected configs exist
        expected_configs = ['control', 'ko_rescue', 'high_detail', 'large_vessels']
        for config_name in expected_configs:
            assert config_name in configs
            assert isinstance(configs[config_name], SegmentationConfig)
            
        # Check some specific parameters
        assert configs['control'].max_components == 1
        assert configs['ko_rescue'].max_components == 6
        assert len(configs['high_detail'].frangi_scales) == 4
        assert len(configs['large_vessels'].frangi_scales) == 4
        
    def test_configs_pipeline_compatibility(self):
        """Test that all standard configs work with pipeline."""
        configs = create_standard_configs()
        
        for config_name, config in configs.items():
            # Should not raise validation errors
            pipeline = VesselSegmentationPipeline(config)
            assert pipeline.config == config


class TestMATLABIntegration:
    """Test MATLAB configuration generation."""
    
    def test_matlab_config_generation(self):
        """Test MATLAB configuration file generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config = SegmentationConfig()
            pipeline = VesselSegmentationPipeline(config)
            
            matlab_path = pipeline._generate_matlab_config(temp_path, "test_sample")
            
            # Check file exists and has expected content
            matlab_file = Path(matlab_path)
            assert matlab_file.exists()
            assert matlab_file.suffix == '.m'
            
            content = matlab_file.read_text()
            assert "p.meshMode = 'loadMask';" in content
            assert "p.scaleOtsu = 0.35;" in content
            assert f"Gamma correction: {config.gamma}" in content


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("VESSEL SEGMENTATION PIPELINE - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    test_classes = [
        TestSegmentationConfig,
        TestVesselSegmentationPipeline, 
        TestConfigTemplateManager,
        TestStandardConfigs,
        TestMATLABIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n--- Testing {test_class.__name__} ---")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            print(f"  Running {test_method}...", end=" ")
            
            try:
                # Create test instance
                test_instance = test_class()
                
                # Run setup if exists
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run the test
                getattr(test_instance, test_method)()
                
                print("✓ PASSED")
                passed_tests += 1
                
            except Exception as e:
                print(f"✗ FAILED: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)