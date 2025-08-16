"""
3D Vessel Segmentation Pipeline for Scientific Applications

Production-ready pipeline for segmenting vascular networks from confocal microscopy data.
Optimized for biological imaging datasets with regional adaptive processing.

Dependencies:
    numpy, scipy, scikit-image, tifffile

Author: Philipp Kaintoch 
Date: 2025
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import tifffile
from scipy import ndimage
from skimage import exposure, filters, measure, morphology
from skimage.filters import frangi

# Suppress warnings for clean output
warnings.filterwarnings('ignore')


# Default threshold configurations
DEFAULT_THRESHOLDS = {
    'control': {'top': 2.0, 'middle': 1.5, 'bottom': 1.5},
    'ko_rescue': {'top': 5.0, 'middle': 3.0, 'bottom': 1.5}
}

DEFAULT_SCALES = [1.0, 1.5, 2.0, 2.5, 3.0]


@dataclass
class ConfigTemplate:
    """Template metadata for saved configurations."""
    name: str
    description: str
    sample_type: str
    created_date: str
    version: str = "1.0"


class ConfigTemplateManager:
    """Manages saving and loading of configuration templates."""
    
    @staticmethod
    def save_template(config: 'SegmentationConfig', 
                     name: str, 
                     output_dir: Path,
                     description: str = "",
                     sample_type: str = "general") -> Path:
        """Save configuration as reusable template."""
        template = ConfigTemplate(
            name=name,
            description=description,
            sample_type=sample_type,
            created_date=datetime.now().isoformat()
        )
        
        template_path = output_dir / f"{name}_template.json"
        template_path.parent.mkdir(parents=True, exist_ok=True)
        
        template_dict = {
            'metadata': asdict(template),
            'config': asdict(config)
        }
        
        with open(template_path, 'w') as f:
            json.dump(template_dict, f, indent=2)
        
        return template_path
    
    @staticmethod
    def load_template(template_path: Path) -> 'SegmentationConfig':
        """Load configuration from template file."""
        with open(template_path, 'r') as f:
            template_dict = json.load(f)
        
        config_data = template_dict['config']
        return SegmentationConfig(**config_data)
    
    @staticmethod
    def list_templates(template_dir: Path) -> List[Dict]:
        """List available templates with metadata."""
        templates = []
        if template_dir.exists():
            for template_file in template_dir.glob("*_template.json"):
                try:
                    with open(template_file, 'r') as f:
                        template_dict = json.load(f)
                    templates.append({
                        'file': str(template_file),
                        'name': template_dict['metadata']['name'],
                        'description': template_dict['metadata']['description'],
                        'sample_type': template_dict['metadata']['sample_type'],
                        'created_date': template_dict['metadata']['created_date']
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
        return templates


@dataclass 
class SegmentationConfig:
    """Configuration parameters for vessel segmentation pipeline."""
    
    # Preprocessing parameters
    gamma: float = 1.8
    bg_removal_radius: int = 8
    
    # Regional threshold parameters  
    threshold_percentiles: Dict[str, Dict[str, float]] = field(default_factory=lambda: DEFAULT_THRESHOLDS.copy())
    
    # Frangi filter parameters
    frangi_scales: List[float] = field(default_factory=lambda: DEFAULT_SCALES.copy())
    frangi_alpha: float = 0.5
    frangi_beta: float = 0.5  
    frangi_gamma: float = 15
    
    # Component selection parameters
    max_components: int = 6
    min_object_size: int = 50
    
    # Smoothing parameters
    distance_weight: float = 0.4
    sigma_smooth: float = 1.5
    
    # Processing parameters
    chunk_size: int = 15


class VesselSegmentationPipeline:
    """
    Production-ready 3D vessel segmentation pipeline.
    
    Implements regional adaptive thresholding and multi-scale vessel enhancement
    optimized for biological imaging datasets with varying intensity distributions.
    """
    
    def __init__(self, config: SegmentationConfig, verbose: bool = False):
        """Initialize pipeline with configuration."""
        self.config = config
        self.verbose = verbose
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not 1.0 <= self.config.gamma <= 3.0:
            raise ValueError(f"Gamma must be 1.0-3.0, got {self.config.gamma}")
        
        if not 0.1 <= self.config.distance_weight <= 0.8:
            raise ValueError(f"Distance weight must be 0.1-0.8, got {self.config.distance_weight}")
            
        if self.config.max_components < 1:
            raise ValueError(f"Max components must be >= 1, got {self.config.max_components}")
    
    def save_config_template(self, name: str, output_dir: Union[str, Path], 
                           description: str = "", sample_type: str = "general") -> Path:
        """Save current configuration as reusable template."""
        return ConfigTemplateManager.save_template(
            self.config, name, Path(output_dir), description, sample_type
        )
    
    @classmethod
    def from_template(cls, template_path: Union[str, Path], verbose: bool = False) -> 'VesselSegmentationPipeline':
        """Create pipeline instance from saved template."""
        config = ConfigTemplateManager.load_template(Path(template_path))
        return cls(config, verbose)
    
    def _print_progress(self, percentage: int, stage: str = "", force: bool = False):
        """Print progress bar for pipeline stages."""
        if not self.verbose and not force:
            return
            
        bar_length = 30
        filled_length = int(bar_length * percentage // 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\r[{bar}] {percentage}% {stage}", end='', flush=True)
        if percentage == 100:
            print()
    
    def segment_vessels(self, 
                       input_path: Union[str, Path], 
                       output_dir: Union[str, Path],
                       sample_type: str = 'control',
                       generate_configs: bool = True,
                       show_progress: bool = None) -> Tuple[np.ndarray, Dict]:
        """
        Main pipeline for vessel segmentation.
        
        Args:
            input_path: Path to input TIFF stack
            output_dir: Directory for output masks  
            sample_type: Sample type key ('control' or 'ko_rescue')
            generate_configs: Generate MATLAB and JSON config files
            show_progress: Display progress indicators (defaults to verbose setting)
            
        Returns:
            Tuple of (smooth_mask, metadata_dict)
        """
        if show_progress is None:
            show_progress = self.verbose
            
        if show_progress:
            print(f"Processing {Path(input_path).name}")
        
        # Load and validate input
        self._print_progress(10, "Loading image data...", show_progress)
        image = self._load_and_validate_image(input_path)
        
        # Preprocessing pipeline
        self._print_progress(25, "Preprocessing image...", show_progress)
        enhanced_image = self._preprocess_image(image)
        
        # Regional vessel detection  
        self._print_progress(45, "Detecting vessels...", show_progress)
        vessel_mask = self._detect_vessels_regional(enhanced_image, sample_type, show_progress)
        
        # Refine segmentation
        self._print_progress(80, "Refining segmentation...", show_progress)
        refined_mask = self._refine_segmentation(vessel_mask)
        
        # Create smooth mask for u-shape3D
        self._print_progress(90, "Creating smooth mask...", show_progress)
        smooth_mask = self._create_smooth_mask(refined_mask)
        
        # Generate outputs
        self._print_progress(100, "Saving outputs...", show_progress)
        metadata = self._save_outputs(smooth_mask, output_dir, sample_type, input_path, generate_configs)
        
        if show_progress:
            print("Segmentation completed successfully")
        
        return smooth_mask, metadata
    
    def preview_segmentation(self, 
                            input_path: Union[str, Path], 
                            sample_type: str = 'control',
                            preview_slices: int = 5,
                            show_progress: bool = None) -> Dict[str, Union[str, float]]:
        """
        Quick parameter validation on subset of slices.
        
        Args:
            input_path: Path to input TIFF stack
            sample_type: Sample type key for threshold selection
            preview_slices: Number of middle slices to process
            show_progress: Display progress indicators (defaults to verbose setting)
            
        Returns:
            Dictionary with coverage, components, recommendation, and advice
        """
        if show_progress is None:
            show_progress = self.verbose
            
        if show_progress:
            print("Running parameter preview...")
        
        # Load image
        self._print_progress(20, "Loading image data...", show_progress)
        image = self._load_and_validate_image(input_path)
        
        # Select middle slices for preview
        total_slices = image.shape[0]
        start_slice = max(0, (total_slices - preview_slices) // 2)
        end_slice = min(total_slices, start_slice + preview_slices)
        preview_image = image[start_slice:end_slice]
        
        # Run abbreviated pipeline
        self._print_progress(40, "Preprocessing image...", show_progress)
        enhanced = self._preprocess_image(preview_image)
        
        self._print_progress(60, "Detecting vessels...", show_progress)
        vessel_mask = self._detect_vessels_regional(enhanced, sample_type, False)
        
        self._print_progress(80, "Analyzing results...", show_progress)
        refined_mask = self._refine_segmentation(vessel_mask)
        
        # Calculate quality metrics
        coverage = float(np.sum(refined_mask) / refined_mask.size)
        components = int(measure.label(refined_mask).max()) if np.any(refined_mask) else 0
        
        # Generate recommendation
        if coverage < 0.005:
            recommendation = "check_parameters"
            advice = "Very low vessel coverage (<0.5%). Consider lowering thresholds."
        elif coverage > 0.20:
            recommendation = "check_parameters" 
            advice = "Very high coverage (>20%). May include noise. Consider higher thresholds."
        elif components > 15:
            recommendation = "check_parameters"
            advice = f"High fragmentation ({components} components). Consider adjusting max_components."
        elif components == 0:
            recommendation = "check_parameters"
            advice = "No vessels detected. Lower thresholds or check image quality."
        else:
            recommendation = "good"
            advice = "Parameters look suitable for full processing."
        
        self._print_progress(100, "Preview complete", show_progress)
        
        return {
            'coverage': coverage,
            'components': components,
            'recommendation': recommendation,
            'advice': advice,
            'preview_slices': f"{start_slice}-{end_slice}",
            'estimated_time': "~30 seconds"
        }
    
    def _load_and_validate_image(self, input_path: Union[str, Path]) -> np.ndarray:
        """Load TIFF stack and validate format."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        if not input_path.suffix.lower() in ['.tif', '.tiff']:
            raise ValueError(f"Input must be TIFF format, got {input_path.suffix}")
        
        # Load image
        img = tifffile.imread(str(input_path))
        
        # Validate dimensions (expect 3D: Z, Y, X)
        if img.ndim != 3:
            raise ValueError(f"Expected 3D image (Z,Y,X), got shape {img.shape}")
            
        # Convert to float and normalize
        img = img.astype(np.float32)
        if img.max() > 0:
            img = img / img.max()
            
        return img
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline for contrast enhancement."""
        # Gamma correction for contrast enhancement
        enhanced = exposure.adjust_gamma(image, gamma=self.config.gamma)
        
        # Background subtraction using morphological operations
        selem = morphology.ball(self.config.bg_removal_radius)
        background = ndimage.minimum_filter(enhanced, footprint=selem)
        background = ndimage.maximum_filter(background, footprint=selem)
        corrected = enhanced - background
        
        # Normalize and clip
        corrected = np.clip(corrected, 0, 1)
        if corrected.max() > 0:
            corrected = corrected / corrected.max()
            
        return corrected
    
    def _detect_vessels_regional(self, 
                                image: np.ndarray, 
                                sample_type: str,
                                show_progress: bool = False) -> np.ndarray:
        """Regional adaptive vessel detection with multi-scale enhancement."""
        # Get threshold parameters for sample type
        if sample_type not in self.config.threshold_percentiles:
            sample_type = 'control'  # Fallback to control parameters
            
        thresholds = self.config.threshold_percentiles[sample_type]
        
        # Divide image into regions
        height = image.shape[1]  # Y-axis dimension
        top_boundary = height // 3
        bottom_boundary = 2 * height // 3
        
        # Extract regional data
        top_region = image[:, :top_boundary, :]
        middle_region = image[:, top_boundary:bottom_boundary, :]
        bottom_region = image[:, bottom_boundary:, :]
        
        # Calculate regional thresholds
        def safe_percentile(region, percentile):
            valid_pixels = region[region > 0]
            return np.percentile(valid_pixels, percentile) if len(valid_pixels) > 100 else 0.0001
        
        top_thresh = safe_percentile(top_region, thresholds['top'])
        middle_thresh = safe_percentile(middle_region, thresholds['middle'])
        bottom_thresh = safe_percentile(bottom_region, thresholds['bottom'])
        
        # Apply regional thresholds
        basic_mask = np.zeros_like(image, dtype=bool)
        basic_mask[:, :top_boundary, :] = top_region > top_thresh
        basic_mask[:, top_boundary:bottom_boundary, :] = middle_region > middle_thresh
        basic_mask[:, bottom_boundary:, :] = bottom_region > bottom_thresh
        
        # Add overlap zones for continuity
        overlap = 10
        if top_boundary > overlap:
            overlap_region = image[:, top_boundary-overlap:top_boundary+overlap, :]
            overlap_thresh = (top_thresh + middle_thresh) / 2
            basic_mask[:, top_boundary-overlap:top_boundary+overlap, :] |= overlap_region > overlap_thresh
            
        if bottom_boundary > overlap and bottom_boundary < height - overlap:
            overlap_region = image[:, bottom_boundary-overlap:bottom_boundary+overlap, :]
            overlap_thresh = (middle_thresh + bottom_thresh) / 2
            basic_mask[:, bottom_boundary-overlap:bottom_boundary+overlap, :] |= overlap_region > overlap_thresh
        
        # Apply multi-scale Frangi enhancement
        if show_progress:
            self._print_progress(65, "Applying Frangi filter...", True)
        frangi_mask = self._apply_frangi_multiscale(image)
        
        # Combine basic thresholding with vessel enhancement
        combined_mask = basic_mask | frangi_mask
        
        return combined_mask
    
    def _apply_frangi_multiscale(self, image: np.ndarray) -> np.ndarray:
        """Apply Frangi filter across multiple scales with chunked processing."""
        frangi_response = np.zeros_like(image)
        
        # Process in chunks for memory efficiency
        for z_start in range(0, image.shape[0], self.config.chunk_size):
            z_end = min(z_start + self.config.chunk_size, image.shape[0])
            chunk = image[z_start:z_end]
            
            chunk_response = np.zeros_like(chunk)
            for sigma in self.config.frangi_scales:
                try:
                    response = frangi(
                        chunk, 
                        sigmas=[sigma], 
                        alpha=self.config.frangi_alpha,
                        beta=self.config.frangi_beta, 
                        gamma=self.config.frangi_gamma,
                        black_ridges=False
                    )
                    chunk_response = np.maximum(chunk_response, response)
                except (ValueError, RuntimeError, MemoryError):
                    # Silently skip failed scales in production mode
                    continue
                    
            frangi_response[z_start:z_end] = chunk_response
        
        # Normalize Frangi response
        if frangi_response.max() > 0:
            frangi_response = frangi_response / frangi_response.max()
        
        # Create enhanced mask with conservative threshold
        vessel_threshold = np.percentile(frangi_response[frangi_response > 0], 90)
        enhanced_mask = frangi_response > vessel_threshold
        
        return enhanced_mask
    
    def _refine_segmentation(self, mask: np.ndarray) -> np.ndarray:
        """Refine vessel segmentation through component selection and noise removal."""
        # Remove small noise objects
        mask = morphology.remove_small_objects(mask, min_size=self.config.min_object_size)
        
        # Component selection
        labeled = measure.label(mask)
        if labeled.max() > 0:
            component_sizes = np.bincount(labeled.flat)[1:]
            n_components = min(self.config.max_components, len(component_sizes))
            
            if n_components > 0:
                # Select largest components
                large_components = np.argsort(component_sizes)[-n_components:]
                mask = np.isin(labeled, large_components + 1)
        
        return mask
    
    def _create_smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create smooth mask optimized for u-shape3D isosurface generation."""
        # Create distance transform for smooth gradients
        distance_from_vessels = ndimage.distance_transform_edt(~mask)
        max_distance = 5
        distance_from_vessels = np.clip(distance_from_vessels, 0, max_distance)
        distance_from_vessels = 1 - (distance_from_vessels / max_distance)
        
        # Combine vessel mask with distance transform
        vessel_prob = mask.astype(float)
        smooth_mask = ((1 - self.config.distance_weight) * vessel_prob + 
                      self.config.distance_weight * distance_from_vessels)
        
        # Apply Gaussian smoothing
        smooth_mask = ndimage.gaussian_filter(smooth_mask, sigma=self.config.sigma_smooth)
        
        # Ensure single component after smoothing
        mask_binary = smooth_mask > 0.1
        labeled = measure.label(mask_binary)
        if labeled.max() > 0:
            largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
            largest_component = labeled == largest_label
            smooth_mask = smooth_mask * largest_component
        
        # Final normalization
        if smooth_mask.max() > 0:
            smooth_mask = smooth_mask / smooth_mask.max()
            
        return smooth_mask
    
    def _save_mask_files(self, smooth_mask: np.ndarray, output_dir: Path, input_name: str) -> Dict[str, str]:
        """Save 3D mask and individual slices."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save 3D mask
        mask_path = output_dir / f"{input_name}_vessel_mask.tif"
        tifffile.imwrite(str(mask_path), (smooth_mask * 255).astype(np.uint8))
        
        # Save individual slices for u-shape3D
        slices_dir = output_dir / f"{input_name}_slices"
        slices_dir.mkdir(exist_ok=True)
        
        for i, slice_mask in enumerate(smooth_mask):
            slice_path = slices_dir / f"slice_{i:03d}.tif"
            tifffile.imwrite(str(slice_path), (slice_mask * 255).astype(np.uint8))
        
        return {
            'mask_3d': str(mask_path),
            'slices_dir': str(slices_dir)
        }
    
    def _create_metadata(self, smooth_mask: np.ndarray, sample_type: str, file_paths: Dict[str, str]) -> Dict:
        """Generate processing metadata."""
        return {
            'sample_type': sample_type,
            'vessel_coverage': float(np.sum(smooth_mask > 0.1) / smooth_mask.size),
            'components': int(measure.label(smooth_mask > 0.1).max()),
            'output_files': file_paths.copy(),
            'parameters_used': {
                'gamma': self.config.gamma,
                'thresholds': self.config.threshold_percentiles.get(sample_type, {}),
                'frangi_scales': self.config.frangi_scales,
                'sigma_smooth': self.config.sigma_smooth
            }
        }
    
    def _generate_config_files(self, output_dir: Path, input_name: str) -> Dict[str, str]:
        """Generate MATLAB configuration files."""
        config_paths = {}
        
        # Generate MATLAB configuration
        matlab_config_path = self._generate_matlab_config(output_dir, input_name)
        config_paths['matlab_config'] = matlab_config_path
        
        return config_paths
    
    def _save_metadata_file(self, metadata: Dict, output_dir: Path, input_name: str) -> str:
        """Save metadata JSON file."""
        metadata_path = output_dir / f"{input_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return str(metadata_path)
    
    def _save_outputs(self, 
                     smooth_mask: np.ndarray, 
                     output_dir: Union[str, Path],
                     sample_type: str,
                     input_path: Union[str, Path],
                     generate_configs: bool = True) -> Dict:
        """Save outputs and generate metadata."""
        output_dir = Path(output_dir)
        input_name = Path(input_path).stem
        
        # Save mask files
        file_paths = self._save_mask_files(smooth_mask, output_dir, input_name)
        
        # Generate metadata
        metadata = self._create_metadata(smooth_mask, sample_type, file_paths)
        
        # Generate config files if requested
        if generate_configs:
            config_paths = self._generate_config_files(output_dir, input_name)
            metadata['output_files'].update(config_paths)
            self._save_metadata_file(metadata, output_dir, input_name)
        
        return metadata
    
    def _generate_matlab_config(self, output_dir: Path, sample_name: str) -> str:
        """Generate MATLAB configuration for u-shape3D."""
        config_content = f'''% u-shape3D Configuration for {sample_name}
% Generated by vessel segmentation pipeline

p.meshMode = 'loadMask';
p.maskDir = '{output_dir / f"{sample_name}_slices"}';
p.maskName = 'slice_';
p.smoothImageSize = 1.0;  % Pre-smoothed masks
p.scaleOtsu = 0.35;       % Optimized for vessel masks
p.smoothMeshIterations = 10;
p.curvatureMedianFilterRadius = 2;
p.removeSmallComponents = 1;

% Processing parameters used:
% Gamma correction: {self.config.gamma}
% Smoothing sigma: {self.config.sigma_smooth}
% Distance weight: {self.config.distance_weight}
'''
        
        config_path = output_dir / f"{sample_name}_config.m"
        with open(config_path, 'w') as f:
            f.write(config_content)
            
        return str(config_path)


def create_standard_configs() -> Dict[str, SegmentationConfig]:
    """Create standard configurations for different sample types."""
    return {
        'control': SegmentationConfig(
            gamma=1.8,
            max_components=1,
            sigma_smooth=1.5
        ),
        
        'ko_rescue': SegmentationConfig(
            gamma=1.8,
            max_components=6,
            sigma_smooth=1.2
        ),
        
        'high_detail': SegmentationConfig(
            gamma=1.8,
            frangi_scales=[0.5, 1.0, 1.5, 2.0],
            sigma_smooth=1.2,
            max_components=3
        ),
        
        'large_vessels': SegmentationConfig(
            gamma=1.5,
            frangi_scales=[2.0, 3.0, 4.0, 5.0],
            sigma_smooth=2.0,
            max_components=1
        )
    }
