"""
VesselSegmenter: 3D Vascular Network Segmentation Pipeline

A production-ready pipeline for segmenting vascular networks from confocal microscopy data,
optimized for biological imaging with regional adaptive processing.

Author: Philipp Kaintoch
Date: 2025
Version: 1.0.0
"""

from .vessel_segmentation_pipeline import (
    SegmentationConfig,
    VesselSegmentationPipeline,
    ConfigTemplateManager,
    create_standard_configs
)

__version__ = "1.0.0"
__author__ = "Philipp Kaintoch"
__email__ = "your.email@domain.com"  # Update as needed

__all__ = [
    "SegmentationConfig",
    "VesselSegmentationPipeline", 
    "ConfigTemplateManager",
    "create_standard_configs"
]