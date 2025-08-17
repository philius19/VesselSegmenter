"""
Setup script for VesselSegmenter package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')
requirements = [req for req in requirements if req and not req.startswith('#')]

setup(
    name="vessel-segmenter",
    version="1.0.0",
    author="Philipp Kaintoch",
    author_email="p.kaintoch@uni-muenster.de",  
    description="3D Vascular Network Segmentation Pipeline for Confocal Microscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/philius19/VesselSegmenter",  
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov>=2.0"],
        "examples": ["matplotlib>=3.3.0", "jupyter>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "vessel-segmenter=vessel_segmentation_pipeline:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)