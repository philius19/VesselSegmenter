"""
Vessel Segmentation Pipeline - Usage Examples and Tutorials

This file demonstrates all features of the VesselSegmentationPipeline with
comprehensive examples, interactive workflows, and best practices for 
scientific image processing applications.

Run sections individually or execute main() for guided tutorial.

Author: Philipp Kaintoch
Date: 2025
"""

import glob
from pathlib import Path
from typing import Dict, List

from vesselsegmenter import (
    VesselSegmentationPipeline, 
    SegmentationConfig, 
    ConfigTemplateManager,
    create_standard_configs
)


def basic_usage_example():
    """
    Example 1: Basic vessel segmentation workflow
    
    Demonstrates the simplest way to process a single image with default settings.
    Perfect for getting started or quick processing tasks.
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    # Basic setup with default configuration
    config = SegmentationConfig()
    pipeline = VesselSegmentationPipeline(config, verbose=True)
    
    # Simple segmentation call
    input_file = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Data/48hcontrol.tif"
    output_dir = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/output/basic_example"
    
    try:
        # Run segmentation with progress display
        mask, metadata = pipeline.segment_vessels(
            input_path=input_file,
            output_dir=output_dir,
            sample_type='control'
        )
        
        print(f"\n✓ Basic processing completed!")
        print(f"  Vessel coverage: {metadata['vessel_coverage']:.2%}")
        print(f"  Components detected: {metadata['components']}")
        print(f"  Output directory: {output_dir}")
        
    except FileNotFoundError:
        print(f"\n⚠️  Input file not found: {input_file}")
        print("  Update the path to your TIFF file to run this example")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        
    print("\nKey takeaways:")
    print("- Default SegmentationConfig works well for most vessel images")
    print("- Set verbose=True for progress monitoring")
    print("- Always check vessel coverage in metadata for quality assessment")


def template_management_example():
    """
    Example 2: Configuration template management
    
    Shows how to save, load, and manage configuration templates for
    reproducible processing across different experiments.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Template Management")
    print("=" * 60)
    
    # Create template directory
    template_dir = Path("/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/output/templates")
    template_dir.mkdir(parents=True, exist_ok=True)
    
    # Create custom configuration
    custom_config = SegmentationConfig(
        gamma=2.0,  # More aggressive enhancement
        max_components=3,  # Allow multiple vessel networks
        sigma_smooth=1.2,  # Preserve fine details
        frangi_scales=[0.5, 1.0, 1.5, 2.0, 2.5]  # Focus on smaller vessels
    )
    
    pipeline = VesselSegmentationPipeline(custom_config)
    
    # Save configuration as template
    template_path = pipeline.save_config_template(
        name="high_sensitivity_vessels",
        output_dir=template_dir,
        description="High sensitivity configuration for detecting fine vessels and capillaries",
        sample_type="control"
    )
    print(f"✓ Saved template: {template_path}")
    
    # List available templates
    print("\nAvailable templates:")
    templates = ConfigTemplateManager.list_templates(template_dir)
    for i, template in enumerate(templates, 1):
        print(f"  {i}. {template['name']}")
        print(f"     Description: {template['description']}")
        print(f"     Sample type: {template['sample_type']}")
        print(f"     Created: {template['created_date']}")
    
    # Load template and create new pipeline
    if templates:
        loaded_pipeline = VesselSegmentationPipeline.from_template(
            template_path, verbose=True
        )
        print(f"\n✓ Loaded pipeline from template")
        print(f"  Gamma: {loaded_pipeline.config.gamma}")
        print(f"  Max components: {loaded_pipeline.config.max_components}")
        print(f"  Frangi scales: {loaded_pipeline.config.frangi_scales}")
    
    print("\nTemplate management benefits:")
    print("- Consistent parameters across experiments")
    print("- Easy sharing of optimized configurations")
    print("- Metadata tracking for reproducibility")
    print("- Version control for parameter evolution")


def parameter_preview_example():
    """
    Example 3: Parameter validation with preview mode
    
    Demonstrates how to quickly test parameters on a subset of data
    before running full processing. Essential for parameter optimization.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Parameter Preview and Validation")
    print("=" * 60)
    
    input_file = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Data/48hcontrol.tif"
    
    # Test different configurations quickly
    configs_to_test = {
        'conservative': SegmentationConfig(gamma=1.5, max_components=1),
        'standard': SegmentationConfig(gamma=1.8, max_components=3),
        'aggressive': SegmentationConfig(gamma=2.2, max_components=6)
    }
    
    print("Testing different parameter sets:")
    results = {}
    
    for config_name, config in configs_to_test.items():
        print(f"\n--- Testing {config_name} configuration ---")
        pipeline = VesselSegmentationPipeline(config, verbose=True)
        
        try:
            # Quick preview on 5 middle slices (30 second execution)
            preview_result = pipeline.preview_segmentation(
                input_path=input_file,
                sample_type='control',
                preview_slices=5
            )
            
            results[config_name] = preview_result
            print(f"  Coverage: {preview_result['coverage']:.2%}")
            print(f"  Components: {preview_result['components']}")
            print(f"  Recommendation: {preview_result['recommendation']}")
            print(f"  Advice: {preview_result['advice']}")
            
        except FileNotFoundError:
            print(f"  ⚠️  Input file not found: {input_file}")
            print("  Update the path to run this example")
            continue
        except Exception as e:
            print(f"  ✗ Preview failed: {e}")
            continue
    
    # Recommend best configuration
    if results:
        print(f"\n--- Configuration Comparison ---")
        good_configs = [name for name, result in results.items() 
                       if result['recommendation'] == 'good']
        
        if good_configs:
            print(f"✓ Suitable configurations: {', '.join(good_configs)}")
            # Select config with moderate coverage
            best_config = min(good_configs, 
                            key=lambda name: abs(results[name]['coverage'] - 0.05))
            print(f"✓ Recommended for full processing: {best_config}")
        else:
            print("⚠️  No configurations passed validation")
            print("   Consider adjusting thresholds or preprocessing parameters")
    
    print("\nPreview workflow benefits:")
    print("- Quick parameter validation (~30 seconds vs. 5-10 minutes)")
    print("- Prevents wasted time on poor parameter choices")
    print("- Enables systematic parameter optimization")
    print("- Provides quality metrics before full processing")


def batch_processing_example():
    """
    Example 4: Batch processing workflow
    
    Shows how to process multiple files efficiently with consistent
    parameters and organized output structure.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Batch Processing")
    print("=" * 60)
    
    # Setup batch processing
    data_directory = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Data"
    output_base = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/output/batch_processing"
    
    # Find all TIFF files (simulated - adjust pattern as needed)
    pattern = "*.tif"
    tiff_files = glob.glob(str(Path(data_directory) / pattern))
    
    if not tiff_files:
        print(f"No TIFF files found in {data_directory}")
        print("This is a demonstration of the batch processing workflow:")
        
        # Simulate file list for demonstration
        tiff_files = [
            "sample1_control.tif",
            "sample2_ko_rescue.tif", 
            "sample3_control.tif"
        ]
        print(f"Simulated file list: {tiff_files}")
    
    # Create optimized configuration for batch processing
    batch_config = SegmentationConfig(
        gamma=1.8,
        max_components=3,
        sigma_smooth=1.5,
        chunk_size=10  # Smaller chunks for memory efficiency
    )
    
    # Silent processing for batch jobs
    pipeline = VesselSegmentationPipeline(batch_config, verbose=False)
    
    print(f"\nProcessing {len(tiff_files)} files...")
    
    # Process each file
    batch_results = []
    for i, file_path in enumerate(tiff_files, 1):
        filename = Path(file_path).name
        
        # Determine sample type from filename (customize logic as needed)
        if "ko_rescue" in filename.lower():
            sample_type = "ko_rescue"
        else:
            sample_type = "control"
        
        # Create organized output directory
        output_dir = Path(output_base) / Path(file_path).stem
        
        print(f"[{i}/{len(tiff_files)}] Processing {filename} ({sample_type})")
        
        try:
            # Process with minimal output for batch efficiency
            mask, metadata = pipeline.segment_vessels(
                input_path=file_path,
                output_dir=output_dir,
                sample_type=sample_type,
                show_progress=False  # Silent processing
            )
            
            # Collect results for summary
            batch_results.append({
                'filename': filename,
                'sample_type': sample_type,
                'coverage': metadata['vessel_coverage'],
                'components': metadata['components'],
                'output_dir': str(output_dir),
                'status': 'success'
            })
            
            print(f"  ✓ Completed - Coverage: {metadata['vessel_coverage']:.2%}")
            
        except FileNotFoundError:
            print(f"  ⚠️  File not found (demonstration mode)")
            batch_results.append({
                'filename': filename,
                'status': 'file_not_found'
            })
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            batch_results.append({
                'filename': filename,
                'status': 'error',
                'error': str(e)
            })
    
    # Generate batch summary
    print(f"\n--- Batch Processing Summary ---")
    successful = [r for r in batch_results if r.get('status') == 'success']
    failed = [r for r in batch_results if r.get('status') != 'success']
    
    print(f"Processed: {len(successful)}/{len(batch_results)} files")
    
    if successful:
        avg_coverage = sum(r['coverage'] for r in successful) / len(successful)
        print(f"Average vessel coverage: {avg_coverage:.2%}")
        
        print("\nDetailed results:")
        for result in successful:
            print(f"  {result['filename']}: {result['coverage']:.2%} coverage, "
                  f"{result['components']} components")
    
    if failed:
        print(f"\nFailed files: {len(failed)}")
        for result in failed:
            print(f"  {result['filename']}: {result['status']}")
    
    print("\nBatch processing best practices:")
    print("- Use verbose=False for silent processing")
    print("- Organize outputs by sample name or date")
    print("- Collect metadata for quality control analysis")
    print("- Handle errors gracefully to continue processing")
    print("- Generate summary reports for experiment overview")


def interactive_workflow():
    """
    Example 5: Interactive workflow with user choices
    
    Demonstrates a complete interactive workflow where users can make
    choices about configuration, preview results, and control processing.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Interactive Workflow")
    print("=" * 60)
    
    # File selection (simplified for example)
    input_file = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Data/48hcontrol.tif"
    output_dir = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/output/interactive_example"
    template_dir = Path(output_dir) / "templates"
    
    print("Welcome to the Interactive Vessel Segmentation Workflow!")
    print(f"Input file: {Path(input_file).name}")
    
    # Step 1: Configuration selection
    print(f"\n--- Step 1: Configuration Selection ---")
    
    # Check for existing templates
    templates = ConfigTemplateManager.list_templates(template_dir)
    standard_configs = create_standard_configs()
    
    print("Available configurations:")
    print("  1. Standard configurations:")
    for i, (name, config) in enumerate(standard_configs.items(), 1):
        print(f"     {i}. {name}")
    
    if templates:
        print("  2. Saved templates:")
        for i, template in enumerate(templates, len(standard_configs) + 1):
            print(f"     {i}. {template['name']} - {template['description']}")
    
    # Simulate user choice (in real usage, use input())
    print("\nFor this demo, selecting 'control' configuration...")
    selected_config = standard_configs['control']
    
    # Step 2: Parameter preview
    print(f"\n--- Step 2: Parameter Preview ---")
    print("Running quick validation on subset of data...")
    
    pipeline = VesselSegmentationPipeline(selected_config, verbose=True)
    
    try:
        preview_result = pipeline.preview_segmentation(
            input_path=input_file,
            sample_type='control'
        )
        
        print(f"Preview results:")
        print(f"  Coverage: {preview_result['coverage']:.2%}")
        print(f"  Components: {preview_result['components']}")
        print(f"  Recommendation: {preview_result['recommendation']}")
        print(f"  Advice: {preview_result['advice']}")
        
        # Decision point
        if preview_result['recommendation'] == 'good':
            print("✓ Parameters look good! Proceeding with full processing...")
            proceed = True
        else:
            print("⚠️  Parameter adjustment may be needed.")
            print("For demo purposes, proceeding anyway...")
            proceed = True
            
    except FileNotFoundError:
        print("Input file not found - continuing with workflow demonstration")
        proceed = False
    except Exception as e:
        print(f"Preview failed: {e} - continuing with workflow demonstration")
        proceed = False
    
    # Step 3: Processing options
    print(f"\n--- Step 3: Processing Options ---")
    
    # Simulate user choices
    generate_configs = True
    save_template = False
    
    print(f"Configuration choices:")
    print(f"  Generate MATLAB configs: {generate_configs}")
    print(f"  Save as template: {save_template}")
    
    # Step 4: Main processing
    if proceed:
        print(f"\n--- Step 4: Main Processing ---")
        
        try:
            mask, metadata = pipeline.segment_vessels(
                input_path=input_file,
                output_dir=output_dir,
                sample_type='control',
                generate_configs=generate_configs
            )
            
            print(f"\n✓ Processing completed successfully!")
            print(f"Results:")
            print(f"  Vessel coverage: {metadata['vessel_coverage']:.2%}")
            print(f"  Components: {metadata['components']}")
            print(f"  Output files:")
            for key, path in metadata['output_files'].items():
                print(f"    {key}: {Path(path).name}")
            
            # Optional template saving
            if save_template:
                template_path = pipeline.save_config_template(
                    "interactive_session",
                    template_dir,
                    "Configuration from interactive session",
                    'control'
                )
                print(f"  ✓ Saved template: {template_path}")
                
        except Exception as e:
            print(f"Processing demonstration completed (file not found)")
    
    print(f"\n--- Workflow Complete ---")
    print("Interactive workflow features:")
    print("- User-guided configuration selection")
    print("- Parameter validation before full processing")
    print("- Flexible processing options")
    print("- Template management integration")
    print("- Comprehensive result reporting")


def scientific_workflow_example():
    """
    Example 6: Complete scientific workflow
    
    Demonstrates a publication-ready workflow with quality control,
    standardized processing, and comprehensive documentation.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Scientific Publication Workflow")
    print("=" * 60)
    
    # Experimental setup
    experiment_name = "vessel_analysis_2025"
    base_output = f"/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/output/{experiment_name}"
    
    print(f"Experiment: {experiment_name}")
    print("This workflow ensures reproducibility and quality control for scientific publications")
    
    # Step 1: Standardized configuration
    print(f"\n--- Step 1: Standardized Configuration ---")
    
    # Create publication-quality configuration with documentation
    publication_config = SegmentationConfig(
        gamma=1.8,              # Standard enhancement for confocal data
        max_components=1,       # Single network analysis
        sigma_smooth=1.5,       # Balanced detail/smoothness for meshing
        frangi_scales=[1.0, 1.5, 2.0, 2.5, 3.0],  # Multi-scale vessel detection
        distance_weight=0.4,    # Optimized gradient blending
        min_object_size=50      # Remove small noise artifacts
    )
    
    pipeline = VesselSegmentationPipeline(publication_config, verbose=True)
    
    # Save configuration for reproducibility
    config_dir = Path(base_output) / "configurations"
    config_path = pipeline.save_config_template(
        f"{experiment_name}_standard",
        config_dir,
        f"Standardized configuration for {experiment_name} vessel analysis",
        "publication"
    )
    print(f"✓ Saved standardized configuration: {config_path}")
    
    # Step 2: Quality control validation
    print(f"\n--- Step 2: Quality Control Validation ---")
    
    input_file = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Data/48hcontrol.tif"
    
    try:
        # Comprehensive preview with quality metrics
        qc_result = pipeline.preview_segmentation(
            input_path=input_file,
            sample_type='control',
            preview_slices=10  # More slices for better QC
        )
        
        print("Quality Control Metrics:")
        print(f"  Vessel coverage: {qc_result['coverage']:.3f} ({qc_result['coverage']:.1%})")
        print(f"  Component count: {qc_result['components']}")
        print(f"  QC status: {qc_result['recommendation']}")
        print(f"  Assessment: {qc_result['advice']}")
        
        # Quality control thresholds for publication
        qc_passed = (
            0.005 <= qc_result['coverage'] <= 0.15 and  # Reasonable coverage range
            1 <= qc_result['components'] <= 5 and       # Not too fragmented
            qc_result['recommendation'] == 'good'
        )
        
        print(f"  Publication QC: {'PASS' if qc_passed else 'REVIEW NEEDED'}")
        
    except FileNotFoundError:
        print("Input file not found - demonstrating QC workflow")
        qc_passed = True  # Continue demo
    except Exception as e:
        print(f"QC validation error: {e}")
        qc_passed = False
    
    # Step 3: Standardized processing
    if qc_passed:
        print(f"\n--- Step 3: Standardized Processing ---")
        
        # Organized output structure
        sample_output = Path(base_output) / "processed_samples" / Path(input_file).stem
        
        try:
            # Process with full documentation generation
            mask, metadata = pipeline.segment_vessels(
                input_path=input_file,
                output_dir=sample_output,
                sample_type='control',
                generate_configs=True  # Include MATLAB configs for u-shape3D
            )
            
            # Enhanced metadata for publication
            enhanced_metadata = {
                **metadata,
                'experiment_name': experiment_name,
                'processing_date': pipeline.config.__dict__,  # Full parameter record
                'qc_metrics': qc_result,
                'publication_notes': 'Processed with standardized pipeline for vessel analysis'
            }
            
            print(f"✓ Processing completed with full documentation")
            print(f"  Final coverage: {metadata['vessel_coverage']:.3f}")
            print(f"  Output location: {sample_output}")
            
        except FileNotFoundError:
            print("Processing demonstration completed (file not available)")
        except Exception as e:
            print(f"Processing error: {e}")
    
    # Step 4: Results documentation
    print(f"\n--- Step 4: Results Documentation ---")
    
    print("Generated documentation:")
    print(f"  📁 Configuration templates: {config_dir}")
    print(f"  📁 Processed samples: {Path(base_output) / 'processed_samples'}")
    print(f"  📄 MATLAB integration configs")
    print(f"  📄 Complete processing metadata")
    print(f"  📄 Quality control reports")
    
    print(f"\nPublication workflow features:")
    print("- Standardized, documented configurations")
    print("- Comprehensive quality control validation")
    print("- Organized output structure")
    print("- Complete parameter documentation for Methods section")
    print("- MATLAB integration for downstream analysis")
    print("- Reproducible processing pipeline")


def advanced_customization_example():
    """
    Example 7: Advanced customization
    
    Shows how to customize the pipeline for specific research needs,
    including parameter optimization and specialized configurations.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Advanced Customization")
    print("=" * 60)
    
    print("Advanced customization for specialized research applications")
    
    # Custom configurations for different research scenarios
    specialized_configs = {
        'capillary_analysis': SegmentationConfig(
            gamma=2.2,  # High contrast for dim capillaries
            frangi_scales=[0.5, 1.0, 1.5],  # Small vessel focus
            max_components=10,  # Allow many small segments
            sigma_smooth=1.0,  # Preserve fine details
            min_object_size=25  # Capture smaller structures
        ),
        
        'large_vessel_networks': SegmentationConfig(
            gamma=1.5,  # Gentle enhancement for bright vessels
            frangi_scales=[2.0, 3.0, 4.0, 5.0],  # Large vessel focus
            max_components=1,  # Single network expected
            sigma_smooth=2.5,  # Heavy smoothing for clean meshes
            distance_weight=0.6  # Strong gradient smoothing
        ),
        
        'pathological_samples': SegmentationConfig(
            gamma=1.8,
            frangi_scales=[1.0, 1.5, 2.0, 2.5, 3.0],
            max_components=8,  # Allow fragmented networks
            sigma_smooth=1.2,  # Moderate smoothing
            threshold_percentiles={
                'pathological': {
                    'top': 1.0,     # Very sensitive
                    'middle': 0.8,  # Very sensitive
                    'bottom': 1.2   # Standard
                }
            }
        )
    }
    
    print("Specialized configurations created:")
    for name, config in specialized_configs.items():
        print(f"  {name}:")
        print(f"    Gamma: {config.gamma}")
        print(f"    Frangi scales: {config.frangi_scales}")
        print(f"    Max components: {config.max_components}")
        print(f"    Smoothing: {config.sigma_smooth}")
    
    # Parameter optimization workflow
    print(f"\n--- Parameter Optimization Workflow ---")
    
    input_file = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Data/48hcontrol.tif"
    
    # Test multiple gamma values systematically
    gamma_values = [1.5, 1.8, 2.0, 2.2]
    print(f"Testing gamma values: {gamma_values}")
    
    optimization_results = []
    
    for gamma in gamma_values:
        test_config = SegmentationConfig(gamma=gamma)
        test_pipeline = VesselSegmentationPipeline(test_config)
        
        try:
            result = test_pipeline.preview_segmentation(
                input_path=input_file,
                sample_type='control',
                show_progress=False  # Silent for optimization
            )
            
            optimization_results.append({
                'gamma': gamma,
                'coverage': result['coverage'],
                'components': result['components'],
                'recommendation': result['recommendation']
            })
            
            print(f"  γ={gamma}: {result['coverage']:.3f} coverage, "
                  f"{result['components']} components, {result['recommendation']}")
            
        except FileNotFoundError:
            print(f"  γ={gamma}: Demo mode (file not found)")
            optimization_results.append({
                'gamma': gamma,
                'coverage': 0.05,  # Simulated
                'components': 2,
                'recommendation': 'good'
            })
        except Exception as e:
            print(f"  γ={gamma}: Error - {e}")
    
    # Find optimal parameters
    if optimization_results:
        valid_results = [r for r in optimization_results if r['recommendation'] == 'good']
        if valid_results:
            # Select gamma with coverage closest to target (5%)
            optimal = min(valid_results, key=lambda x: abs(x['coverage'] - 0.05))
            print(f"\n✓ Optimal gamma: {optimal['gamma']} "
                  f"(coverage: {optimal['coverage']:.3f})")
        else:
            print("\n⚠️  No optimal parameters found in tested range")
    
    # Custom threshold example
    print(f"\n--- Custom Regional Thresholds ---")
    
    # Example: Adjust thresholds for specific imaging conditions
    custom_thresholds = {
        'high_background': {
            'top': 5.0,     # Higher threshold for noisy top region
            'middle': 3.0,  # Moderate for middle
            'bottom': 1.0   # Sensitive for dim bottom region
        }
    }
    
    custom_config = SegmentationConfig(
        threshold_percentiles=custom_thresholds
    )
    
    print("Custom threshold configuration created:")
    print(f"  High background sample thresholds: {custom_thresholds['high_background']}")
    
    print(f"\nAdvanced customization capabilities:")
    print("- Specialized configurations for different vessel types")
    print("- Systematic parameter optimization workflows") 
    print("- Custom regional threshold definitions")
    print("- Application-specific processing pipelines")
    print("- Quality-driven parameter selection")


def main():
    """
    Run all examples in sequence to demonstrate complete pipeline capabilities.
    
    This comprehensive tutorial covers all aspects of the vessel segmentation
    pipeline from basic usage to advanced scientific workflows.
    """
    print("🔬 VESSEL SEGMENTATION PIPELINE - COMPLETE TUTORIAL")
    print("=" * 80)
    print("\nThis tutorial demonstrates all features of the VesselSegmentationPipeline")
    print("with practical examples for scientific image processing applications.")
    print("\nNote: Some examples require actual TIFF files to run completely.")
    print("The workflow and concepts are demonstrated even when files are missing.")
    
    # Run all examples
    examples = [
        basic_usage_example,
        template_management_example,
        parameter_preview_example,
        batch_processing_example,
        interactive_workflow,
        scientific_workflow_example,
        advanced_customization_example
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except KeyboardInterrupt:
            print(f"\n\nTutorial interrupted by user")
            break
        except Exception as e:
            print(f"\n\nExample {i} error: {e}")
            print("Continuing with next example...")
        
        if i < len(examples):
            print(f"\n{'':->80}")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("🎯 TUTORIAL COMPLETE")
    print("=" * 80)
    print("\nYou've seen all major features of the VesselSegmentationPipeline:")
    print("\n📋 Core Features:")
    print("  ✓ Basic vessel segmentation with quality metrics")
    print("  ✓ Configuration template management")
    print("  ✓ Parameter validation and optimization")
    print("  ✓ Batch processing workflows")
    print("  ✓ Interactive user workflows")
    print("  ✓ Scientific publication workflows")
    print("  ✓ Advanced customization options")
    
    print("\n🔧 Key Workflow Patterns:")
    print("  1. Preview → Validate → Process → Document")
    print("  2. Template → Customize → Save → Reuse")
    print("  3. Batch → Quality Control → Analysis")
    print("  4. Optimize → Standardize → Publish")
    
    print("\n📖 Next Steps:")
    print("  • Adapt file paths to your data")
    print("  • Create templates for your specific applications")
    print("  • Integrate with your analysis workflows")
    print("  • Use generated configs with u-shape3D for meshing")
    
    print("\n💡 Best Practices:")
    print("  • Always run preview before full processing")
    print("  • Save configurations as templates for reproducibility")
    print("  • Monitor vessel coverage for quality control")
    print("  • Document parameters for publication methods")
    
    print(f"\nHappy vessel segmentation! 🧬")


if __name__ == "__main__":
    main()
