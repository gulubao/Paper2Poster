import os
import json
import argparse
from typing import Dict, Any, List

# Import existing modules
from PosterAgent.gen_beamer_code import (
    generate_beamer_poster_code,
    convert_pptx_layout_to_beamer,
    save_beamer_code,
    compile_beamer_to_pdf
)
from PosterAgent.gen_pptx_code import generate_poster_code
from utils.wei_utils import run_code
from utils.theme_utils import get_default_theme, create_theme_with_alignment

def generate_beamer_poster(
    panel_arrangement_inches: List[Dict[str, Any]],
    text_arrangement_inches: List[Dict[str, Any]], 
    figure_arrangement_inches: List[Dict[str, Any]],
    bullet_content: List[Dict[str, Any]],
    poster_info: Dict[str, str],
    args,
    width_cm: float = 120,
    height_cm: float = 90,
    theme: str = "default"
):
    """
    Generate Beamer poster instead of PowerPoint.
    
    Args:
        panel_arrangement_inches: Panel layout data
        text_arrangement_inches: Text layout data
        figure_arrangement_inches: Figure layout data
        bullet_content: Content for text boxes
        poster_info: Poster metadata (title, author, institute)
        args: Command line arguments
        width_cm: Poster width in centimeters
        height_cm: Poster height in centimeters
        theme: Beamer theme name
    """
    print("\n🎯 Generating Beamer poster code...", flush=True)
    
    # Convert layout data to Beamer format
    beamer_data = convert_pptx_layout_to_beamer({
        'text_arrangement': text_arrangement_inches,
        'figure_arrangement': figure_arrangement_inches
    })
    
    # Update poster info
    beamer_data['poster_info'].update(poster_info)
    
    # Generate Beamer code
    beamer_code = generate_beamer_poster_code(
        sections=beamer_data['sections'],
        figures=beamer_data['figures'],
        poster_info=beamer_data['poster_info'],
        width_cm=width_cm,
        height_cm=height_cm,
        theme=theme
    )
    
    # Save Beamer code
    tex_path = f'{args.tmp_dir}/poster.tex'
    save_beamer_code(beamer_code, tex_path)
    
    # Compile to PDF
    print("\n📄 Compiling Beamer to PDF...", flush=True)
    success = compile_beamer_to_pdf(tex_path, args.tmp_dir)
    
    if not success:
        raise RuntimeError('Error in compiling Beamer to PDF')
    
    print(f"✅ Beamer poster generated successfully: {tex_path}")
    return tex_path

def modify_new_pipeline_for_beamer(args):
    """
    Modified version of new_pipeline.py to support Beamer output.
    This function replaces the PowerPoint generation part with Beamer generation.
    """
    # Import the original pipeline components
    from PosterAgent.new_pipeline import (
        parse_paper_content,
        gen_outline_layout_parallel, 
        gen_poster_content,
        deoverflow_parallel,
        apply_theme
    )
    
    # ... (keep all the existing pipeline steps until poster generation)
    
    # At the poster generation step, replace PowerPoint with Beamer:
    
    # === Beamer Poster Generation ===
    print("\n🎯 Generating Beamer poster...", flush=True)
    
    # Extract poster information from content
    poster_info = {
        'title': 'Research Poster Title',  # Extract from paper content
        'author': 'Author Name',           # Extract from paper content  
        'institute': 'Institute Name'      # Extract from paper content
    }
    
    # Convert inches to centimeters (1 inch = 2.54 cm)
    width_cm = args.poster_width_inches * 2.54
    height_cm = args.poster_height_inches * 2.54
    
    # Generate Beamer poster
    tex_path = generate_beamer_poster(
        panel_arrangement_inches=panel_arrangement_inches,
        text_arrangement_inches=text_arrangement_inches,
        figure_arrangement_inches=figure_arrangement_inches,
        bullet_content=bullet_content,
        poster_info=poster_info,
        args=args,
        width_cm=width_cm,
        height_cm=height_cm,
        theme=getattr(args, 'beamer_theme', 'default')
    )
    
    # Copy output to final directory
    output_dir = f'<{args.model_name_t}_{args.model_name_v}>_generated_posters/{args.poster_path.replace("paper.pdf", "")}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy generated files
    import shutil
    shutil.copy(tex_path, f'{output_dir}/poster.tex')
    shutil.copy(f'{args.tmp_dir}/poster.pdf', f'{output_dir}/poster.pdf')
    
    print(f"✅ Beamer poster saved to: {output_dir}")
    return output_dir

def add_beamer_arguments(parser):
    """Add Beamer-specific command line arguments."""
    parser.add_argument(
        '--output_format', 
        choices=['pptx', 'beamer'], 
        default='pptx',
        help='Output format: pptx (PowerPoint) or beamer (LaTeX)'
    )
    parser.add_argument(
        '--beamer_theme',
        default='default',
        help='Beamer theme name (default, Madrid, Warsaw, etc.)'
    )
    parser.add_argument(
        '--beamer_width_cm',
        type=float,
        default=120,
        help='Beamer poster width in centimeters'
    )
    parser.add_argument(
        '--beamer_height_cm', 
        type=float,
        default=90,
        help='Beamer poster height in centimeters'
    )
    return parser

# Example integration with existing pipeline
def integrate_beamer_with_existing_pipeline():
    """
    Example of how to integrate Beamer generation with the existing pipeline.
    """
    # This would be added to the main pipeline function
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Beamer poster from paper')
    parser = add_beamer_arguments(parser)
    
    # Add other existing arguments...
    
    args = parser.parse_args()
    
    if args.output_format == 'beamer':
        modify_new_pipeline_for_beamer(args)
    else:
        # Use original PowerPoint pipeline
        pass

