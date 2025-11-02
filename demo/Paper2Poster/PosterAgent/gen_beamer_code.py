import re
import json
import os
from typing import List, Dict, Any

def sanitize_for_latex(name):
    """Convert any character that is not alphanumeric into underscore for LaTeX compatibility."""
    return re.sub(r'[^0-9a-zA-Z_]+', '_', name)

def initialize_beamer_document(width_cm=120, height_cm=90, theme="default"):
    """
    Initialize a Beamer document with specified dimensions and theme.
    
    Args:
        width_cm: Width in centimeters (default 120cm for poster)
        height_cm: Height in centimeters (default 90cm for poster) 
        theme: Beamer theme name (default, Madrid, Warsaw, etc.)
    """
    code = f'''\\documentclass[aspectratio=169]{{beamer}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{graphicx}}
\\usepackage{{tikz}}
\\usepackage{{xcolor}}
\\usepackage{{geometry}}
\\usepackage{{multicol}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{adjustbox}}

% Set page dimensions for poster
\\geometry{{paperwidth={width_cm}cm, paperheight={height_cm}cm, margin=1cm}}

% Beamer theme
\\usetheme{{{theme}}}
\\usecolortheme{{default}}

% Custom colors
\\definecolor{{titlecolor}}{{RGB}}{{47, 85, 151}}
\\definecolor{{textcolor}}{{RGB}}{{0, 0, 0}}
\\definecolor{{bgcolor}}{{RGB}}{{255, 255, 255}}

% Remove navigation symbols
\\setbeamertemplate{{navigation symbols}}{{}}

% Custom title page
\\setbeamertemplate{{title page}}{{
    \\begin{{center}}
        \\vspace{{1cm}}
        {{\\color{{titlecolor}}\\Huge\\textbf{{\\inserttitle}}}}
        \\vspace{{0.5cm}}
        \\Large{{\\insertauthor}}
        \\vspace{{0.3cm}}
        \\normalsize{{\\insertinstitute}}
    \\end{{center}}
}}

% Custom frame title
\\setbeamertemplate{{frametitle}}{{
    \\vspace{{0.5cm}}
    \\begin{{flushleft}}
        {{\\color{{titlecolor}}\\Large\\textbf{{\\insertframetitle}}}}
    \\end{{flushleft}}
    \\vspace{{0.3cm}}
}}

\\begin{{document}}

% Title frame
\\title{{POSTER_TITLE_PLACEHOLDER}}
\\author{{POSTER_AUTHOR_PLACEHOLDER}}
\\institute{{POSTER_INSTITUTE_PLACEHOLDER}}
\\date{{\\today}}

\\begin{{frame}}[plain]
    \\titlepage
\\end{{frame}}

'''
    return code

def generate_beamer_section_code(section_data: Dict[str, Any], section_index: int):
    """
    兼容 Paper2Poster bullet JSON:
    - section_data 包含 title_blocks / textbox1_blocks / textbox2_blocks
    - 每个 *_blocks 是 list[ {bullet: bool, runs: [{text: str, ...}], ...} ]
    """
    def blocks_to_lines(blocks):
        """把 blocks 转成 list[str]，并标注是否 bullet"""
        lines = []
        for blk in blocks or []:
            text = " ".join([r.get("text","") for r in blk.get("runs", [])]).strip()
            if not text:
                continue
            lines.append({
                "text": text,
                "bullet": bool(blk.get("bullet", False))
            })
        return lines

    # Frame title 优先用 title_blocks 的文本，否则用 title_str，否则 Untitled
    if isinstance(section_data.get("title_blocks"), list) and section_data["title_blocks"]:
        frame_title = " ".join([r.get("text","") for r in section_data["title_blocks"][0].get("runs", [])]).strip()
    else:
        frame_title = section_data.get("title_str") or "Untitled"

    frame_title = frame_title.replace("{","\\{").replace("}","\\}")  # 简单转义以防标题含花括号

    code = f"\n% ===== Section {section_index} =====\n"
    code += f"\\begin{{frame}}[t]{{{frame_title}}}\n"
    code += "  \\vspace{-0.5cm}\n"

    for key in ["textbox1_blocks", "textbox2_blocks"]:
        lines = blocks_to_lines(section_data.get(key, []))
        if not lines:
            continue

        # 如果全是 bullet，就合并成一个 itemize；否则分别处理
        if all(l["bullet"] for l in lines):
            code += "  \\begin{itemize}\n"
            for l in lines:
                code += f"    \\item {l['text']}\n"
            code += "  \\end{itemize}\n"
        else:
            for l in lines:
                if l["bullet"]:
                    code += f"  \\begin{{itemize}}\\item {l['text']}\\end{{itemize}}\n"
                else:
                    code += f"  {l['text']}\\\\\n"

    code += "\\end{frame}\n\n"
    return code



def generate_beamer_figure_code(figure_data: Dict[str, Any], figure_index: int):
    """
    Generate Beamer code for including figures.
    
    Args:
        figure_data: Dictionary containing figure information
        figure_index: Index of the figure
    """
    figure_name = sanitize_for_latex(figure_data.get('figure_name', f'figure_{figure_index}'))
    figure_path = figure_data.get('figure_path', '')
    
    # Convert inches to centimeters (1 inch = 2.54 cm)
    width_cm = figure_data.get('width', 10) * 2.54
    height_cm = figure_data.get('height', 8) * 2.54
    
    code = f'''
% Figure: {figure_name}
\\begin{{frame}}[t]{{{figure_data.get('title', 'Figure')}}}
    \\vspace{{-0.5cm}}
    \\begin{{center}}
        \\includegraphics[width={width_cm:.2f}cm, height={height_cm:.2f}cm]{{{figure_path}}}
    \\end{{center}}
    \\vspace{{0.3cm}}
    \\begin{{center}}
        \\small{{\\textbf{{{figure_data.get('caption', 'Figure Caption')}}}}}
    \\end{{center}}
\\end{{frame}}

'''
    return code

def generate_beamer_poster_code(
    sections: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
    poster_info: Dict[str, str],
    width_cm: float = 120,
    height_cm: float = 90,
    theme: str = "default",
    output_path: str = "poster.tex"
):
    """
    Generate complete Beamer poster code.
    
    Args:
        sections: List of section dictionaries
        figures: List of figure dictionaries  
        poster_info: Dictionary with title, author, institute
        width_cm: Poster width in centimeters
        height_cm: Poster height in centimeters
        theme: Beamer theme name
        output_path: Output .tex file path
    """
    code = initialize_beamer_document(width_cm, height_cm, theme)
    
    # Replace placeholders with actual content
    code = code.replace('POSTER_TITLE_PLACEHOLDER', poster_info.get('title', 'Poster Title'))
    code = code.replace('POSTER_AUTHOR_PLACEHOLDER', poster_info.get('author', 'Author Name'))
    code = code.replace('POSTER_INSTITUTE_PLACEHOLDER', poster_info.get('institute', 'Institute Name'))
    
    # Add sections
    for i, section in enumerate(sections):
        code += generate_beamer_section_code(section, i)
    
    # Add figures
    for i, figure in enumerate(figures):
        code += generate_beamer_figure_code(figure, i)
    
    # Close document
    code += '''
\\end{document}
'''
    
    return code

def save_beamer_code(code: str, output_path: str):
    """Save Beamer code to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code)

def compile_beamer_to_pdf(tex_path: str, output_dir: str = "."):
    """
    Compile Beamer .tex file to PDF using pdflatex.
    
    Args:
        tex_path: Path to .tex file
        output_dir: Output directory for PDF
    """
    import subprocess
    
    try:
        # Run pdflatex twice for proper cross-references
        result1 = subprocess.run(
            ['pdflatex', '-output-directory', output_dir, tex_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        result2 = subprocess.run(
            ['pdflatex', '-output-directory', output_dir, tex_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result1.returncode == 0 and result2.returncode == 0:
            print(f"Successfully compiled {tex_path} to PDF")
            return True
        else:
            print(f"Error compiling {tex_path}:")
            print(result1.stderr)
            print(result2.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Timeout while compiling {tex_path}")
        return False
    except Exception as e:
        print(f"Error compiling {tex_path}: {e}")
        return False

# Example usage functions
def convert_pptx_layout_to_beamer(pptx_layout_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert PowerPoint layout data to Beamer-compatible format.
    
    Args:
        pptx_layout_data: Layout data from PowerPoint generation
    """
    beamer_data = {
        'sections': [],
        'figures': [],
        'poster_info': {
            'title': 'Default Title',
            'author': 'Default Author', 
            'institute': 'Default Institute'
        }
    }
    
    # Convert text arrangements to sections
    if 'text_arrangement' in pptx_layout_data:
        for i, text_item in enumerate(pptx_layout_data['text_arrangement']):
            section = {
                'section_name': text_item.get('textbox_name', f'section_{i}'),
                'title': text_item.get('title', f'Section {i+1}'),
                'content': text_item.get('content', 'Content placeholder')
            }
            beamer_data['sections'].append(section)
    
    # Convert figure arrangements to figures
    if 'figure_arrangement' in pptx_layout_data:
        for i, figure_item in enumerate(pptx_layout_data['figure_arrangement']):
            figure = {
                'figure_name': figure_item.get('figure_name', f'figure_{i}'),
                'figure_path': figure_item.get('figure_path', ''),
                'width': figure_item.get('width', 10),
                'height': figure_item.get('height', 8),
                'title': figure_item.get('title', f'Figure {i+1}'),
                'caption': figure_item.get('caption', 'Figure caption')
            }
            beamer_data['figures'].append(figure)
    
    return beamer_data

