import os
import json
import time
from dotenv import load_dotenv
from jinja2 import Environment, StrictUndefined

from utils.src.utils import get_json_from_response, account_token, html_to_png
from utils.config_utils import load_poster_yaml_config

from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.types import ModelPlatformType, ModelType

load_dotenv()

def gen_beamer_poster_direct(
    paper_text: str,
    poster_width_cm: float = 120,
    poster_height_cm: float = 90,
    beamer_theme: str = "default",
    output_dir: str = "output",
    model_name: str = "4o"
):
    """
    Generate Beamer poster directly from paper text using LLM.
    
    Args:
        paper_text: Extracted text from the paper
        poster_width_cm: Poster width in centimeters
        poster_height_cm: Poster height in centimeters  
        beamer_theme: Beamer theme name
        output_dir: Output directory
        model_name: Model name for generation
    """
    start_time = time.time()
    total_input_token, total_output_token = 0, 0
    
    # Load configuration
    config_path = "utils/prompt_templates/LLM_gen_Beamer.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create model and agent
    actor_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O,
        model_config_dict=ChatGPTConfig().as_dict(),
    )
    
    actor_agent = ChatAgent(
        system_message=config['system_prompt'],
        model=actor_model,
        message_window_size=None
    )
    
    # Prepare template arguments
    jinja_args = {
        'document_markdown': paper_text,
        'poster_width_cm': poster_width_cm,
        'poster_height_cm': poster_height_cm,
        'beamer_theme': beamer_theme,
        'aspect_ratio': "169",
        'title_color': "[47, 85, 151]",
        'text_color': "[0, 0, 0]"
    }
    
    # Render template
    jinja_env = Environment(undefined=StrictUndefined)
    template = jinja_env.from_string(config["template"])
    prompt = template.render(**jinja_args)
    
    # Generate Beamer code
    actor_agent.reset()
    response = actor_agent.step(prompt)
    input_token, output_token = account_token(response)
    total_input_token += input_token
    total_output_token += output_token
    
    # Extract LaTeX code
    result_json = get_json_from_response(response.msgs[0].content)
    latex_str = result_json['LATEX']
    
    # Save LaTeX file
    os.makedirs(output_dir, exist_ok=True)
    tex_path = os.path.join(output_dir, 'poster.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    
    # Compile to PDF
    print("Compiling LaTeX to PDF...")
    success = compile_beamer_to_pdf(tex_path, output_dir)
    
    if success:
        print(f"✅ Beamer poster generated successfully: {tex_path}")
    else:
        print("❌ Failed to compile LaTeX to PDF")
    
    # Save log
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    log = {
        'input_token': total_input_token,
        'output_token': total_output_token,
        'time_taken': elapsed_time,
        'output_format': 'beamer',
        'beamer_theme': beamer_theme
    }
    
    with open(os.path.join(output_dir, 'log.json'), 'w') as f:
        json.dump(log, f, indent=4)
    
    return tex_path, success

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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Beamer poster directly from paper')
    parser.add_argument('--paper_path', required=True, help='Path to paper PDF')
    parser.add_argument('--output_dir', default='beamer_output', help='Output directory')
    parser.add_argument('--poster_width_cm', type=float, default=120, help='Poster width in cm')
    parser.add_argument('--poster_height_cm', type=float, default=90, help='Poster height in cm')
    parser.add_argument('--beamer_theme', default='default', help='Beamer theme')
    parser.add_argument('--model_name', default='4o', help='Model name')
    
    args = parser.parse_args()
    
    # Extract text from paper (you'll need to implement this)
    # For now, using placeholder text
    paper_text = "This is placeholder text. In practice, you would extract text from the PDF."
    
    # Generate Beamer poster
    tex_path, success = gen_beamer_poster_direct(
        paper_text=paper_text,
        poster_width_cm=args.poster_width_cm,
        poster_height_cm=args.poster_height_cm,
        beamer_theme=args.beamer_theme,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    if success:
        print(f"Beamer poster generated at: {tex_path}")
    else:
        print("Failed to generate Beamer poster")

