from dotenv import load_dotenv
# from utils.src.utils import get_json_from_response
from utils.src.model_utils import parse_pdf
import json
import random
import json_repair

from camel.models import ModelFactory
from camel.agents import ChatAgent
from tenacity import retry, stop_after_attempt
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from pathlib import Path

import PIL

from marker.models import create_model_dict

from utils.wei_utils import *

from utils.pptx_utils import *
from utils.critic_utils import *
import torch
from jinja2 import Template
import re
import argparse
import os
load_dotenv()
IMAGE_RESOLUTION_SCALE = 5.0

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
def account_token(response):
    input_token = response.info['usage']['prompt_tokens']
    output_token = response.info['usage']['completion_tokens']

    return input_token, output_token
def get_json_from_response(raw_response: str):
    response = raw_response.strip()
    l, r = response.rfind("```json"), response.rfind("```")
    try:
        if l == -1 or r == -1:
            response = json_repair.loads(response)
        else:
            response = json_repair.loads(response[l + 7 : r].strip())
        return response
    except Exception as e:
        raise RuntimeError("Failed to parse JSON from response", e)
@retry(stop=stop_after_attempt(5))
def parse_raw(args, actor_config, version=2):
    raw_source = args.poster_path
    markdown_clean_pattern = re.compile(r"<!--[\s\S]*?-->")
    print(f'\nParsing raw content from {raw_source}...\n')
    try:
        raw_result = doc_converter.convert(raw_source)
        print('✅ PDF converted to document format.')
        raw_markdown = raw_result.document.export_to_markdown()
        text_content = markdown_clean_pattern.sub("", raw_markdown)
        print(f'Extracted {len(text_content)} characters from the document.')
        if len(text_content) < 500:
            print('\nParsing with docling failed, using marker instead\n')
            parser_model = create_model_dict(device='cuda', dtype=torch.float16)
            text_content, rendered = parse_pdf(raw_source, model_lst=parser_model, save_file=False)
    except Exception as e:
        print(f'❌ PDF parsing failed: {e}')
        raise e

    # Load prompt template safely
    try:
        if version == 1:
            template = Template(open("utils/prompts/gen_poster_raw_content.txt").read())
        elif version == 2:
            print('Using v2 prompt template')
            template = Template(open("utils/prompts/gen_poster_raw_content_v2.txt").read())
        else:
            raise ValueError("Invalid version number.")
    except Exception as e:
        print(f'❌ Failed to load prompt template: {e}')
        raise e

    # Initialize actor model
    try:
        if args.model_name_t.startswith('vllm_qwen'):
            actor_model = ModelFactory.create(
                model_platform=actor_config['model_platform'],
                model_type=actor_config['model_type'],
                model_config_dict=actor_config['model_config'],
                url=actor_config['url'],
            )
        else:
            actor_model = ModelFactory.create(
                model_platform=actor_config['model_platform'],
                model_type=actor_config['model_type'],
                model_config_dict=actor_config['model_config'],
            )

        actor_sys_msg = 'You are the author of the paper, and you will create a poster for the paper.'
        actor_agent = ChatAgent(
            system_message=actor_sys_msg,
            model=actor_model,
            message_window_size=10,
            token_limit=actor_config.get('token_limit', None)
        )
    except Exception as e:
        print(f'❌ Failed to initialize actor model: {e}')
        raise e

    # === main conversation loop ===
    max_retry = 5
    content_json = {}
    for attempt in range(max_retry):
        try:
            print(f'\n🌀 Generating poster content... (Attempt {attempt+1}/{max_retry})')
            prompt = template.render(markdown_document=text_content)
            actor_agent.reset()
            response = actor_agent.step(prompt)
            input_token, output_token = account_token(response)

            content_json = get_json_from_response(response.msgs[0].content)
            if len(content_json) > 0:
                print('✅ Successfully parsed JSON content.')
                break
            else:
                print('⚠️ Empty JSON response, retrying...')
                if args.model_name_t.startswith('vllm_qwen'):
                    text_content = text_content[:80000]
        except Exception as e:
            print(f'❌ Error in actor_agent loop: {e}')
        # short delay could be added if needed
    else:
        raise RuntimeError("Failed to get valid JSON content after multiple attempts.")

    # Trim sections if too many
    if len(content_json.get('sections', [])) > 9:
        selected_sections = (
            content_json['sections'][:2]
            + random.sample(content_json['sections'][2:-2], 5)
            + content_json['sections'][-2:]
        )
        content_json['sections'] = selected_sections

    has_title = False

    for section in content_json['sections']:
        if type(section) != dict or not 'title' in section or not 'content' in section:
            print(f"Ouch! The response is invalid, the LLM is not following the format :(")
            print('Trying again...')
            raise
    #     if 'title' in section['title'].lower():
    #         has_title = True

    # if not has_title:
    #     print('Ouch! The response is invalid, the LLM is not following the format :(')
    #     raise

    os.makedirs('contents', exist_ok=True)
    json.dump(content_json, open(f'contents/<{args.model_name_t}_{args.model_name_v}>_{args.poster_name}_raw_content.json', 'w'), indent=4)
    return input_token, output_token, raw_result


def gen_image_and_table(args, conv_res):
    input_token, output_token = 0, 0
    raw_source = args.poster_path

    output_dir = Path(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}')

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = args.poster_name

    # Save page images
    for page_no, page in conv_res.document.pages.items():
        page_no = page.page_no
        page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

    # Save markdown with embedded pictures
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Save markdown with externally referenced pictures
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    # Save HTML with externally referenced pictures
    html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
    conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

    tables = {}

    table_index = 1
    for table in conv_res.document.tables:
        caption = table.caption_text(conv_res.document)
        if len(caption) > 0:
            table_img_path = f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}/{args.poster_name}-table-{table_index}.png'
            table_img = PIL.Image.open(table_img_path)
            tables[str(table_index)] = {
                'caption': caption,
                'table_path': table_img_path,
                'width': table_img.width,
                'height': table_img.height,
                'figure_size': table_img.width * table_img.height,
                'figure_aspect': table_img.width / table_img.height,
            }

        table_index += 1

    images = {}
    image_index = 1
    for image in conv_res.document.pictures:
        caption = image.caption_text(conv_res.document)
        if len(caption) > 0:
            image_img_path = f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}/{args.poster_name}-picture-{image_index}.png'
            image_img = PIL.Image.open(image_img_path)
            images[str(image_index)] = {
                'caption': caption,
                'image_path': image_img_path,
                'width': image_img.width,
                'height': image_img.height,
                'figure_size': image_img.width * image_img.height,
                'figure_aspect': image_img.width / image_img.height,
            }
        image_index += 1

    json.dump(images, open(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}_images.json', 'w'), indent=4)
    json.dump(tables, open(f'<{args.model_name_t}_{args.model_name_v}>_images_and_tables/{args.poster_name}_tables.json', 'w'), indent=4)

    return input_token, output_token, images, tables

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poster_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='4o')
    parser.add_argument('--poster_path', type=str, required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    agent_config = get_agent_config(args.model_name)

    if args.poster_name is None:
        args.poster_name = args.poster_path.split('/')[-1].replace('.pdf', '').replace(' ', '_')

    # Parse raw content
    input_token, output_token = parse_raw(args, agent_config)

    # Generate images and tables
    _, _ = gen_image_and_table(args)

    print(f'Token consumption: {input_token} -> {output_token}')
