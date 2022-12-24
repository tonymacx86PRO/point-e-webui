# Imports
import io
import platform
import sys
import gradio as gr
import os
import torch
from tqdm.auto import tqdm
import PIL
import random

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

# Variables
VERSION = "0.1.4"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_name = ''
base_model = None
base_diffusion = None
upsampler_model = None
upsampler_diffusion = None
samples = None
sampler = None

cwd_path = os.getcwd()
gd_scale = 3.0
text2pc_path = f'{cwd_path}\\outputs\\text2pc\\'
image2pc_path = f'{cwd_path}\\outputs\\image2pc\\'


# Print information
def info(msg):
    print(f"[INFO] {msg}")

# Print error
def error(msg):
    print(f"[ERROR] {msg}")

# Print warning
def warn(msg):
    print(f"[WARNING] {msg}")

# Load model by name
def load_model(model_name):
    global device
    global base_name
    global base_model
    global base_diffusion
    global upsampler_model
    global upsampler_diffusion
    try:
        info(f'Loading model {model_name}')
        base_name = model_name # Image-based: base40M, base300M, base1B; Text-based: base40M-textvec
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

        base_model.load_state_dict(load_checkpoint(base_name, device))

        upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    except:
        error(f'Failed to load {model_name}')
        base_name = ''

# Create sampler by type 0: TEXT SAMPLER; 1: IMAGE SAMPLER
def create_sampler(type, gd_scale):
    global device
    global base_model
    global upsampler_model
    global base_diffusion
    global upsampler_diffusion
    global sampler
    if type == 0:
        info(f'Text sampler created')
        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[gd_scale, 0.0],
            model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
        )
    elif type == 1:
        info(f'Image sampler created')
        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[gd_scale, gd_scale],
        )

def buffer_plot_and_get(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return PIL.Image.open(buf)

# Text to point cloud
def text2pc(prompt):
    global samples
    global sampler
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x

# Image to point cloud
def image2pc(image):
    global samples
    global sampler
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[image]))):
        samples = x

# Get output image
def output_figure():
    global sampler
    global samples
    pc = sampler.output_to_point_clouds(samples)[0]
    return plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

# Resize to 256x256 and crop
def prepare_img(img):
    w, h = img.size
    if w > h:
        img = img.crop((w - h) / 2, 0, w - (w - h) / 2, h)
    else:
        img = img.crop((0, (h - w) / 2, w, h - (h - w) / 2))
    
    img = img.resize((256, 256))
    
    return img

# Button "Generate" text to 3D click
def text2model(text, model_type):
    global gd_scale
    if len(text) == 0:
        return None
    else:
        load_model(model_type)
        create_sampler(0, gd_scale)
        text2pc(text)
        fig = output_figure()
        image = buffer_plot_and_get(fig)
        image.save(text2pc_path + text + "-" + str(random.randint(1, 9999)) + ".png")
        return image

# Button "Generate" image to 3D click
def image2model(image, model_type):
    global gd_scale
    if image is None:
        return None
    else:
        load_model(model_type)    
        create_sampler(1, gd_scale)
        prep_img = prepare_img(image)
        image2pc(prep_img)
        fig = output_figure()
        image = buffer_plot_and_get(fig)
        image.save(image2pc_path + str(random.randint(1, 9999)) + ".png")
        return image

# Update guidance scale
def gd_scale_changed(i):
    global gd_scale
    gd_scale = float(i)

# Entry Point
def main():
    global device
    with gr.Blocks() as gui:
        gr.Markdown("# POINT-E WebUI by @tonyx86")

        with gr.Tab("Text to 3D"):
            with gr.Group():
                input_prompt = gr.Textbox(label='Prompt')
                model_type_t = gr.Dropdown(label='Model', choices=['base40M-textvec'], interactive=True, value='base40M-textvec')
                gd_scale_t = gr.Slider(0.0, 50.0, 3.0, label='Guidance scale')
                gd_scale_t.change(gd_scale_changed, [gd_scale_t])
            with gr.Group():
                output_image_t = gr.Image(label='Output',interactive=False)
            text2model_btn = gr.Button(value="Generate")
            text2model_btn.click(text2model, [input_prompt, model_type_t], [output_image_t])
        
        with gr.Tab("Image to 3D"):
            with gr.Group():
                input_image = gr.Image(label='Input image')
                model_type_i = gr.Dropdown(label='Model', choices=['base40M', 'base300M', 'base1B'], interactive=True, value='base40M')
                gd_scale_i = gr.Slider(0.0, 50.0, 3.0, label='Guidance scale')
                gd_scale_i.change(gd_scale_changed, [gd_scale_i])
            with gr.Group():
                output_image_i = gr.Image(label='Output',interactive=False)
            image2model_btn = gr.Button(value="Generate")
            image2model_btn.click(image2model, [input_image, model_type_i], [output_image_i])
        
        with gr.Tab("Information"):
            gr.Label(VERSION, label='WebUI version')
            gr.Label(sys.version, label='Python version')
            gr.Label(platform.platform(), label='Platform information')
            gr.Label(torch.cuda.get_device_name(device), label='Current pytorch device')
            gr.Label(cwd_path, label='Current directory')

        gr.HTML('<a href="https://www.donationalerts.com/r/tonyonyxyt">Donations</a>')
    gui.launch()

if __name__ == '__main__':
    main()