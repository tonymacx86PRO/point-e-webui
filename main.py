# Import standalone libraries
import json
import platform
import sys
import random
import os

# Gradio(UI), Torch (ML Library), Progress bar library (tqdm), Pillow (Library for images)
import gradio as gr
import torch
from tqdm.auto import tqdm
import PIL

# My modules
from modules import log

# POINT-E Library
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh

# Library for plots
import plotly.graph_objs as go

# 3D Library
import trimesh

# Diffusers library
from diffusers import StableDiffusionPipeline

# Variables

# CONSTANTS
VERSION = "0.2.3"
CWD = os.getcwd()

# Pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# All models variables
base_name = ''
base_model = None
base_diffusion = None
upsampler_model = None
upsampler_diffusion = None
sdf_model = None
sdf_name = 'sdf'
samples = None
sampler = None

# Stable diffusion pipe
sd_pipe = None

# Config dictionary
cfg = {
    "PublicURL" : False
}

# Settings in the POINT-E
gd_scale = 3.0 # Guidance scale
grid_size = 32 # Grid size of the 3D model
text2pc_path = f'{CWD}\\outputs\\text2pc\\' # text2pc path
image2pc_path = f'{CWD}\\outputs\\image2pc\\' # image2pc path
sd2pc_path = f'{CWD}\\outputs\\sd2pc\\' # sd2pc path

# Before main()

log.info(f'Starting POINT-E WebUI by @tonyx86, version: {VERSION}')

# Load config
if os.path.exists("config.json"):
    with open("config.json", "r") as cfgfile:
        data = json.load(cfgfile)
        cfg = data
        log.info('Loaded config file')
else:
    with open("config.json", "w") as cfgfile:
        cfgfile.write(json.dumps(cfg))
        log.info("Saved a new config file")

# Load model by name
def base_load(model_name, preload = False):
    global device
    global base_name
    global base_model
    global base_diffusion
    try:
        if model_name != base_name:
            if preload != True:
                log.info(f'Loading base model {model_name}')
            else:
                log.info('Preloading default base model')
            # Base model preparation
            base_name = model_name # Image-based: base40M, base300M, base1B; Text-based: base40M-textvec
            base_model = model_from_config(MODEL_CONFIGS[base_name], device)
            base_model.eval()
            base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

            # Load all models
            base_model.load_state_dict(load_checkpoint(base_name, device))

    except:
        log.error(f'Failed to load base model {model_name}')
        sys.exit(1)


def upsamplesdf_model_load():
    global device
    global upsampler_model
    global upsampler_diffusion
    global sdf_model
    global sdf_name

    try:
        log.info("Prepared upsampler model")
        # Upsampler model preparation
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

        log.info("Prepared SDF model")
        # SDF model preparation
        sdf_model = model_from_config(MODEL_CONFIGS[sdf_name], device)
        sdf_model.eval()

        # Load this up
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))
        log.info("Loaded upsampler model")
        sdf_model.load_state_dict(load_checkpoint(sdf_name, device))
        log.info("Loaded SDF model")
    except:
        log.error('Failed to load UPSAMPLER and SDF model')
        sys.exit(1)

def sd_model_load():
    global sd_pipe
    global device
    sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
    sd_pipe.to(device)
    log.info("Loaded stable diffusion 1.5")


# Create sampler by type 0: TEXT SAMPLER; 1: IMAGE SAMPLER
def create_sampler(type, gd_scale):
    global device
    global base_model
    global upsampler_model
    global base_diffusion
    global upsampler_diffusion
    global sampler
    if type == 0:
        log.info(f'Text sampler created')
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
        log.info(f'Image sampler created')
        sampler = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[gd_scale, gd_scale],
        )

# Sampling text
def text2samples(prompt):
    global samples
    global sampler
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt])), desc="Sampling text"):
        samples = x

# Sampling image
def image2samples(image):
    global samples
    global sampler
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[image])), desc="Sampling image"):
        samples = x

# Get plot from point cloud
def pc2plot(pc):
    log.info("Making plot from point cloud")
    return go.Figure(
        data=[
            go.Scatter3d(
                x=pc.coords[:,0], y=pc.coords[:,1], z=pc.coords[:,2], 
                mode='markers',
                marker=dict(
                  size=2,
                  color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(pc.channels["R"], pc.channels["G"], pc.channels["B"])],
              )
            )
        ],
        layout=dict(
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
        ),
    )

# Save *.ply mesh file from point cloud
def save_ply(pc, file_name, grid_size):
    global sdf_model
    log.info(f"Saving *.ply mesh with {grid_size} grid size.")
    # Produce a mesh (with vertex colors)
    mesh = marching_cubes_mesh(
        pc=pc,
        model=sdf_model,
        batch_size=4096,
        grid_size=int(grid_size), # increase to 128 for resolution used in evals
        progress=True,
    )

    # Write the mesh to a PLY file to import into some other program.
    with open(file_name, 'wb') as f:
        mesh.write_ply(f)

# *.ply -> *.obj and return obj
def ply2obj(ply_file, obj_file):
    log.info("Converting *.ply to *.obj")
    mesh = trimesh.load(ply_file)
    mesh.export(obj_file)
    return obj_file

# Button "Generate" text to 3D click
def text2model(text, model_type):
    global gd_scale
    global grid_size
    global sampler
    global samples
    if len(text) == 0:
        return None
    else:
        fake_seed = random.randint(1, 99999999)
        base_load(model_type)
        create_sampler(0, gd_scale)
        text2samples(text)
        pc = sampler.output_to_point_clouds(samples)[0]
        fig = pc2plot(pc)

        with open(text2pc_path + text + "-" + str(fake_seed) + "-pc.ply", "wb") as f:
            pc.write_ply(f)

        save_ply(pc, text2pc_path + text + "-" + str(fake_seed) + "-mesh.ply", grid_size)
        with open(f'outputs\\text2pc\\{text}-{fake_seed}.json', 'w') as sum_file:
            sum_file.write(log.generation_settings(model_type=model_type, prompt=text, gd_scale=gd_scale,
            grid_size=grid_size, fake_seed=fake_seed, version=VERSION))
        return pc2plot(pc), ply2obj(text2pc_path + text + "-" + str(fake_seed) + "-mesh.ply", text2pc_path + text + "-" + str(fake_seed) + ".obj")

# Button "Generate" image to 3D click
def image2model(image, model_type):
    global gd_scale
    global grid_size
    global sampler
    global samples
    if image is None:
        return None
    else:
        fake_seed = random.randint(1, 99999999)
        base_load(model_type)    
        create_sampler(1, gd_scale)
        image2samples(image)
        pc = sampler.output_to_point_clouds(samples)[0]
        fig = pc2plot(pc)

        with open(image2pc_path + str(fake_seed) + "-pc.ply", "wb") as f:
            pc.write_ply(f)
        save_ply(pc, image2pc_path + str(fake_seed) + "-mesh.ply", grid_size)

        with open(f'outputs\\image2pc\\{fake_seed}.json', 'w') as sum_file:
            sum_file.write(log.generation_settings(model_type=model_type, gd_scale=gd_scale,
            grid_size=grid_size, fake_seed=fake_seed, version=VERSION))
        return pc2plot(pc), ply2obj(image2pc_path + str(fake_seed) + "-mesh.ply", image2pc_path + str(fake_seed) + ".obj")


# Button "Generate" stable diffusion to 3D click
def sd2model(prompt, neg_prompt, steps, model_type):
    global sd_pipe
    global device
    global gd_scale
    global grid_size
    global sampler
    global samples
    if len(prompt) == 0 and len(neg_prompt) == 0:
        pass
    else:
        fake_seed = random.randint(1, 999999999999)
        base_load(model_type)
        create_sampler(1, gd_scale)
        tmp_generator = torch.Generator(device).manual_seed(fake_seed)

        log.info('Generating image with SD 1.5')
        generated_image = sd_pipe(prompt, guidance_scale = 7.5, num_inference_steps=steps, generator=tmp_generator).images[0]
        res_img = generated_image.resize((256, 256))
        log.info('Image generated passing to img2pc')
        image2samples(res_img)
        pc = sampler.output_to_point_clouds(samples)[0]
        fig = pc2plot(pc)
        with open(sd2pc_path + str(fake_seed) + "-pc.ply", "wb") as f:
            pc.write_ply(f)
        save_ply(pc, sd2pc_path + str(fake_seed) + "-mesh.ply", grid_size)

        with open(f'outputs\\sd2pc\\{fake_seed}.json', 'w') as sum_file:
            sum_file.write(log.generation_settings(prompt = prompt,model_type=model_type, gd_scale=gd_scale,
            grid_size=grid_size, fake_seed=fake_seed, version=VERSION))
        
        res_img.save(f'outputs\\sd2pc\\{fake_seed}.png')

        return pc2plot(pc), ply2obj(sd2pc_path + str(fake_seed) + "-mesh.ply", sd2pc_path + str(fake_seed) + ".obj")

# Update guidance scale (setter)
def gd_scale_changed(i):
    global gd_scale
    gd_scale = i

# Update grid size (setter)
def grid_size_changed(i):
    global grid_size
    grid_size = i

# Shared URL update
def sharedurl_update(chk_state):
    global cfg
    cfg["PublicURL"] = chk_state
    return cfg["PublicURL"]


def button_save():
    global cfg
    with open("config.json", "w") as cfgfile:
        cfgfile.write(json.dumps(cfg))
        log.info("Config file saved")

# Entry Point
def main():
    global device
    global cfg

    # Preload once when starting interface SDF and UPSAMPLER
    upsamplesdf_model_load()

    # Preload default model
    base_load('base40M-textvec', True)

    # Preload SD
    sd_model_load()

    # GRADIO GUI
    with gr.Blocks() as gui:
        gr.Markdown("# POINT-E WebUI by @tonyx86")

        with gr.Tab("Text to 3D"):
            with gr.Row():
                with gr.Column():
                    input_prompt = gr.Textbox(label='Prompt')
                    model_type_t = gr.Dropdown(label='Model', choices=['base40M-textvec'], interactive=True, value='base40M-textvec')
                    gd_scale_t = gr.Slider(0.0, 50.0, 3.0, label='Guidance scale', step=0.5)
                    grid_size_t = gr.Slider(0, 500, 32, label='Grid size of 3D model', step=1)
                    text2model_btn = gr.Button(value="Generate")
                with gr.Column():
                    output_plot_t = gr.Plot(label='Point Cloud')
                    output_3d_t = gr.Model3D(value=None)
            text2model_btn.click(text2model, [input_prompt, model_type_t], [output_plot_t, output_3d_t])
            gd_scale_t.change(gd_scale_changed, [gd_scale_t])
            grid_size_t.change(grid_size_changed, [grid_size_t])
        
        with gr.Tab("Image to 3D"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label='Input image')
                    model_type_i = gr.Dropdown(label='Model', choices=['base40M', 'base300M', 'base1B'], interactive=True, value='base40M')
                    gd_scale_i = gr.Slider(0.0, 50.0, 3.0, label='Guidance scale', step=0.5)
                    grid_size_i = gr.Slider(0, 500, 32, label='Grid size of 3D model', step=1)
                    image2model_btn = gr.Button(value="Generate")
                with gr.Column():
                    output_plot_i = gr.Plot(label='Point Cloud')
                    output_3d_i = gr.Model3D(value=None)
                image2model_btn.click(image2model, [input_image, model_type_i], [output_plot_i, output_3d_i])
                gd_scale_i.change(gd_scale_changed, [gd_scale_i])
                grid_size_i.change(grid_size_changed, [grid_size_i])
        
        with gr.Tab("Stable Diffusion to 3D"):
            gr.Markdown("## Temporarily use the runwayml/stable-diffusion-v1-5 model and also a Euler A sampler")
            with gr.Row():
                with gr.Column():
                        sd_prompt = gr.Textbox(label='Prompt')
                        sd_prompt_neg = gr.Textbox(label='Negative prompt')
                        sd_steps = gr.Slider(0, 200, 20, label='Stable diffusion steps', step=1)
                        model_type_esd = gr.Dropdown(label='Model', choices=['base40M', 'base300M', 'base1B'], interactive=True, value='base40M')
                        gd_scale_esd = gr.Slider(0.0, 50.0, 3.0, label='Guidance scale', step=0.5)
                        grid_size_esd = gr.Slider(0, 500, 32, label='Grid size of 3D model', step=1)
                        sd2model_btn = gr.Button(value="Generate")
                with gr.Column():
                    output_plot_esd = gr.Plot(label='Point Cloud')
                    output_3d_esd = gr.Model3D(value=None)
                sd2model_btn.click(sd2model, [sd_prompt, sd_prompt_neg, sd_steps, model_type_esd], [output_plot_esd, output_3d_esd])
                gd_scale_esd.change(gd_scale_changed, [gd_scale_esd])
                grid_size_esd.change(grid_size_changed, [grid_size_esd])
                
        
        with gr.Tab("Information"):
            gr.Label(VERSION, label='WebUI version')
            gr.Label(sys.version, label='Python version')
            gr.Label(platform.platform(), label='Platform information')
            gr.Label(torch.cuda.get_device_name(device), label='Current pytorch device')
            gr.Label(CWD, label='Current directory')
        
        with gr.Tab("Settings"):
            gr.Markdown("## You can change the settings in the config file but also here visually."
            + " Also here you will find all sorts of useful infrequently used buttons.")
            shared_url = gr.Checkbox(value = cfg["PublicURL"], label = 'Give a public link to the Internet when starting WebUI')
            save_btn = gr.Button(value="Save")
            sdf_upsampler_reload_btn = gr.Button(value='SDF and Upsampler reload')
            shared_url.change(sharedurl_update, [shared_url], [shared_url])
            save_btn.click(button_save)
            sdf_upsampler_reload_btn.click(upsamplesdf_model_load)
        
        gr.HTML('<a href="https://www.donationalerts.com/r/tonyonyxyt">Donations</a>')
    log.info("Loaded WebUI")
    gui.launch(share=cfg["PublicURL"])

if __name__ == '__main__':
    main()