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
from point_e.util.pc_to_mesh import marching_cubes_mesh

from pyntcloud import PyntCloud
import matplotlib.colors
import plotly.graph_objs as go

import trimesh

# Variables
VERSION = "0.1.6"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_name = ''
base_model = None
base_diffusion = None
upsampler_model = None
upsampler_diffusion = None
sdf_model = None
samples = None
sampler = None

cwd_path = os.getcwd()
gd_scale = 3.0
grid_size = 32.0
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
    global sdf_model
    try:
        info(f'Loading model {model_name}')
        # Base model preparation
        base_name = model_name # Image-based: base40M, base300M, base1B; Text-based: base40M-textvec
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        # Upsampler model preparation
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

        # SDF model preparation
        sdf_name = 'sdf'
        sdf_model = model_from_config(MODEL_CONFIGS[sdf_name], device)
        sdf_model.eval()

        # Load all models
        base_model.load_state_dict(load_checkpoint(base_name, device))
        upsampler_model.load_state_dict(load_checkpoint('upsample', device))
        sdf_model.load_state_dict(load_checkpoint(sdf_name, device))

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

# Sampling text
def text2samples(prompt):
    global samples
    global sampler
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x

# Sampling image
def image2samples(image):
    global samples
    global sampler
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[image]))):
        samples = x

# Get plot from point cloud
def pc2plot(pc):
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


def save_ply(pc, file_name, grid_size):
    global sdf_model
    # Produce a mesh (with vertex colors)
    mesh = marching_cubes_mesh(
        pc=pc,
        model=sdf_model,
        batch_size=4096,
        grid_size=grid_size, # increase to 128 for resolution used in evals
        progress=True,
    )

    # Write the mesh to a PLY file to import into some other program.
    with open(file_name, 'wb') as f:
        mesh.write_ply(f)

# *.ply -> *.obj and return obj
def ply2obj(ply_file, obj_file):
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
        fake_seed = random.randint(1, 999999)
        load_model(model_type)
        create_sampler(0, gd_scale)
        text2samples(text)
        pc = sampler.output_to_point_clouds(samples)[0]
        fig = pc2plot(pc)

        with open(text2pc_path + text + "-" + str(fake_seed) + "-pc.ply", "wb") as f:
            pc.write_ply(f)
        save_ply(pc, text2pc_path + text + "-" + str(fake_seed) + "-mesh.ply", grid_size)
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
        fake_seed = random.randint(1, 999999)
        load_model(model_type)    
        create_sampler(1, gd_scale)
        image2samples(image)
        pc = sampler.output_to_point_clouds(samples)[0]
        fig = pc2plot(pc)

        with open(image2pc_path + str(fake_seed) + "-pc.ply", "wb") as f:
            pc.write_ply(f)
        save_ply(pc, image2pc_path + str(fake_seed) + "-mesh.ply", grid_size)
        return pc2plot(pc), ply2obj(image2pc_path + str(fake_seed) + "-mesh.ply", image2pc_path + str(fake_seed) + ".obj")

# Update guidance scale
def gd_scale_changed(i):
    global gd_scale
    gd_scale = float(i)

# Update grid size
def grid_size_changed(i):
    global grid_size
    grid_size = float(i)

# Entry Point
def main():
    global device
    with gr.Blocks() as gui:
        gr.Markdown("# POINT-E WebUI by @tonyx86")

        with gr.Tab("Text to 3D"):
            with gr.Row():
                with gr.Column():
                    input_prompt = gr.Textbox(label='Prompt')
                    model_type_t = gr.Dropdown(label='Model', choices=['base40M-textvec'], interactive=True, value='base40M-textvec')
                    gd_scale_t = gr.Slider(0.0, 50.0, 3.0, label='Guidance scale', step=0.5)
                    grid_size_t = gr.Slider(0.0, 500.0, 32.0, label='Grid size of 3D model', step=0.5)
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
                    grid_size_i = gr.Slider(0.0, 500.0, 32.0, label='Grid size of 3D model', step=0.5)
                    image2model_btn = gr.Button(value="Generate")
                with gr.Column():
                    output_plot_i = gr.Plot(label='Point Cloud')
                    output_3d_i = gr.Model3D(value=None)
                image2model_btn.click(image2model, [input_image, model_type_i], [output_plot_i, output_3d_i])
                gd_scale_i.change(gd_scale_changed, [gd_scale_i])
                grid_size_i.change(grid_size_changed, [grid_size_i])
        
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