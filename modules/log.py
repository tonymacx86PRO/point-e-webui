# Print information
def info(msg):
    print(f"[INFO] {msg}\n")

# Print error
def error(msg):
    print(f"[ERROR] {msg}\n")

# Print warning
def warn(msg):
    print(f"[WARN] {msg}\n")

# Print debug
def debug(msg):
    print(f"[DBG] {msg}\n")

# Print summary of generated 3D model and return json
def generation_settings(model_type, prompt = 'Image', gd_scale = 3.0, grid_size = 32, fake_seed = 0, version = '0.0.0'):
    info(f'''\nThe 3d model was successfully created with the following settings:
    Prompt: {prompt}
    POINT-E Model: {model_type}
    Guidance scale: {gd_scale}
    Grid size: {grid_size}
    Generation ID: {fake_seed}
    WebUI version: {version}\n
    ''')
    import json
    generation_summary = {
        "prompt": prompt,
        "point-e-model": model_type,
        "guidance-scale": gd_scale,
        "grid-size": grid_size,
        "gen-id": fake_seed,
        "webui-version": version
    }
    return json.dumps(generation_summary)
