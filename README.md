# POINT-E web interface made by tonyx86

## What is POINT-E
**POINT-E** is a neural network that creates a cloud of points from text or an image that can be translated into a 3D Model (**Text / Image-> 3D**) created by **OpenAI**

## Requirements

 - Python 3.10.6
 - Internet connection (POINT-E loads from the Internet models that are not in the folder, if there is, then it starts loading into memory and generating)
 - Git
 - Install pytorch from official website for CPU or GPU
 - Run in the main folder: `pip install -r requirements.txt`

## How to use it?
Run the `main.py` file

Example: `python main.py`

## FAQ

Q: How to update this WebUI?

A: You need run in main webui folder: `git pull`

Q: How to update POINT-E?

A: Just like WebUI, only you have to go to the point-e folder and run `git pull`

Q: If I have an error `ModuleNotFoundError`, what I need to do?

A: Try run `pip install -r requirements.txt` again. The fact is that if a new version of the script requires some kind of library, you must install it.

Q: Where are my results saved?

A: In the outputs/ folder there are two folders, one model created by text and the other by images.

Q: What are the files inside the outputs folder?

A: The first file is a *.ply file with a point cloud
The second file is a *.ply file with a mesh
The third file is a *.obj file, the same with a mesh only converted from *.ply

Q: What is the point_e_modelcache folder?

A: This is the folder where the POINT-E models are located, which are downloaded to generate 3d models.