This is a HandTracker based on Particle Swarm Optimization (PSO). 
The input to the system is a depth map, and the output will be and mesh fit to the image.
The Renderer used in this system is adopted from: https://github.com/martinResearch/DEODR

Instruction:

1- Install deodr package using the command line: pip install deodr
I have made slight changes to their code to make the Renderer use Orthographic Projection.
So, when you have deodr package installed, go to where the deodr package is installed, open the file differentiable_renderer.py. Go the the class definition of Camera in the beginning, then go the the function project_points(), modify the third line to be :    projected = p_camera[:, :2]

2- Download Mano repository by running this command: git clone https://github.com/hassony2/manopth.git

3- Download MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/ and copy it to the folder manopth/mano/models

4- Place the main file in the directory manopth and run it (both PSO.py and test.ipynb should be placed in the directory /manopth)
