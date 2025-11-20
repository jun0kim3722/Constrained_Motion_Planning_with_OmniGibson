# Constrained Motion Planning with OmniGibson

This repository provides a framework for performing constrained motion planning on a **UR5e** robot manipulator using **OMPL**, within simulation environments created using **OmniGibson**.  
The system supports several manipulation tasks that involve environmental and kinematic constraints.

## Available Tasks
- Liquid Pouring
- Object Cutting
- Drawer Opening
- Cabinet Opening
- Stirring


## Running the Project

### 1. Start the Docker Environment
Run the following script from your terminal:
```
./run_docker.sh [path_to_current_git_directory]
```

### 2. Launch the Simulation
```
cd benchmark_src/
python run_scene.py [SpaceType] [PlannerType]
```
You will then be prompted to select a task.

## Demo Video
[![Demo Video](https://img.youtube.com/vi/VIDEO_ID/0.jpg)]([https://www.youtube.com/watch?v=VIDEO_ID](https://youtu.be/5mM_1vnh3U8))
