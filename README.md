# **DAI Project**

## **Overview**

This project is a comprehensive framework for **Autonomous Driving Simulation** and related tasks. It integrates modules for vehicle control, computer vision, reinforcement learning environments, and simulation via CARLA.

The DAI project includes:
1. **CARLA Simulator**: For autonomous vehicle simulation.
2. **Computer Vision (CV)**: Lane detection, traffic light/sign detection, object recognition, and segmentation.
3. **Environment Modeling**: RL-based environments and custom reward systems.
4. **Dataset Generation**: Tools for generating synthetic datasets.
5. **Visualization Tools**: Insights into agent behavior and system performance.

---

## **Project Structure**

All source code is located in the DAI folder and is structured as follows

- **cv/**  
   Contains computer vision utilities:
   - `lane_detection.py` - Lane detection.
   - `traffic_sign_classification.py` - Traffic sign recognition.
   - `road_marker_detection.py` - Road marker detection.
   - `object_detection.py` - Object recognition.
   - `traffic_light_detection.py` - Traffic light detection.
   - `traffic_lane_segmentation.py` - Image segmentation for traffic lanes.

- **dataset_generator/**  
   Tools for generating datasets:
   - `extract.py` - Extracts simulation data.
   - `generate.py` - Generates synthetic datasets.
   - `spawn.py` - Spawns objects for data collection.

- **environment/**  
   CARLA-based RL environments:
   - `carla_env.py` - Environment wrapper.
   - `mock/` - Contains mock reward systems for RL.

- **interfaces/**  
   Interfaces for data handling:
   - `data_carriers.py` - Data transfer utilities.
   - `image.py` - Image interface module.

- **simulator/**  
   Core simulation utilities:
   - `carla_world.py` - CARLA world setup.
   - `wrappers/` - Simulation and vehicle control wrappers.
   - `segmentation.py` - Handles segmentation tasks.

- **scripts/**  
   Scripts for running and training agents:
   - `train.py` - Training pipeline.
   - `main.py` - Main simulation entry point.

- **visuals/**  
   Visualization tools:
   - `visuals.py` - Core visualization module.
   - `visual_utils.py` - Helper functions for visualization.

- **utils/**  
   General utilities for the project.

---

## **Installation**

### **Prerequisites**
- [uv](https://astral.sh/blog/uv)
- Access to a Carla Simulator

### **Steps**

1. Clone the repository:
   ```bash
   git clone https://github.com/femw03/DAI_project
   ```
2. Setup python environment:
   ```bash
   uv sync
   ```
3. Run the main scripts
   ```bash
   uv run -m DAI.scripts.main
   ```

## **Features**

-	CARLA Integration:
    -	Uses CARLA for autonomous vehicles simulation.
-	Computer Vision:
    -	Lane detection, traffic sign recognition,and object detection.
-	Reinforcement Learning:
    -	Environments and reward systems forRL-based agents.
-	Dataset Generation:
    -	Tools to generate synthetic datasets from simulation.
-	Visualization:
    -	Visualize agent behavior and system outputs.