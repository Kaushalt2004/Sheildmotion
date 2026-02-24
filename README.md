
# Level 5 Autonomy V2V Communication System

## Overview

This project implements a cutting-edge Level 5 Autonomous Driving System with advanced Vehicle-to-Vehicle (V2V) communication capabilities. Built on the CARLA simulator, it features:

- Full-stack autonomous driving (perception, planning, control)
- Real-time V2V communication for cooperative driving
- Sensor fusion (Camera, LiDAR, Radar, GPS, IMU)
- Neural trajectory planning and adaptive cruise control
- Emergency collision avoidance and predictive traffic light handling
- Modular, extensible Python and C++ codebase

## Features

- **Level 5 Autonomy**: No human intervention required in any scenario
- **V2V Communication**: Cooperative awareness, intent sharing, and safety messaging
- **Sensor Suite**: Multi-modal sensor integration for robust perception
- **Advanced Planning**: Neural and rule-based trajectory generation
- **Smooth Control**: Jerk-minimized PID and adaptive cruise
- **Simulation & Visualization**: Real-time metrics, logging, and scenario playback

## Repository Structure

- `PythonAPI/` — Core Python modules, examples, and hybrid autonomy stack
- `WindowsNoEditor/` — Source/config files from CARLA build (no binaries)
- `HDMaps/adas_layers/` — ADAS and sensor setup modules
- `Plugins/` — V2V, visualization, and integration plugins

## Getting Started

1. Clone this repository
2. Install dependencies (see requirements.txt in relevant folders)
3. Launch the CARLA simulator
4. Run example scripts from `PythonAPI/examples/`

## Key Technologies

- **CARLA Simulator**
- **Python 3.7+ / C++**
- **Pygame, NumPy, OpenCV, PyTorch**
- **Custom V2V Protocols**

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License. See LICENSE for details.
