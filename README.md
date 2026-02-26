

---

## 🚗 Overview

Level 5 Autonomy V2V Communication System is a state-of-the-art autonomous driving platform built on the CARLA simulator. It features:

- Full-stack autonomy: perception, planning, and control
- Real-time V2V (Vehicle-to-Vehicle) communication for cooperative driving
- Advanced sensor fusion (Camera, LiDAR, Radar, GPS, IMU)
- Neural trajectory planning, adaptive cruise, and emergency handling
- Modular, extensible Python and C++ codebase

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Key Technologies](#-key-technologies)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Level 5 Autonomy**: No human intervention required in any scenario
- **V2V Communication**: Cooperative awareness, intent sharing, and safety messaging
- **Sensor Suite**: Multi-modal sensor integration for robust perception
- **Advanced Planning**: Neural and rule-based trajectory generation
- **Smooth Control**: Jerk-minimized PID and adaptive cruise
- **Simulation & Visualization**: Real-time metrics, logging, and scenario playback

---

## 📁 Repository Structure

```text
PythonAPI/         # Core Python modules, examples, hybrid autonomy stack
WindowsNoEditor/   # Source/config files from CARLA build (no binaries)
HDMaps/adas_layers/ # ADAS and sensor setup modules
Plugins/           # V2V, visualization, and integration plugins
```

---

## 🚀 Getting Started

1. **Clone this repository**
	 ```bash
	 git clone https://github.com/Kaushalt2004/Sheildmotion.git
	 cd Sheildmotion
	 ```
2. **Install dependencies** (see requirements.txt in relevant folders)
3. **Launch the CARLA simulator**
4. **Run example scripts** from `PythonAPI/examples/`

---

## 🛠️ Key Technologies

- CARLA Simulator
- Python 3.7+ / C++
- Pygame, NumPy, OpenCV, PyTorch
- Custom V2V Protocols

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
