# Main script to run hybrid autonomous driving system with CARLA, ROS 2, and neural perception
import numpy as np
from perception import PerceptionModule
from planning import Planner
from carla_interface import CarlaInterface

# Optionally import ROS 2 node (run separately if needed)
# from ros2_interface import main as ros2_main

def main():
    # Initialize modules
    carla_sim = CarlaInterface()
    perception = PerceptionModule()
    planner = Planner()

    # Get vehicle (placeholder: get first vehicle in world)
    vehicle = carla_sim.world.get_actors().filter('vehicle.*')[0]

    for step in range(1000):
        # 1. Get camera image from CARLA
        image = carla_sim.get_camera_image(vehicle)

        # 2. Perception: detect objects
        boxes, labels, scores = perception.detect_objects(image)
        print(f"Step {step}: Detected {len(boxes)} objects.")

        # 3. Get vehicle state
        state = carla_sim.get_vehicle_state(vehicle)
        current_speed = state['speed']
        target_speed = 10.0  # m/s, example target

        # 4. Planning/control
        throttle = planner.compute_control(target_speed, current_speed)
        control = carla_sim.world.get_blueprint_library().find('vehicle.tesla.model3').get_attribute('role_name')
        # For demo, just print throttle value
        print(f"Throttle command: {throttle}")

        # 5. Apply control (placeholder, not a real carla.VehicleControl)
        # carla_sim.apply_control(vehicle, control)

        # 6. Step simulation (if needed)
        # carla_sim.world.tick()

if __name__ == '__main__':
    main()
