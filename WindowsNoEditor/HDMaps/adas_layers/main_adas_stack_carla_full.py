# Full ADAS stack orchestrator with CARLA integration and sensor setup
import carla
from perception_layer import perception_layer
from planning_layer import planning_layer
from control_layer import control_layer
from carla_sensor_setup import attach_sensors

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Get first vehicle in the world (for demo)
vehicle = world.get_actors().filter('vehicle.*')[0]

# Attach sensors
sensors = attach_sensors(world, vehicle)

def main():
    vehicle_state = {
        'speed': vehicle.get_velocity().length(),
        'location': (vehicle.get_location().x, vehicle.get_location().y, vehicle.get_location().z)
    }

    # 1. Perception
    perception_data = perception_layer(sensors)

    # 2. Planning
    planned_trajectory = planning_layer(perception_data, vehicle_state)

    # 3. Control
    actuation_commands = control_layer(planned_trajectory, vehicle_state)

    print("ADAS stack complete. Actuation commands:", actuation_commands)
    # Here you would convert actuation_commands to carla.VehicleControl and apply to vehicle
    # vehicle.apply_control(control)

if __name__ == '__main__':
    main()
