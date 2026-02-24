# Main ADAS stack orchestrator: Perception -> Planning -> Control
from perception_layer import perception_layer
from planning_layer import planning_layer
from control_layer import control_layer

# Placeholder vehicle and sensor objects
class DummyVehicle:
    def __init__(self):
        self.state = {'speed': 0.0, 'location': (0,0,0)}

class DummySensors:
    pass

def main():
    vehicle = DummyVehicle()
    sensors = DummySensors()
    vehicle_state = vehicle.state

    # 1. Perception
    perception_data = perception_layer(sensors)

    # 2. Planning
    planned_trajectory = planning_layer(perception_data, vehicle_state)

    # 3. Control
    actuation_commands = control_layer(planned_trajectory, vehicle_state)

    print("ADAS stack complete. Actuation commands:", actuation_commands)

if __name__ == '__main__':
    main()
