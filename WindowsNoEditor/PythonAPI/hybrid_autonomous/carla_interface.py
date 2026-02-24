# CARLA simulation interface
import carla
import numpy as np

class CarlaInterface:
    def __init__(self, host='127.0.0.1', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

    def get_camera_image(self, vehicle):
        # Placeholder: return dummy image
        return np.zeros((720, 1280, 3), dtype=np.uint8)

    def get_vehicle_state(self, vehicle):
        # Placeholder: return dummy state
        return {'speed': 0.0, 'location': (0,0,0)}

    def apply_control(self, vehicle, control):
        vehicle.apply_control(control)
