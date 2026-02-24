# Example: Convert ADAS control output to CARLA VehicleControl and apply to vehicle
import carla

def apply_adas_control(vehicle, actuation_commands):
    control = carla.VehicleControl()
    # Example: actuation_commands should contain 'throttle', 'steer', 'brake' keys
    control.throttle = float(actuation_commands.get('throttle', 0.0))
    control.steer = float(actuation_commands.get('steer', 0.0))
    control.brake = float(actuation_commands.get('brake', 0.0))
    control.hand_brake = False
    control.manual_gear_shift = False
    vehicle.apply_control(control)
    print(f"Applied control: throttle={control.throttle}, steer={control.steer}, brake={control.brake}")
