# Control Layer: Low-level actuation (steering, throttle, brake)

def control_layer(planned_trajectory, vehicle_state):
    # Return actuation commands
    print("Control layer active: Executing control commands.")
    # Simple proportional controller for throttle
    target_speed = planned_trajectory.get("target_speed", 0.0)
    current_speed = vehicle_state.get("speed", 0.0)
    error = target_speed - current_speed
    throttle = min(max(0.2 * error, 0.0), 1.0)  # P controller, clamp 0-1
    return {"throttle": throttle, "steer": 0.0, "brake": 0.0}
