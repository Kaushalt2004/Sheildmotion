# Planning Layer: Path, behavior, and trajectory planning

def planning_layer(perception_data, vehicle_state):
    # Return planned trajectory or control commands
    print("Planning layer active: Generating path and behavior.")
    # Simple planning: always target 10 m/s
    return {"target_speed": 10.0}
