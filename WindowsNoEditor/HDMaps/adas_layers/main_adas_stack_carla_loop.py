# Full ADAS stack orchestrator with CARLA integration, sensor setup, and closed-loop control
import carla
from perception_layer import perception_layer
from planning_layer import planning_layer
from control_layer import control_layer
from carla_sensor_setup import attach_sensors
from adas_vehicle_control import apply_adas_control
import time

# Wait for CARLA server to be ready
def wait_for_carla(timeout=120, retry_interval=2):
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    start = time.time()
    while True:
        try:
            world = client.get_world()
            print("Connected to CARLA server.")
            return client, world
        except RuntimeError as e:
            elapsed = time.time() - start
            print(f"[CARLA WAIT] {str(e)} (elapsed: {elapsed:.1f}s)")
            if elapsed > timeout:
                raise RuntimeError(f"Timeout waiting for CARLA server to be ready after {timeout}s. Last error: {str(e)}")
            print("Waiting for CARLA server to be ready...")
            time.sleep(retry_interval)

client, world = wait_for_carla()
# Remove all existing vehicles before spawning new ones
actors = world.get_actors().filter('vehicle.*')
for actor in actors:
    actor.destroy()
print(f"Destroyed {len(actors)} existing vehicles.")

# Spawn multiple vehicles (ego + NPCs)
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
num_vehicles = min(6, len(spawn_points))
vehicles = []
for i in range(num_vehicles):
    if i == 0:
        # Ego vehicle (with sensors)
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0] if blueprint_library.filter('vehicle.tesla.model3') else blueprint_library.filter('vehicle.*')[0]
    else:
        # NPC vehicles
        vehicle_bp = blueprint_library.filter('vehicle.audi.a2')[0] if blueprint_library.filter('vehicle.audi.a2') else blueprint_library.filter('vehicle.*')[i % len(blueprint_library.filter('vehicle.*'))]
    # Set a visible color if possible (must set on blueprint before spawning)
    if vehicle_bp.has_attribute('color'):
        color = vehicle_bp.get_attribute('color').recommended_values[0]
        vehicle_bp.set_attribute('color', color)
    spawn_point = spawn_points[i]
    v = world.try_spawn_actor(vehicle_bp, spawn_point)
    if v:
        vehicles.append(v)
        print(f"Spawned vehicle {i}: {v.type_id} at {v.get_location()}")
    else:
        print(f"Failed to spawn vehicle at {spawn_point}")

if not vehicles:
    raise RuntimeError('No vehicles could be spawned!')

vehicle = vehicles[0]  # Ego vehicle
sensors = attach_sensors(world, vehicle)
print(f"Ego vehicle is {vehicle.type_id} at {vehicle.get_location()}")


def follow_vehicle(world, vehicle):
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    # Set camera 8m behind and 3m above the vehicle
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(x=-8, z=3),
        transform.rotation))

def main():
    for step in range(1000):
        vehicle_state = {
            'speed': vehicle.get_velocity().length(),
            'location': (vehicle.get_location().x, vehicle.get_location().y, vehicle.get_location().z)
        }
        # Move spectator camera to follow the vehicle
        follow_vehicle(world, vehicle)
        # 1. Perception
        perception_data = perception_layer(sensors)
        # 2. Planning
        planned_trajectory = planning_layer(perception_data, vehicle_state)
        # 3. Control
        actuation_commands = control_layer(planned_trajectory, vehicle_state)
        # 4. Apply to vehicle
        apply_adas_control(vehicle, actuation_commands)
        time.sleep(0.05)

if __name__ == '__main__':
    main()
