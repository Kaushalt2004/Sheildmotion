#!/usr/bin/env python
# Minimal autonomous driving starter for CARLA 0.9.13 (Windows)
# Spawns a vehicle and enables Traffic Manager autopilot.
# Run from: WindowsNoEditor/PythonAPI/examples
#   python ./autonomous_start.py --host 127.0.0.1 --port 2000

import argparse
import glob
import os
import queue
import random
import sys
import time
from pathlib import Path

import numpy as np


def _ensure_carla_import():
    """Try to import carla; give clear guidance if not found.
    On Windows with CARLA 0.9.13, Python 3.7 (x64) is required for the bundled egg.
    """
    try:
        import carla  # noqa: F401
        return
    except Exception:
        pass

    egg_glob = os.path.join('..', 'carla', 'dist',
                             f"carla-*{sys.version_info.major}.{sys.version_info.minor}-"
                             f"{'win-amd64' if os.name == 'nt' else 'linux-x86_64'}.egg")
    matches = glob.glob(egg_glob)
    if matches:
        sys.path.append(matches[0])
        try:
            import carla  # noqa: F401
            return
        except Exception:
            pass

    msg = (
        f"CARLA Python API not found for Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        "On Windows with CARLA 0.9.13, use Python 3.7 (64-bit), activate it, and run from PythonAPI/examples.\n"
        f"Searched for egg: {egg_glob}"
    )
    raise RuntimeError(msg)


def main():
    _ensure_carla_import()
    import carla

    parser = argparse.ArgumentParser(description='Start a Traffic Manager autopilot vehicle')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    parser.add_argument('--port', default=2000, type=int, help='CARLA server port')
    parser.add_argument('--timeout', default=20.0, type=float, help='Client RPC timeout in seconds')
    parser.add_argument('--tm-port', default=8000, type=int, help='Traffic Manager port')
    parser.add_argument('--town', default='', help='Town name to load (e.g., Town03). Empty = current world')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for blueprint selection')
    parser.add_argument('--no-camera', action='store_true', help='Disable the RGB camera sensor')
    parser.add_argument('--no-lidar', action='store_true', help='Disable the LiDAR sensor')
    parser.add_argument('--display-camera', action='store_true', help='Open a pygame window showing RGB frames')
    parser.add_argument('--lidar-log-dir', default='', help='Directory to dump LiDAR frames as NPZ files')
    parser.add_argument('--cam-width', type=int, default=1280, help='Camera width in pixels')
    parser.add_argument('--cam-height', type=int, default=720, help='Camera height in pixels')
    parser.add_argument('--cam-fov', type=float, default=90.0, help='Camera field of view in degrees')
    parser.add_argument('--lidar-range', type=float, default=70.0, help='LiDAR range in meters')
    parser.add_argument('--lidar-rotation-frequency', type=float, default=20.0,
                        help='LiDAR rotation frequency (Hz)')
    parser.add_argument('--lidar-points-per-second', type=int, default=1_000_000,
                        help='LiDAR points per second')
    args = parser.parse_args()

    random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = _wait_for_world(client, args.town)

    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(False)
    tm.set_hybrid_physics_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.global_percentage_speed_difference(0.0)

    blueprint_library = world.get_blueprint_library()
    bps = blueprint_library.filter('vehicle.*')
    if not bps:
        print('No vehicle blueprints found')
        return

    bp = random.choice(bps)
    if bp.has_attribute('role_name'):
        bp.set_attribute('role_name', 'hero')

    spawns = world.get_map().get_spawn_points()
    if not spawns:
        print('No spawn points available')
        return

    vehicle = None
    sensors = []
    camera_queue = None
    lidar_queue = None
    pygame_display = None
    pygame_module = None
    lidar_log_dir = Path(args.lidar_log_dir).expanduser().resolve() if args.lidar_log_dir else None
    if lidar_log_dir:
        lidar_log_dir.mkdir(parents=True, exist_ok=True)
    try:
        vehicle = world.try_spawn_actor(bp, random.choice(spawns))
        if vehicle is None:
            print('Failed to spawn vehicle; try re-running')
            return

        vehicle.set_autopilot(True, args.tm_port)
        print(f'Hero vehicle {vehicle.id} driving with Traffic Manager. Press Ctrl+C to stop.')

        if not args.no_camera:
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(args.cam_width))
            camera_bp.set_attribute('image_size_y', str(args.cam_height))
            camera_bp.set_attribute('fov', str(args.cam_fov))
            camera_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera = world.spawn_actor(camera_bp, camera_tf, attach_to=vehicle)
            sensors.append(camera)
            camera_queue = queue.Queue()
            camera.listen(camera_queue.put)
            print(f'RGB camera {camera.id} streaming {args.cam_width}x{args.cam_height} @ {args.cam_fov}°')

            if args.display_camera:
                try:
                    import pygame as _pygame
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError('pygame is required for --display-camera. Install with "pip install pygame".') from exc
                pygame_module = _pygame
                pygame_module.init()
                pygame_display = pygame_module.display.set_mode((args.cam_width, args.cam_height))
                pygame_module.display.set_caption('CARLA Hero RGB')

        if not args.no_lidar:
            lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', str(args.lidar_range))
            lidar_bp.set_attribute('rotation_frequency', str(args.lidar_rotation_frequency))
            lidar_bp.set_attribute('points_per_second', str(args.lidar_points_per_second))
            lidar_bp.set_attribute('channels', '32')
            lidar_bp.set_attribute('upper_fov', '10.0')
            lidar_bp.set_attribute('lower_fov', '-30.0')
            lidar_tf = carla.Transform(carla.Location(x=0.0, z=2.5))
            lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
            sensors.append(lidar)
            lidar_queue = queue.Queue()
            lidar.listen(lidar_queue.put)
            print(f'LiDAR {lidar.id} streaming {args.lidar_points_per_second:,} pts/s over {args.lidar_range} m')

        while True:
            latest_image = None
            latest_lidar = None
            if camera_queue is not None:
                while True:
                    try:
                        latest_image = camera_queue.get_nowait()
                    except queue.Empty:
                        break
            if latest_image is not None:
                if args.display_camera and pygame_display is not None and pygame_module is not None:
                    frame = _image_to_array(latest_image)
                    surface = pygame_module.surfarray.make_surface(frame.swapaxes(0, 1))
                    pygame_display.blit(surface, (0, 0))
                    pygame_module.display.flip()
                    for event in pygame_module.event.get():
                        if event.type == pygame_module.QUIT:
                            raise KeyboardInterrupt
                else:
                    print(f'RGB frame {latest_image.frame} captured at t={latest_image.timestamp:.2f}s')

            if lidar_queue is not None:
                while True:
                    try:
                        latest_lidar = lidar_queue.get_nowait()
                    except queue.Empty:
                        break
            if latest_lidar is not None:
                point_count = len(latest_lidar)
                if lidar_log_dir:
                    pts = np.frombuffer(latest_lidar.raw_data, dtype=np.float32)
                    pts = np.reshape(pts, (-1, 4))
                    np.savez_compressed(lidar_log_dir / f'lidar_{latest_lidar.frame:06d}.npz', points=pts)
                    print(f'Saved LiDAR frame {latest_lidar.frame} ({point_count} pts) -> {lidar_log_dir}')
                else:
                    print(f'LiDAR frame {latest_lidar.frame} with {point_count} points at t={latest_lidar.timestamp:.2f}s')

            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        for sensor in sensors:
            try:
                sensor.stop()
            except RuntimeError:
                pass
            sensor.destroy()
        if vehicle is not None:
            vehicle.destroy()
        if pygame_module is not None:
            pygame_module.quit()
        print('Done.')


def _image_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    # Convert BGRA to RGB
    return array[:, :, ::-1]


def _wait_for_world(client, desired_town, total_timeout=60.0):
    start = time.time()
    last_error = None
    while time.time() - start < total_timeout:
        try:
            if desired_town:
                print(f'Loading town {desired_town}...')
                return client.load_world(desired_town)
            world = client.get_world()
            return world
        except RuntimeError as err:
            last_error = err
            print('Waiting for simulator to be ready...')
            time.sleep(1.0)
    raise RuntimeError(f'Unable to connect to simulator after {total_timeout:.0f}s: {last_error}')


if __name__ == '__main__':
    main()
