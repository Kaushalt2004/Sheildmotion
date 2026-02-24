#!/usr/bin/env python

"""
Tesla-Style Autonomous Driving System with V2V Communication

Professional autonomous driving simulation that replicates Tesla's Autopilot/FSD capabilities:
- Advanced sensor fusion (Camera + LiDAR + Radar + GPS + IMU)
- Neural trajectory planning with multi-path evaluation
- Smooth PID control with jerk minimization
- Adaptive cruise control and lane keeping assist
- Vehicle-to-Vehicle (V2V) communication for cooperative driving
- Predictive traffic light and obstacle detection
- Emergency collision avoidance
- Real-time visualization and logging

Usage:
    python tesla_autonomous_v2v.py --num-vehicles 5 --enable-v2v --display
"""

import argparse
import glob
import os
import sys
import time
import random
import queue
import math
from collections import deque
from datetime import datetime

import numpy as np

# Import CARLA
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
# Import custom ADAS layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../adas_layers')))
from perception_layer import perception_layer
from planning_layer import planning_layer
from control_layer import control_layer
from carla_sensor_setup import attach_sensors
from adas_vehicle_control import apply_adas_control

# Import custom modules
# Add parent directory to path to import our custom modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'carla'))
sys.path.insert(0, parent_dir)

# Add agents directory
agents_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'carla', 'agents', 'navigation'))
sys.path.insert(0, agents_dir)

try:
    from tesla_autopilot import TeslaBrainController, SensorFusionModule, TrajectoryPlanner
    from v2v_communication import V2VCommunicationManager, V2VMessage, MessageType, MessagePriority
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Make sure tesla_autopilot.py and v2v_communication.py are in the correct directories")
    print(f"Looking in: {parent_dir}")
    print(f"Looking in: {agents_dir}")
    sys.exit(1)

# Try pygame for visualization
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available, display will be disabled")


class AutonomousVehicle:
    """
    Autonomous vehicle with Tesla-style controller and V2V communication
    """
    
    def __init__(self, 
                 vehicle_actor,
                 world,
                 v2v_manager=None,
                 enable_sensors=True,
                 target_speed=50.0):
        """
        Initialize autonomous vehicle
        
        Args:
            vehicle_actor: CARLA vehicle actor
            world: CARLA world
            v2v_manager: V2V communication manager (optional)
            enable_sensors: Whether to attach sensors
            target_speed: Target cruising speed in km/h
        """
        self.vehicle = vehicle_actor
        self.world = world
        self.map = world.get_map()
        self.v2v_manager = v2v_manager
        self.vehicle_id = vehicle_actor.id
        
        # Attach ADAS sensors (replaces _setup_sensors)
        self.sensors = attach_sensors(world, vehicle_actor)
        self.metrics = {
            'total_distance': 0.0,
            'avg_speed': 0.0,
            'collisions': 0
        }
    
    def _setup_sensors(self):
        """Setup multi-sensor suite similar to Tesla"""
        blueprint_library = self.world.get_blueprint_library()
        
        # 1. Front-facing camera (main perception camera)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(
            carla.Location(x=2.5, z=0.9),
            carla.Rotation(pitch=-10)
        )
        camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle
        )
        camera.listen(lambda image: self._process_camera(image))
        self.sensors.append(camera)
        
        # 2. LiDAR (roof-mounted)
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '1000000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('range', '100')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.4))
        lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle
        )
        lidar.listen(lambda data: self._process_lidar(data))
        self.sensors.append(lidar)
        
        # 3. GNSS for global localization
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
        gnss = self.world.spawn_actor(
            gnss_bp, gnss_transform, attach_to=self.vehicle
        )
        gnss.listen(lambda data: self._process_gnss(data))
        self.sensors.append(gnss)
        
        # 4. IMU for motion sensing
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0.0, z=0.0))
        imu = self.world.spawn_actor(
            imu_bp, imu_transform, attach_to=self.vehicle
        )
        imu.listen(lambda data: self._process_imu(data))
        self.sensors.append(imu)
        
        # 5. Collision sensor for safety metrics
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        collision.listen(lambda event: self._on_collision(event))
        self.sensors.append(collision)
    
    def _process_camera(self, image):
        """Process camera image for vision-based perception"""
        self.sensor_data['camera'] = image
        # In production, this would run neural networks for:
        # - Lane detection
        # - Object detection
        # - Traffic sign/light recognition
        # - Drivable area segmentation
    
    def _process_lidar(self, lidar_data):
        """Process LiDAR point cloud for 3D perception"""
        self.sensor_data['lidar'] = lidar_data
        # Extract 3D obstacles from point cloud
        self._detect_obstacles_from_lidar(lidar_data)
    
    def _process_gnss(self, gnss_data):
        """Process GPS data for localization"""
        self.sensor_data['gnss'] = gnss_data
    
    def _process_imu(self, imu_data):
        """Process IMU data for motion tracking"""
        self.sensor_data['imu'] = imu_data
    
    def _on_collision(self, event):
        """Handle collision event"""
        self.metrics['collisions'] += 1
        print(f"[Vehicle {self.vehicle_id}] Collision detected!")
    
    def _detect_obstacles_from_lidar(self, lidar_data):
        """
        Extract obstacle information from LiDAR point cloud
        Simplified version - in production would use clustering algorithms
        """
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # Get vehicle location
        vehicle_location = self.vehicle.get_location()
        
        # Detect obstacles from world actors (simplified)
        self.detected_obstacles = []
        actors = self.world.get_actors().filter('vehicle.*')
        
        for actor in actors:
            if actor.id == self.vehicle_id:
                continue
            
            actor_location = actor.get_location()
            distance = vehicle_location.distance(actor_location)
            
            if distance < 50.0:  # Within sensing range
                velocity = actor.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                self.detected_obstacles.append({
                    'id': actor.id,
                    'distance': distance,
                    'position': [actor_location.x, actor_location.y, actor_location.z],
                    'speed': speed,
                    'type': 'vehicle'
                })
    
    def _initialize_route(self):
        """Initialize planned route"""
        current_location = self.vehicle.get_location()
        current_waypoint = self.map.get_waypoint(current_location)
        
        # Generate waypoints for next 50 meters
        self.waypoints = []
        for i in range(25):
            next_waypoints = current_waypoint.next(2.0)
            if next_waypoints:
                current_waypoint = random.choice(next_waypoints)
                self.waypoints.append(current_waypoint)
    
    def _update_traffic_state(self):
        """Update traffic light and sign information"""
        vehicle_location = self.vehicle.get_location()
        
        # Check for traffic lights
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light:
                tl_location = traffic_light.get_location()
                distance = vehicle_location.distance(tl_location)
                
                if traffic_light.state == carla.TrafficLightState.Red:
                    self.traffic_state['red_light_distance'] = distance
                else:
                    self.traffic_state['red_light_distance'] = None
        else:
            self.traffic_state['red_light_distance'] = None
        
        # Get speed limit from current waypoint
        current_waypoint = self.map.get_waypoint(vehicle_location)
        if current_waypoint:
            self.traffic_state['speed_limit'] = current_waypoint.lane_width * 10  # Approximation
    
    def _get_v2v_data(self):
        """Get data from other vehicles via V2V communication"""
        v2v_data = {'nearby_vehicles': []}
        
        if self.v2v_manager:
            # Broadcast own state
            vehicle_transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            acceleration = self.vehicle.get_acceleration()
            
            message = V2VMessage(
                sender_id=self.vehicle_id,
                timestamp=time.time(),
                message_type=MessageType.COOPERATIVE_AWARENESS,
                priority=MessagePriority.MEDIUM,
                position=[vehicle_transform.location.x, 
                         vehicle_transform.location.y,
                         vehicle_transform.location.z],
                velocity=[velocity.x, velocity.y, velocity.z],
                acceleration=[acceleration.x, acceleration.y, acceleration.z],
                heading=vehicle_transform.rotation.yaw,
                data={'obstacles': self.detected_obstacles}
            )
            
            self.v2v_manager.broadcast_message(message)
            
            # Receive messages from nearby vehicles
            received_messages = self.v2v_manager.receive_messages(
                vehicle_id=self.vehicle_id,
                current_time=time.time(),
                max_age=1.0
            )
            
            v2v_data['nearby_vehicles'] = [msg.data for msg in received_messages]
        
        return v2v_data
    
    def update(self):
        """Main update loop for ADAS stack"""
        # Get vehicle state
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        vehicle_state = {
            'speed': speed,
            'location': (self.vehicle.get_location().x, self.vehicle.get_location().y, self.vehicle.get_location().z)
        }
        # Perception
        perception_data = perception_layer(self.sensors)
        # Planning
        planned_trajectory = planning_layer(perception_data, vehicle_state)
        # Control
        actuation_commands = control_layer(planned_trajectory, vehicle_state)
        # Apply
        apply_adas_control(self.vehicle, actuation_commands)
        # Metrics
        self.metrics['avg_speed'] = 0.9 * self.metrics['avg_speed'] + 0.1 * speed
        self.metrics['total_distance'] += speed * 0.05
    
    def get_status(self):
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
        return {
            'id': self.vehicle_id,
            'speed': speed,
            'metrics': self.metrics
        }
    
    def cleanup(self):
        for sensor in self.sensors:
            if hasattr(sensor, 'is_alive') and sensor.is_alive:
                sensor.destroy()
        self.sensors.clear()


def main():
    """Main simulation entry point"""
    parser = argparse.ArgumentParser(description='Tesla-style Autonomous Driving with V2V')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--num-vehicles', type=int, default=3, help='Number of autonomous vehicles')
    parser.add_argument('--target-speed', type=float, default=50.0, help='Target speed in km/h')
    parser.add_argument('--enable-v2v', action='store_true', help='Enable V2V communication')
    parser.add_argument('--v2v-range', type=float, default=200.0, help='V2V communication range in meters')
    parser.add_argument('--display', action='store_true', help='Enable visualization')
    parser.add_argument('--duration', type=int, default=300, help='Simulation duration in seconds')
    parser.add_argument('--town', default='Town03', help='CARLA town to load')
    parser.add_argument('--sync-mode', action='store_true', help='Enable synchronous mode')
    parser.add_argument('--tm-port', type=int, default=8000, help='Traffic Manager port')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize pygame for display
    display = None
    if args.display and PYGAME_AVAILABLE:
        pygame.init()
        display = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption('Tesla Autonomous Driving - V2V System')
        font = pygame.font.Font(None, 24)
    
    client = None
    autonomous_vehicles = []
    
    import os
    # Set CARLA to low graphics mode for better performance
    os.environ['CARLA_LOW_QUALITY_MODE'] = '1'
    try:
        # Connect to CARLA
        print(f"Connecting to CARLA server at {args.host}:{args.port}...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        
        # Load or get world
        if args.town and args.town.strip():
            print(f"Loading world: {args.town}")
            world = client.load_world(args.town)
        else:
            print("Using current world...")
            world = client.get_world()
        
        # Configure world settings for low graphics
        settings = world.get_settings()
        settings.no_rendering_mode = True  # Disable rendering for max performance
        if args.sync_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Setup traffic manager
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_synchronous_mode(args.sync_mode)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        
        # Initialize V2V communication manager
        v2v_manager = None
        if args.enable_v2v:
            print("Initializing V2V communication system...")
            v2v_manager = V2VCommunicationManager(
                communication_range=args.v2v_range,
                latency=0.05,
                packet_loss_rate=0.02
            )
        
        # Spawn autonomous vehicles
        print(f"Spawning {args.num_vehicles} autonomous vehicles...")
        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        vehicle_blueprints = [bp for bp in vehicle_blueprints if int(bp.get_attribute('number_of_wheels')) == 4]
        
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        for i in range(args.num_vehicles):
            if i >= len(spawn_points):
                print(f"Warning: Not enough spawn points. Spawned {i} vehicles.")
                break
            blueprint = random.choice(vehicle_blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', f'autopilot_{i}')
            vehicle = world.spawn_actor(blueprint, spawn_points[i])
            # Reduce lag: lower camera/lidar resolution if possible
            # (handled in attach_sensors)
            av = AutonomousVehicle(
                vehicle_actor=vehicle,
                world=world,
                v2v_manager=v2v_manager,
                enable_sensors=True,
                target_speed=args.target_speed
            )
            autonomous_vehicles.append(av)
            print(f"  Spawned vehicle {vehicle.id} at {spawn_points[i].location}")
        
        print(f"\nSimulation started with {len(autonomous_vehicles)} vehicles")
        print("Press Ctrl+C to stop\n")
        
        # Main simulation loop
        start_time = time.time()
        frame = 0
        
        while time.time() - start_time < args.duration:
            frame += 1
            
            if args.sync_mode:
                world.tick()
            else:
                time.sleep(0.05)
            
            # Update all autonomous vehicles
            for av in autonomous_vehicles:
                av.update()
            
            # Display information
            if display and frame % 10 == 0:  # Update display every 10 frames
                display.fill((0, 0, 0))
                
                y_offset = 20
                title = font.render('Tesla Autonomous Driving System - V2V Enabled' if args.enable_v2v else 'Tesla Autonomous Driving System', True, (255, 255, 255))
                display.blit(title, (20, y_offset))
                y_offset += 40
                
                for av in autonomous_vehicles:
                    status = av.get_status()
                    target_speed = status.get('target_speed', 30.0)  # Default to 30 km/h
                    obstacles_detected = status.get('obstacles_detected', 0)
                    text = f"Vehicle {status['id']}: Speed={status['speed']:.1f} km/h | Target={target_speed:.1f} km/h | Obstacles={obstacles_detected}"
                    color = (100, 255, 100)
                    surface = font.render(text, True, color)
                    display.blit(surface, (20, y_offset))
                    y_offset += 30
                
                pygame.display.flip()
            
            # Print status periodically
            if frame % 200 == 0:  # Every 10 seconds
                elapsed = time.time() - start_time
                print(f"\n[{elapsed:.1f}s] Status:")
                for av in autonomous_vehicles:
                    status = av.get_status()
                    print(f"  Vehicle {status['id']}: {status['speed']:.1f} km/h, "
                          f"Obstacles: {status['obstacles_detected']}, "
                          f"Collisions: {status['metrics']['collisions']}")
        
        print("\nSimulation completed successfully!")
        
        # Print final metrics
        print("\n" + "="*60)
        print("FINAL METRICS")
        print("="*60)
        for av in autonomous_vehicles:
            status = av.get_status()
            metrics = status['metrics']
            print(f"\nVehicle {status['id']}:")
            print(f"  Total Distance: {metrics['total_distance']:.2f} m")
            print(f"  Average Speed: {metrics['avg_speed']*3.6:.2f} km/h")
            print(f"  Collisions: {metrics['collisions']}")
            print(f"  Red Light Violations: {metrics['red_light_violations']}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        
        for av in autonomous_vehicles:
            try:
                av.cleanup()
                if av.vehicle.is_alive:
                    av.vehicle.destroy()
            except:
                pass
        
        if client:
            # Restore world settings
            try:
                world = client.get_world()
                settings = world.get_settings()
                settings.synchronous_mode = False
                world.apply_settings(settings)
            except:
                pass
        
        if display:
            pygame.quit()
        
        print("Cleanup complete")


if __name__ == '__main__':
    main()
