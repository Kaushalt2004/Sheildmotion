"""
Vehicle State Manager for Autonomous Driving

Manages the state and sensor data for autonomous vehicles in the simulation.
Tracks position, velocity, acceleration, detected objects, and driving context.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import carla


class DrivingState(Enum):
    """Current driving state of the vehicle"""
    IDLE = "idle"
    FOLLOWING = "following"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    STOPPING = "stopping"
    EMERGENCY_BRAKE = "emergency_brake"
    OVERTAKING = "overtaking"


@dataclass
class VehicleState:
    """Complete state information for an autonomous vehicle"""
    vehicle_id: int
    timestamp: float
    
    # Position and orientation
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [roll, pitch, yaw]
    
    # Dynamics
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    
    # Control inputs
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    
    # Driving context
    driving_state: DrivingState = DrivingState.IDLE
    current_lane_id: Optional[int] = None
    current_road_id: Optional[int] = None
    target_lane_id: Optional[int] = None
    target_speed: float = 30.0  # km/h
    
    # Sensor data
    detected_vehicles: List[Dict] = field(default_factory=list)
    detected_pedestrians: List[Dict] = field(default_factory=list)
    detected_obstacles: List[Dict] = field(default_factory=list)
    detected_traffic_lights: List[Dict] = field(default_factory=list)
    
    # Safety metrics
    time_to_collision: Optional[float] = None
    minimum_safe_distance: float = 5.0
    collision_risk_level: float = 0.0  # 0.0 to 1.0
    
    def get_speed(self) -> float:
        """Get current speed in km/h"""
        return np.linalg.norm(self.velocity) * 3.6
    
    def get_heading(self) -> float:
        """Get current heading (yaw) in degrees"""
        return self.rotation[2]


class VehicleStateManager:
    """
    Manages state information for all autonomous vehicles in the simulation
    Provides state updates, predictions, and safety checks
    """
    
    def __init__(self, world: carla.World):
        """
        Initialize Vehicle State Manager
        
        Args:
            world: CARLA world object
        """
        self.world = world
        self.carla_map = world.get_map()
        
        # Vehicle states
        self.vehicle_states: Dict[int, VehicleState] = {}
        self.vehicle_actors: Dict[int, carla.Vehicle] = {}
        
        # State history for prediction
        self.state_history: Dict[int, List[VehicleState]] = {}
        self.history_length = 10  # Number of past states to keep
        
    def register_vehicle(self, vehicle: carla.Vehicle) -> int:
        """
        Register a new autonomous vehicle
        
        Args:
            vehicle: CARLA vehicle actor
            
        Returns:
            Vehicle ID
        """
        vehicle_id = vehicle.id
        self.vehicle_actors[vehicle_id] = vehicle
        
        # Initialize state
        self._update_vehicle_state(vehicle_id)
        
        # Initialize history
        if vehicle_id not in self.state_history:
            self.state_history[vehicle_id] = []
        
        return vehicle_id
    
    def unregister_vehicle(self, vehicle_id: int):
        """Remove a vehicle from management"""
        if vehicle_id in self.vehicle_actors:
            del self.vehicle_actors[vehicle_id]
        if vehicle_id in self.vehicle_states:
            del self.vehicle_states[vehicle_id]
        if vehicle_id in self.state_history:
            del self.state_history[vehicle_id]
    
    def update_all_states(self, current_time: float):
        """
        Update states for all registered vehicles
        
        Args:
            current_time: Current simulation timestamp
        """
        for vehicle_id in list(self.vehicle_actors.keys()):
            self._update_vehicle_state(vehicle_id, current_time)
    
    def _update_vehicle_state(self, vehicle_id: int, current_time: Optional[float] = None):
        """Update state for a specific vehicle"""
        if vehicle_id not in self.vehicle_actors:
            return
        
        vehicle = self.vehicle_actors[vehicle_id]
        
        # Get transform and dynamics
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        angular_velocity = vehicle.get_angular_velocity()
        control = vehicle.get_control()
        
        # Get waypoint information
        waypoint = self.carla_map.get_waypoint(transform.location)
        
        # Convert to numpy arrays
        position = np.array([transform.location.x, transform.location.y, transform.location.z])
        rotation = np.array([transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw])
        velocity_vec = np.array([velocity.x, velocity.y, velocity.z])
        acceleration_vec = np.array([acceleration.x, acceleration.y, acceleration.z])
        angular_vel = np.array([angular_velocity.x, angular_velocity.y, angular_velocity.z])
        
        # Create or update state
        if vehicle_id in self.vehicle_states:
            # Store previous state in history
            prev_state = self.vehicle_states[vehicle_id]
            self.state_history[vehicle_id].append(prev_state)
            
            # Limit history length
            if len(self.state_history[vehicle_id]) > self.history_length:
                self.state_history[vehicle_id].pop(0)
        
        # Create new state
        state = VehicleState(
            vehicle_id=vehicle_id,
            timestamp=current_time or self.world.get_snapshot().timestamp.elapsed_seconds,
            position=position,
            rotation=rotation,
            velocity=velocity_vec,
            acceleration=acceleration_vec,
            angular_velocity=angular_vel,
            throttle=control.throttle,
            brake=control.brake,
            steer=control.steer,
            current_lane_id=waypoint.lane_id if waypoint else None,
            current_road_id=waypoint.road_id if waypoint else None
        )
        
        self.vehicle_states[vehicle_id] = state
    
    def get_state(self, vehicle_id: int) -> Optional[VehicleState]:
        """Get current state of a vehicle"""
        return self.vehicle_states.get(vehicle_id)
    
    def get_all_states(self) -> Dict[int, VehicleState]:
        """Get states of all vehicles"""
        return self.vehicle_states.copy()
    
    def get_vehicle_positions(self) -> Dict[int, List[float]]:
        """Get positions of all vehicles as a dictionary"""
        return {
            vid: state.position.tolist() 
            for vid, state in self.vehicle_states.items()
        }
    
    def predict_future_position(self, 
                               vehicle_id: int, 
                               time_horizon: float) -> Optional[np.ndarray]:
        """
        Predict future position of a vehicle using constant velocity model
        
        Args:
            vehicle_id: ID of the vehicle
            time_horizon: Time into the future (seconds)
            
        Returns:
            Predicted position [x, y, z] or None if vehicle not found
        """
        if vehicle_id not in self.vehicle_states:
            return None
        
        state = self.vehicle_states[vehicle_id]
        
        # Simple constant velocity prediction
        predicted_pos = state.position + state.velocity * time_horizon
        
        return predicted_pos
    
    def predict_future_trajectory(self,
                                  vehicle_id: int,
                                  time_horizon: float,
                                  num_points: int = 10) -> Optional[np.ndarray]:
        """
        Predict future trajectory of a vehicle
        
        Args:
            vehicle_id: ID of the vehicle
            time_horizon: Total time into the future (seconds)
            num_points: Number of trajectory points to generate
            
        Returns:
            Array of shape (num_points, 3) with predicted positions
        """
        if vehicle_id not in self.vehicle_states:
            return None
        
        state = self.vehicle_states[vehicle_id]
        trajectory = np.zeros((num_points, 3))
        
        dt = time_horizon / num_points
        
        for i in range(num_points):
            t = (i + 1) * dt
            # Constant velocity + constant acceleration model
            trajectory[i] = (state.position + 
                           state.velocity * t + 
                           0.5 * state.acceleration * t**2)
        
        return trajectory
    
    def calculate_time_to_collision(self,
                                    vehicle_id: int,
                                    other_vehicle_id: int) -> Optional[float]:
        """
        Calculate time to collision between two vehicles
        
        Args:
            vehicle_id: ID of first vehicle
            other_vehicle_id: ID of second vehicle
            
        Returns:
            Time to collision in seconds, or None if no collision predicted
        """
        if vehicle_id not in self.vehicle_states or other_vehicle_id not in self.vehicle_states:
            return None
        
        state1 = self.vehicle_states[vehicle_id]
        state2 = self.vehicle_states[other_vehicle_id]
        
        # Calculate relative position and velocity
        rel_pos = state2.position - state1.position
        rel_vel = state2.velocity - state1.velocity
        
        # Calculate time to closest approach
        rel_speed_squared = np.dot(rel_vel, rel_vel)
        
        if rel_speed_squared < 1e-6:  # Vehicles moving at similar speeds
            return None
        
        ttc = -np.dot(rel_pos, rel_vel) / rel_speed_squared
        
        # Only return positive TTC (collision in future)
        if ttc <= 0:
            return None
        
        # Calculate distance at closest approach
        closest_distance = np.linalg.norm(rel_pos + rel_vel * ttc)
        
        # Consider collision if distance < 5 meters
        if closest_distance < 5.0:
            return ttc
        
        return None
    
    def detect_nearby_vehicles(self,
                              vehicle_id: int,
                              max_distance: float = 50.0) -> List[Tuple[int, float, np.ndarray]]:
        """
        Detect vehicles near a given vehicle
        
        Args:
            vehicle_id: Reference vehicle ID
            max_distance: Maximum detection distance in meters
            
        Returns:
            List of tuples (other_vehicle_id, distance, relative_position)
        """
        if vehicle_id not in self.vehicle_states:
            return []
        
        ego_state = self.vehicle_states[vehicle_id]
        nearby = []
        
        for other_id, other_state in self.vehicle_states.items():
            if other_id == vehicle_id:
                continue
            
            relative_pos = other_state.position - ego_state.position
            distance = np.linalg.norm(relative_pos)
            
            if distance <= max_distance:
                nearby.append((other_id, distance, relative_pos))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        
        return nearby
    
    def get_vehicle_in_front(self,
                            vehicle_id: int,
                            max_distance: float = 50.0) -> Optional[Tuple[int, float]]:
        """
        Get the vehicle directly in front in the same lane
        
        Args:
            vehicle_id: Reference vehicle ID
            max_distance: Maximum look-ahead distance
            
        Returns:
            Tuple of (vehicle_id, distance) or None
        """
        if vehicle_id not in self.vehicle_states:
            return None
        
        ego_state = self.vehicle_states[vehicle_id]
        ego_heading = np.radians(ego_state.get_heading())
        ego_forward = np.array([np.cos(ego_heading), np.sin(ego_heading), 0])
        
        closest_front = None
        min_distance = float('inf')
        
        for other_id, other_state in self.vehicle_states.items():
            if other_id == vehicle_id:
                continue
            
            # Check if in same lane
            if (ego_state.current_lane_id != other_state.current_lane_id or
                ego_state.current_road_id != other_state.current_road_id):
                continue
            
            relative_pos = other_state.position - ego_state.position
            distance = np.linalg.norm(relative_pos[:2])  # Ignore z
            
            if distance > max_distance:
                continue
            
            # Check if vehicle is in front (dot product with forward vector)
            if np.dot(relative_pos[:2], ego_forward[:2]) > 0:
                if distance < min_distance:
                    min_distance = distance
                    closest_front = (other_id, distance)
        
        return closest_front
    
    def update_driving_state(self, vehicle_id: int, new_state: DrivingState):
        """Update the driving state of a vehicle"""
        if vehicle_id in self.vehicle_states:
            self.vehicle_states[vehicle_id].driving_state = new_state
    
    def set_target_speed(self, vehicle_id: int, target_speed: float):
        """Set target speed for a vehicle (km/h)"""
        if vehicle_id in self.vehicle_states:
            self.vehicle_states[vehicle_id].target_speed = target_speed
    
    def calculate_safe_following_distance(self, speed_kmh: float) -> float:
        """
        Calculate safe following distance based on speed
        Uses 2-second rule plus buffer
        
        Args:
            speed_kmh: Current speed in km/h
            
        Returns:
            Safe distance in meters
        """
        speed_ms = speed_kmh / 3.6
        return max(5.0, speed_ms * 2.0)  # Minimum 5 meters
    
    def is_safe_to_change_lane(self,
                              vehicle_id: int,
                              target_lane_offset: int) -> bool:
        """
        Check if it's safe to change lanes
        
        Args:
            vehicle_id: Vehicle attempting lane change
            target_lane_offset: -1 for left, +1 for right
            
        Returns:
            True if safe to change lanes
        """
        if vehicle_id not in self.vehicle_states:
            return False
        
        ego_state = self.vehicle_states[vehicle_id]
        ego_speed = ego_state.get_speed()
        safe_distance = self.calculate_safe_following_distance(ego_speed)
        
        # Check all other vehicles
        for other_id, other_state in self.vehicle_states.items():
            if other_id == vehicle_id:
                continue
            
            # Check if vehicle is in target lane
            if (other_state.current_road_id == ego_state.current_road_id and
                other_state.current_lane_id == ego_state.current_lane_id + target_lane_offset):
                
                # Calculate distance
                distance = np.linalg.norm(other_state.position - ego_state.position)
                
                if distance < safe_distance * 1.5:  # Extra buffer for lane change
                    return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed vehicles"""
        if not self.vehicle_states:
            return {
                'num_vehicles': 0,
                'average_speed': 0.0,
                'total_distance_traveled': 0.0
            }
        
        speeds = [state.get_speed() for state in self.vehicle_states.values()]
        
        return {
            'num_vehicles': len(self.vehicle_states),
            'average_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'min_speed': np.min(speeds),
            'total_vehicles_managed': len(self.vehicle_actors)
        }
