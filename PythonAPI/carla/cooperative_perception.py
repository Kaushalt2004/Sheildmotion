"""
Cooperative Perception Module for V2V Autonomous Driving

This module implements cooperative perception using V2V communication to:
- Share sensor detections between vehicles
- Fuse data from multiple vehicles
- Extend perception range beyond individual sensors
- Improve object detection confidence and reduce false positives
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time


class ObjectType(Enum):
    """Types of detected objects"""
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    OBSTACLE = "obstacle"
    TRAFFIC_LIGHT = "traffic_light"
    STOP_SIGN = "stop_sign"
    UNKNOWN = "unknown"


@dataclass
class DetectedObject:
    """Represents a detected object from sensors or V2V"""
    object_id: int
    object_type: ObjectType
    position: np.ndarray  # [x, y, z]
    velocity: Optional[np.ndarray] = None  # [vx, vy, vz]
    dimensions: Optional[np.ndarray] = None  # [length, width, height]
    confidence: float = 1.0  # Detection confidence 0.0 to 1.0
    detection_time: float = 0.0
    source_vehicle_id: int = -1  # ID of vehicle that detected this object
    sensor_type: str = "unknown"  # lidar, camera, radar, v2v
    
    def get_distance_to(self, position: np.ndarray) -> float:
        """Calculate distance from this object to a given position"""
        return np.linalg.norm(self.position - position)


@dataclass
class FusedObject:
    """Fused object from multiple detections"""
    object_id: int
    object_type: ObjectType
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    dimensions: Optional[np.ndarray] = None
    confidence: float = 0.0
    num_detections: int = 1
    detection_sources: Set[int] = field(default_factory=set)  # Vehicle IDs that detected this
    last_update_time: float = 0.0
    
    def update_from_detection(self, detection: DetectedObject, fusion_weight: float = 0.3):
        """
        Update fused object with new detection using weighted average
        
        Args:
            detection: New detection to fuse
            fusion_weight: Weight for new detection (0.0 to 1.0)
        """
        # Update position with weighted average
        self.position = (1 - fusion_weight) * self.position + fusion_weight * detection.position
        
        # Update velocity if available
        if detection.velocity is not None:
            if self.velocity is None:
                self.velocity = detection.velocity.copy()
            else:
                self.velocity = (1 - fusion_weight) * self.velocity + fusion_weight * detection.velocity
        
        # Update dimensions if available
        if detection.dimensions is not None:
            if self.dimensions is None:
                self.dimensions = detection.dimensions.copy()
            else:
                self.dimensions = (1 - fusion_weight) * self.dimensions + fusion_weight * detection.dimensions
        
        # Increase confidence (capped at 1.0)
        self.confidence = min(1.0, self.confidence + detection.confidence * 0.1)
        
        # Update metadata
        self.num_detections += 1
        self.detection_sources.add(detection.source_vehicle_id)
        self.last_update_time = detection.detection_time


class CooperativePerceptionManager:
    """
    Manages cooperative perception using V2V communication
    Fuses sensor data from multiple vehicles to create a shared world model
    """
    
    def __init__(self,
                 association_threshold: float = 5.0,  # meters
                 confidence_threshold: float = 0.3,
                 object_timeout: float = 2.0):  # seconds
        """
        Initialize Cooperative Perception Manager
        
        Args:
            association_threshold: Maximum distance to associate detections (meters)
            confidence_threshold: Minimum confidence for valid detections
            object_timeout: Time after which unupdated objects are removed (seconds)
        """
        self.association_threshold = association_threshold
        self.confidence_threshold = confidence_threshold
        self.object_timeout = object_timeout
        
        # Fused object database - shared world model
        self.fused_objects: Dict[int, FusedObject] = {}
        self.next_object_id = 1
        
        # Per-vehicle detection history
        self.vehicle_detections: Dict[int, List[DetectedObject]] = {}
        
        # Statistics
        self.total_detections_processed = 0
        self.total_objects_fused = 0
        self.fusion_improvements = 0  # Times V2V improved detection
    
    def add_local_detection(self,
                           vehicle_id: int,
                           object_type: ObjectType,
                           position: np.ndarray,
                           velocity: Optional[np.ndarray] = None,
                           dimensions: Optional[np.ndarray] = None,
                           confidence: float = 0.8,
                           sensor_type: str = "lidar") -> DetectedObject:
        """
        Add a detection from local sensors
        
        Args:
            vehicle_id: ID of vehicle making the detection
            object_type: Type of detected object
            position: Object position [x, y, z]
            velocity: Object velocity [vx, vy, vz] (optional)
            dimensions: Object dimensions [length, width, height] (optional)
            confidence: Detection confidence
            sensor_type: Type of sensor (lidar, camera, radar)
            
        Returns:
            DetectedObject instance
        """
        detection = DetectedObject(
            object_id=self.next_object_id,
            object_type=object_type,
            position=position.copy() if isinstance(position, np.ndarray) else np.array(position),
            velocity=velocity.copy() if velocity is not None and isinstance(velocity, np.ndarray) else 
                     (np.array(velocity) if velocity is not None else None),
            dimensions=dimensions.copy() if dimensions is not None and isinstance(dimensions, np.ndarray) else
                      (np.array(dimensions) if dimensions is not None else None),
            confidence=confidence,
            detection_time=time.time(),
            source_vehicle_id=vehicle_id,
            sensor_type=sensor_type
        )
        
        self.next_object_id += 1
        self.total_detections_processed += 1
        
        # Store in vehicle's detection history
        if vehicle_id not in self.vehicle_detections:
            self.vehicle_detections[vehicle_id] = []
        self.vehicle_detections[vehicle_id].append(detection)
        
        # Fuse with existing objects
        self._fuse_detection(detection)
        
        return detection
    
    def add_v2v_detections(self,
                          vehicle_id: int,
                          detections: List[Dict]) -> List[DetectedObject]:
        """
        Add detections received via V2V communication
        
        Args:
            vehicle_id: ID of vehicle that originally detected these objects
            detections: List of detection dictionaries from V2V message
            
        Returns:
            List of DetectedObject instances
        """
        detected_objects = []
        
        for det_dict in detections:
            detection = DetectedObject(
                object_id=det_dict.get('object_id', self.next_object_id),
                object_type=ObjectType(det_dict.get('object_type', 'unknown')),
                position=np.array(det_dict['position']),
                velocity=np.array(det_dict['velocity']) if 'velocity' in det_dict and det_dict['velocity'] else None,
                dimensions=np.array(det_dict['dimensions']) if 'dimensions' in det_dict and det_dict['dimensions'] else None,
                confidence=det_dict.get('confidence', 0.7),
                detection_time=det_dict.get('detection_time', time.time()),
                source_vehicle_id=vehicle_id,
                sensor_type='v2v'
            )
            
            self.next_object_id += 1
            self.total_detections_processed += 1
            
            detected_objects.append(detection)
            self._fuse_detection(detection)
        
        return detected_objects
    
    def _fuse_detection(self, detection: DetectedObject):
        """
        Fuse a new detection with existing fused objects
        
        Args:
            detection: New detection to fuse
        """
        if detection.confidence < self.confidence_threshold:
            return
        
        # Try to associate with existing fused objects
        best_match_id = None
        best_match_distance = float('inf')
        
        for obj_id, fused_obj in self.fused_objects.items():
            # Only match same object types
            if fused_obj.object_type != detection.object_type:
                continue
            
            distance = np.linalg.norm(fused_obj.position - detection.position)
            
            if distance < self.association_threshold and distance < best_match_distance:
                best_match_distance = distance
                best_match_id = obj_id
        
        if best_match_id is not None:
            # Update existing fused object
            self.fused_objects[best_match_id].update_from_detection(detection)
            
            # Track if V2V improved detection
            if detection.sensor_type == 'v2v':
                self.fusion_improvements += 1
        else:
            # Create new fused object
            new_fused = FusedObject(
                object_id=detection.object_id,
                object_type=detection.object_type,
                position=detection.position.copy(),
                velocity=detection.velocity.copy() if detection.velocity is not None else None,
                dimensions=detection.dimensions.copy() if detection.dimensions is not None else None,
                confidence=detection.confidence,
                num_detections=1,
                detection_sources={detection.source_vehicle_id},
                last_update_time=detection.detection_time
            )
            
            self.fused_objects[detection.object_id] = new_fused
            self.total_objects_fused += 1
    
    def get_fused_objects(self,
                         vehicle_position: Optional[np.ndarray] = None,
                         max_distance: Optional[float] = None,
                         min_confidence: Optional[float] = None) -> List[FusedObject]:
        """
        Get all fused objects, optionally filtered
        
        Args:
            vehicle_position: Filter by distance from this position
            max_distance: Maximum distance from vehicle_position
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of FusedObject instances
        """
        # Remove stale objects
        current_time = time.time()
        stale_ids = [
            obj_id for obj_id, obj in self.fused_objects.items()
            if current_time - obj.last_update_time > self.object_timeout
        ]
        for obj_id in stale_ids:
            del self.fused_objects[obj_id]
        
        # Filter objects
        filtered_objects = []
        
        for fused_obj in self.fused_objects.values():
            # Check confidence
            if min_confidence and fused_obj.confidence < min_confidence:
                continue
            
            # Check distance
            if vehicle_position is not None and max_distance is not None:
                distance = np.linalg.norm(fused_obj.position - vehicle_position)
                if distance > max_distance:
                    continue
            
            filtered_objects.append(fused_obj)
        
        return filtered_objects
    
    def get_objects_in_path(self,
                           vehicle_position: np.ndarray,
                           vehicle_heading: float,
                           path_width: float = 3.5,
                           look_ahead_distance: float = 50.0) -> List[FusedObject]:
        """
        Get objects in the vehicle's path
        
        Args:
            vehicle_position: Vehicle position [x, y, z]
            vehicle_heading: Vehicle heading in degrees
            path_width: Width of the path to check (meters)
            look_ahead_distance: How far ahead to look (meters)
            
        Returns:
            List of FusedObject instances in the path
        """
        heading_rad = np.radians(vehicle_heading)
        forward_vec = np.array([np.cos(heading_rad), np.sin(heading_rad), 0])
        
        objects_in_path = []
        
        for fused_obj in self.fused_objects.values():
            rel_pos = fused_obj.position - vehicle_position
            
            # Check if object is ahead
            forward_distance = np.dot(rel_pos[:2], forward_vec[:2])
            if forward_distance < 0 or forward_distance > look_ahead_distance:
                continue
            
            # Check lateral distance (distance from path centerline)
            lateral_distance = abs(np.cross(rel_pos[:2], forward_vec[:2]))
            if lateral_distance <= path_width / 2:
                objects_in_path.append(fused_obj)
        
        # Sort by distance
        objects_in_path.sort(key=lambda obj: np.linalg.norm(obj.position - vehicle_position))
        
        return objects_in_path
    
    def find_closest_object(self,
                           vehicle_position: np.ndarray,
                           object_type: Optional[ObjectType] = None) -> Optional[Tuple[FusedObject, float]]:
        """
        Find the closest object to a position
        
        Args:
            vehicle_position: Reference position
            object_type: Filter by object type (optional)
            
        Returns:
            Tuple of (FusedObject, distance) or None
        """
        closest_obj = None
        min_distance = float('inf')
        
        for fused_obj in self.fused_objects.values():
            if object_type and fused_obj.object_type != object_type:
                continue
            
            distance = np.linalg.norm(fused_obj.position - vehicle_position)
            if distance < min_distance:
                min_distance = distance
                closest_obj = fused_obj
        
        if closest_obj:
            return (closest_obj, min_distance)
        return None
    
    def predict_object_position(self,
                               object_id: int,
                               time_horizon: float) -> Optional[np.ndarray]:
        """
        Predict future position of an object
        
        Args:
            object_id: ID of the object
            time_horizon: Time into the future (seconds)
            
        Returns:
            Predicted position [x, y, z] or None
        """
        if object_id not in self.fused_objects:
            return None
        
        obj = self.fused_objects[object_id]
        
        if obj.velocity is None:
            return obj.position
        
        # Simple constant velocity prediction
        predicted_pos = obj.position + obj.velocity * time_horizon
        
        return predicted_pos
    
    def check_collision_risk(self,
                            vehicle_position: np.ndarray,
                            vehicle_velocity: np.ndarray,
                            time_horizon: float = 3.0,
                            safety_margin: float = 2.0) -> List[Tuple[FusedObject, float]]:
        """
        Check for potential collisions with detected objects
        
        Args:
            vehicle_position: Current vehicle position
            vehicle_velocity: Current vehicle velocity
            time_horizon: How far ahead to check (seconds)
            safety_margin: Safety buffer distance (meters)
            
        Returns:
            List of (FusedObject, time_to_collision) for risky objects
        """
        risks = []
        
        for fused_obj in self.fused_objects.values():
            if fused_obj.velocity is None:
                # Stationary object - check if we're heading towards it
                rel_pos = fused_obj.position - vehicle_position
                distance = np.linalg.norm(rel_pos)
                
                if distance < 50.0:  # Only check nearby objects
                    # Project velocity onto relative position
                    if np.linalg.norm(vehicle_velocity) > 0:
                        time_to_closest = np.dot(rel_pos, vehicle_velocity) / np.dot(vehicle_velocity, vehicle_velocity)
                        
                        if 0 < time_to_closest < time_horizon:
                            closest_pos = vehicle_position + vehicle_velocity * time_to_closest
                            closest_distance = np.linalg.norm(closest_pos - fused_obj.position)
                            
                            if closest_distance < safety_margin:
                                risks.append((fused_obj, time_to_closest))
            else:
                # Moving object - calculate relative motion
                rel_vel = fused_obj.velocity - vehicle_velocity
                rel_pos = fused_obj.position - vehicle_position
                
                rel_speed_squared = np.dot(rel_vel, rel_vel)
                if rel_speed_squared < 1e-6:
                    continue
                
                # Time to closest approach
                ttc = -np.dot(rel_pos, rel_vel) / rel_speed_squared
                
                if 0 < ttc < time_horizon:
                    closest_distance = np.linalg.norm(rel_pos + rel_vel * ttc)
                    
                    if closest_distance < safety_margin:
                        risks.append((fused_obj, ttc))
        
        # Sort by time to collision
        risks.sort(key=lambda x: x[1])
        
        return risks
    
    def create_detection_message(self, vehicle_id: int) -> List[Dict]:
        """
        Create a list of detections to share via V2V
        
        Args:
            vehicle_id: ID of vehicle sharing detections
            
        Returns:
            List of detection dictionaries for V2V message
        """
        if vehicle_id not in self.vehicle_detections:
            return []
        
        detections = []
        current_time = time.time()
        
        # Share recent detections (last 1 second)
        for detection in self.vehicle_detections[vehicle_id]:
            if current_time - detection.detection_time < 1.0:
                det_dict = {
                    'object_id': detection.object_id,
                    'object_type': detection.object_type.value,
                    'position': detection.position.tolist(),
                    'confidence': detection.confidence,
                    'detection_time': detection.detection_time
                }
                
                if detection.velocity is not None:
                    det_dict['velocity'] = detection.velocity.tolist()
                
                if detection.dimensions is not None:
                    det_dict['dimensions'] = detection.dimensions.tolist()
                
                detections.append(det_dict)
        
        return detections
    
    def get_perception_statistics(self) -> Dict:
        """Get statistics about cooperative perception"""
        return {
            'total_detections_processed': self.total_detections_processed,
            'total_objects_fused': self.total_objects_fused,
            'current_tracked_objects': len(self.fused_objects),
            'fusion_improvements': self.fusion_improvements,
            'average_confidence': np.mean([obj.confidence for obj in self.fused_objects.values()]) 
                                 if self.fused_objects else 0.0,
            'multi_source_objects': sum(1 for obj in self.fused_objects.values() if len(obj.detection_sources) > 1)
        }
    
    def clear_vehicle_detections(self, vehicle_id: int):
        """Clear detection history for a vehicle"""
        if vehicle_id in self.vehicle_detections:
            self.vehicle_detections[vehicle_id].clear()
    
    def reset(self):
        """Reset the cooperative perception system"""
        self.fused_objects.clear()
        self.vehicle_detections.clear()
        self.next_object_id = 1
        self.total_detections_processed = 0
        self.total_objects_fused = 0
        self.fusion_improvements = 0
