"""
Vehicle-to-Vehicle (V2V) Communication System for Autonomous Driving

This module implements a V2V communication protocol that allows vehicles to share:
- Position and velocity information
- Sensor data (detected objects, obstacles)
- Intentions (lane changes, turns, braking)
- Traffic information
- Emergency alerts
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class MessageType(Enum):
    """Types of V2V messages"""
    BASIC_SAFETY = "basic_safety"  # BSM - Basic Safety Message
    COOPERATIVE_AWARENESS = "cooperative_awareness"  # CAM
    COLLECTIVE_PERCEPTION = "collective_perception"  # CPM
    EMERGENCY_WARNING = "emergency_warning"
    LANE_CHANGE_INTENT = "lane_change_intent"
    TRAFFIC_CONDITION = "traffic_condition"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 0  # Emergency braking, collision warning
    HIGH = 1      # Lane change, traffic light
    MEDIUM = 2    # Cooperative awareness
    LOW = 3       # Traffic updates


@dataclass
class V2VMessage:
    """Structure for V2V communication messages"""
    sender_id: int
    timestamp: float
    message_type: MessageType
    priority: MessagePriority
    position: List[float]  # [x, y, z]
    velocity: List[float]  # [vx, vy, vz]
    acceleration: List[float]  # [ax, ay, az]
    heading: float  # yaw angle in degrees
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary for transmission"""
        return {
            'sender_id': self.sender_id,
            'timestamp': self.timestamp,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'heading': self.heading,
            'data': self.data
        }
    
    @staticmethod
    def from_dict(msg_dict: Dict) -> 'V2VMessage':
        """Create message from dictionary"""
        return V2VMessage(
            sender_id=msg_dict['sender_id'],
            timestamp=msg_dict['timestamp'],
            message_type=MessageType(msg_dict['message_type']),
            priority=MessagePriority(msg_dict['priority']),
            position=msg_dict['position'],
            velocity=msg_dict['velocity'],
            acceleration=msg_dict['acceleration'],
            heading=msg_dict['heading'],
            data=msg_dict.get('data', {})
        )


class V2VCommunicationManager:
    """
    Manages V2V communication between autonomous vehicles
    Implements DSRC/C-V2X communication protocols with configurable range and latency
    """
    
    def __init__(self, 
                 communication_range: float = 300.0,  # meters
                 latency: float = 0.05,  # seconds (50ms)
                 packet_loss_rate: float = 0.02,  # 2% packet loss
                 bandwidth: float = 27.0):  # Mbps for DSRC
        """
        Initialize V2V Communication Manager
        
        Args:
            communication_range: Maximum communication range in meters
            latency: Communication latency in seconds
            packet_loss_rate: Probability of packet loss (0.0 to 1.0)
            bandwidth: Available bandwidth in Mbps
        """
        self.communication_range = communication_range
        self.latency = latency
        self.packet_loss_rate = packet_loss_rate
        self.bandwidth = bandwidth
        
        # Storage for messages
        self.message_buffer: Dict[int, List[V2VMessage]] = {}  # vehicle_id -> messages
        self.message_history: Dict[int, List[V2VMessage]] = {}  # vehicle_id -> sent messages
        
        # Network statistics
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.messages_dropped = 0
        
        # Track all known vehicle positions
        self.vehicle_positions: Dict[int, List[float]] = {}
    
    def broadcast_message(self, message: V2VMessage) -> int:
        """
        Simplified broadcast - automatically manages vehicle positions
        
        Args:
            message: V2VMessage to broadcast
            
        Returns:
            Number of vehicles that received the message
        """
        # Update sender position
        self.vehicle_positions[message.sender_id] = message.position
        
        return self._broadcast_to_range(message)
    
    def _broadcast_to_range(self, message: V2VMessage) -> int:
        """
        Internal method to broadcast message to vehicles in range
        """
        self.total_messages_sent += 1
        sender_id = message.sender_id
        
        # Store in sender's history
        if sender_id not in self.message_history:
            self.message_history[sender_id] = []
        self.message_history[sender_id].append(message)
        
        # Calculate which vehicles are in range
        sender_pos = np.array(message.position)
        receivers = []
        
        for vehicle_id, position in self.vehicle_positions.items():
            if vehicle_id == sender_id:
                continue
                
            distance = np.linalg.norm(np.array(position) - sender_pos)
            
            if distance <= self.communication_range:
                # Simulate packet loss
                if np.random.random() > self.packet_loss_rate:
                    receivers.append(vehicle_id)
                else:
                    self.messages_dropped += 1
        
        # Deliver messages to receivers with simulated latency
        for receiver_id in receivers:
            if receiver_id not in self.message_buffer:
                self.message_buffer[receiver_id] = []
            
            # Add delivery time for latency simulation
            delayed_message = V2VMessage(
                sender_id=message.sender_id,
                timestamp=message.timestamp + self.latency,
                message_type=message.message_type,
                priority=message.priority,
                position=message.position.copy(),
                velocity=message.velocity.copy(),
                acceleration=message.acceleration.copy(),
                heading=message.heading,
                data=message.data.copy()
            )
            self.message_buffer[receiver_id].append(delayed_message)
            self.total_messages_received += 1
        
        return len(receivers)
        
    def broadcast_message_legacy(self, 
                         sender_id: int,
                         message: V2VMessage,
                         vehicle_positions: Dict[int, List[float]]) -> int:
        """
        Broadcast a message to all vehicles within communication range
        
        Args:
            sender_id: ID of the sending vehicle
            message: V2VMessage to broadcast
            vehicle_positions: Dictionary of vehicle_id -> [x, y, z] positions
            
        Returns:
            Number of vehicles that received the message
        """
        self.total_messages_sent += 1
        
        # Store in sender's history
        if sender_id not in self.message_history:
            self.message_history[sender_id] = []
        self.message_history[sender_id].append(message)
        
        # Calculate which vehicles are in range
        sender_pos = np.array(message.position)
        receivers = []
        
        for vehicle_id, position in vehicle_positions.items():
            if vehicle_id == sender_id:
                continue
                
            distance = np.linalg.norm(np.array(position) - sender_pos)
            
            if distance <= self.communication_range:
                # Simulate packet loss
                if np.random.random() > self.packet_loss_rate:
                    receivers.append(vehicle_id)
                else:
                    self.messages_dropped += 1
        
        # Deliver messages to receivers with simulated latency
        for receiver_id in receivers:
            if receiver_id not in self.message_buffer:
                self.message_buffer[receiver_id] = []
            
            # Add delivery time for latency simulation
            delayed_message = V2VMessage(
                sender_id=message.sender_id,
                timestamp=message.timestamp + self.latency,
                message_type=message.message_type,
                priority=message.priority,
                position=message.position,
                velocity=message.velocity,
                acceleration=message.acceleration,
                heading=message.heading,
                data=message.data.copy()
            )
            self.message_buffer[receiver_id].append(delayed_message)
            self.total_messages_received += 1
        
        return len(receivers)
    
    def receive_messages(self, 
                        vehicle_id: int, 
                        current_time: float,
                        max_age: float = 1.0) -> List[V2VMessage]:
        """
        Retrieve messages for a specific vehicle
        
        Args:
            vehicle_id: ID of the receiving vehicle
            current_time: Current simulation time
            max_age: Maximum age of messages to return (seconds)
            
        Returns:
            List of received messages
        """
        if vehicle_id not in self.message_buffer:
            return []
        
        # Filter messages by age and delivery time
        valid_messages = []
        for msg in self.message_buffer[vehicle_id]:
            age = current_time - msg.timestamp
            if age >= 0 and age <= max_age:  # Message has been delivered and not too old
                valid_messages.append(msg)
        
        # Clear processed messages
        self.message_buffer[vehicle_id] = []
        
        return valid_messages
    
    def get_nearby_vehicles(self,
                           vehicle_id: int,
                           vehicle_positions: Dict[int, List[float]],
                           max_range: Optional[float] = None) -> Dict[int, float]:
        """
        Get all vehicles within communication range with their distances
        
        Args:
            vehicle_id: ID of the reference vehicle
            vehicle_positions: Dictionary of all vehicle positions
            max_range: Maximum range to consider (defaults to communication_range)
            
        Returns:
            Dictionary of vehicle_id -> distance
        """
        if vehicle_id not in vehicle_positions:
            return {}
        
        max_range = max_range or self.communication_range
        ego_pos = np.array(vehicle_positions[vehicle_id])
        nearby = {}
        
        for other_id, position in vehicle_positions.items():
            if other_id == vehicle_id:
                continue
            
            distance = np.linalg.norm(np.array(position) - ego_pos)
            if distance <= max_range:
                nearby[other_id] = distance
        
        return nearby
    
    def create_basic_safety_message(self,
                                    vehicle_id: int,
                                    position: List[float],
                                    velocity: List[float],
                                    acceleration: List[float],
                                    heading: float,
                                    additional_data: Optional[Dict] = None) -> V2VMessage:
        """
        Create a Basic Safety Message (BSM) - most common V2V message
        
        Args:
            vehicle_id: Sender vehicle ID
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            acceleration: Current acceleration [ax, ay, az]
            heading: Current heading (yaw) in degrees
            additional_data: Optional additional information
            
        Returns:
            V2VMessage with basic safety information
        """
        data = additional_data or {}
        data.update({
            'speed': np.linalg.norm(velocity),
            'acceleration_magnitude': np.linalg.norm(acceleration)
        })
        
        return V2VMessage(
            sender_id=vehicle_id,
            timestamp=time.time(),
            message_type=MessageType.BASIC_SAFETY,
            priority=MessagePriority.HIGH,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            heading=heading,
            data=data
        )
    
    def create_cooperative_perception_message(self,
                                             vehicle_id: int,
                                             position: List[float],
                                             velocity: List[float],
                                             acceleration: List[float],
                                             heading: float,
                                             detected_objects: List[Dict]) -> V2VMessage:
        """
        Create a Cooperative Perception Message (CPM)
        Shares sensor data about detected objects
        
        Args:
            vehicle_id: Sender vehicle ID
            position: Current position
            velocity: Current velocity
            acceleration: Current acceleration
            heading: Current heading
            detected_objects: List of detected objects with their properties
            
        Returns:
            V2VMessage with perception data
        """
        return V2VMessage(
            sender_id=vehicle_id,
            timestamp=time.time(),
            message_type=MessageType.COLLECTIVE_PERCEPTION,
            priority=MessagePriority.MEDIUM,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            heading=heading,
            data={'detected_objects': detected_objects}
        )
    
    def create_emergency_message(self,
                                vehicle_id: int,
                                position: List[float],
                                velocity: List[float],
                                acceleration: List[float],
                                heading: float,
                                emergency_type: str,
                                severity: int) -> V2VMessage:
        """
        Create an Emergency Warning Message
        
        Args:
            vehicle_id: Sender vehicle ID
            position: Current position
            velocity: Current velocity
            acceleration: Current acceleration
            heading: Current heading
            emergency_type: Type of emergency (e.g., "hard_braking", "collision")
            severity: Severity level (1-5, 5 being most severe)
            
        Returns:
            V2VMessage with emergency information
        """
        return V2VMessage(
            sender_id=vehicle_id,
            timestamp=time.time(),
            message_type=MessageType.EMERGENCY_WARNING,
            priority=MessagePriority.CRITICAL,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            heading=heading,
            data={
                'emergency_type': emergency_type,
                'severity': severity
            }
        )
    
    def create_lane_change_intent_message(self,
                                         vehicle_id: int,
                                         position: List[float],
                                         velocity: List[float],
                                         acceleration: List[float],
                                         heading: float,
                                         target_lane: int,
                                         time_to_execute: float) -> V2VMessage:
        """
        Create a Lane Change Intent Message
        
        Args:
            vehicle_id: Sender vehicle ID
            position: Current position
            velocity: Current velocity
            acceleration: Current acceleration
            heading: Current heading
            target_lane: Target lane ID
            time_to_execute: Estimated time until lane change (seconds)
            
        Returns:
            V2VMessage with lane change intent
        """
        return V2VMessage(
            sender_id=vehicle_id,
            timestamp=time.time(),
            message_type=MessageType.LANE_CHANGE_INTENT,
            priority=MessagePriority.HIGH,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            heading=heading,
            data={
                'target_lane': target_lane,
                'time_to_execute': time_to_execute
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication network statistics"""
        total_attempted = self.total_messages_sent
        success_rate = (self.total_messages_received / total_attempted * 100 
                       if total_attempted > 0 else 0)
        loss_rate = (self.messages_dropped / total_attempted * 100 
                    if total_attempted > 0 else 0)
        
        return {
            'total_messages_sent': self.total_messages_sent,
            'total_messages_received': self.total_messages_received,
            'messages_dropped': self.messages_dropped,
            'success_rate': success_rate,
            'packet_loss_rate': loss_rate,
            'communication_range': self.communication_range,
            'average_latency': self.latency
        }
    
    def reset_statistics(self):
        """Reset communication statistics"""
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.messages_dropped = 0
