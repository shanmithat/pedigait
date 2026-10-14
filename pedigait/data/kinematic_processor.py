"""
Kinematic Processor Module

Processes kinematic data from C3D files and extracts joint trajectories.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal, interpolate


@dataclass
class JointTrajectory:
    """Container for joint angle trajectories."""
    name: str
    angles: np.ndarray  # Shape: (n_frames, 3) for x, y, z or sagittal, frontal, transverse
    time: np.ndarray  # Time array in seconds
    rate: float  # Sampling rate in Hz
    units: str = "degrees"


@dataclass
class KinematicData:
    """Container for processed kinematic data."""
    pelvis: JointTrajectory
    hip_left: JointTrajectory
    hip_right: JointTrajectory
    knee_left: JointTrajectory
    knee_right: JointTrajectory
    ankle_left: JointTrajectory
    ankle_right: JointTrajectory
    foot_left: Optional[JointTrajectory] = None
    foot_right: Optional[JointTrajectory] = None
    gait_cycle_percentage: Optional[np.ndarray] = None


class KinematicProcessor:
    """Process kinematic data and extract joint trajectories."""
    
    def __init__(self, point_rate: float = 100.0):
        """
        Initialize kinematic processor.
        
        Args:
            point_rate: Sampling rate in Hz
        """
        self.point_rate = point_rate
        
    def process_marker_data(self, markers: np.ndarray, 
                           labels: List[str]) -> KinematicData:
        """
        Process marker positions to extract joint angles.
        
        Args:
            markers: Array of shape (3, n_markers, n_frames) with x, y, z coordinates
            labels: List of marker labels
            
        Returns:
            KinematicData object with joint trajectories
        """
        n_frames = markers.shape[2]
        time = np.arange(n_frames) / self.point_rate
        
        # Extract joint-specific markers
        joint_markers = self._identify_joint_markers(labels)
        
        # Calculate joint angles
        pelvis = self._calculate_pelvis_angles(markers, joint_markers.get("pelvis", []))
        hip_left = self._calculate_hip_angles(markers, joint_markers.get("hip_left", []), side="left")
        hip_right = self._calculate_hip_angles(markers, joint_markers.get("hip_right", []), side="right")
        knee_left = self._calculate_knee_angles(markers, joint_markers.get("knee_left", []), side="left")
        knee_right = self._calculate_knee_angles(markers, joint_markers.get("knee_right", []), side="right")
        ankle_left = self._calculate_ankle_angles(markers, joint_markers.get("ankle_left", []), side="left")
        ankle_right = self._calculate_ankle_angles(markers, joint_markers.get("ankle_right", []), side="right")
        
        # Try to extract foot angles
        foot_left = self._calculate_foot_angles(markers, joint_markers.get("foot_left", []), side="left")
        foot_right = self._calculate_foot_angles(markers, joint_markers.get("foot_right", []), side="right")
        
        return KinematicData(
            pelvis=pelvis,
            hip_left=hip_left,
            hip_right=hip_right,
            knee_left=knee_left,
            knee_right=knee_right,
            ankle_left=ankle_left,
            ankle_right=ankle_right,
            foot_left=foot_left,
            foot_right=foot_right,
        )
    
    def _identify_joint_markers(self, labels: List[str]) -> Dict[str, List[int]]:
        """
        Identify markers associated with each joint.
        
        Args:
            labels: List of marker labels
            
        Returns:
            Dictionary mapping joint names to marker indices
        """
        # Common marker naming patterns
        marker_patterns = {
            "pelvis": ["SACR", "LASI", "RASI", "LPSI", "RPSI"],
            "hip_left": ["LASI", "LTHI", "LGTRO", "LHFE"],
            "hip_right": ["RASI", "RTHI", "RGTRO", "RHFE"],
            "knee_left": ["LKNE", "LFEME", "LFMH", "LTIB"],
            "knee_right": ["RKNE", "RFEME", "RFMH", "RTIB"],
            "ankle_left": ["LANK", "LHEE", "LMEH", "LTIB"],
            "ankle_right": ["RANK", "RHEE", "RMEH", "RTIB"],
            "foot_left": ["LHEE", "LTOE", "LMEH", "LMT5"],
            "foot_right": ["RHEE", "RTOE", "RMEH", "RMT5"],
        }
        
        joint_markers = {}
        
        for joint, patterns in marker_patterns.items():
            indices = []
            for i, label in enumerate(labels):
                label_upper = label.upper()
                for pattern in patterns:
                    if pattern in label_upper:
                        indices.append(i)
                        break
            joint_markers[joint] = indices
        
        return joint_markers
    
    def _calculate_pelvis_angles(self, markers: np.ndarray, 
                                marker_indices: List[int]) -> JointTrajectory:
        """Calculate pelvis orientation angles."""
        # Placeholder implementation
        n_frames = markers.shape[2]
        angles = np.zeros((n_frames, 3))  # Sagittal, frontal, transverse
        time = np.arange(n_frames) / self.point_rate
        
        return JointTrajectory(
            name="pelvis",
            angles=angles,
            time=time,
            rate=self.point_rate,
            units="degrees"
        )
    
    def _calculate_hip_angles(self, markers: np.ndarray, 
                             marker_indices: List[int], 
                             side: str) -> JointTrajectory:
        """Calculate hip joint angles."""
        n_frames = markers.shape[2]
        angles = np.zeros((n_frames, 3))
        time = np.arange(n_frames) / self.point_rate
        
        return JointTrajectory(
            name=f"hip_{side}",
            angles=angles,
            time=time,
            rate=self.point_rate,
            units="degrees"
        )
    
    def _calculate_knee_angles(self, markers: np.ndarray, 
                              marker_indices: List[int], 
                              side: str) -> JointTrajectory:
        """Calculate knee joint angles."""
        n_frames = markers.shape[2]
        angles = np.zeros((n_frames, 3))
        time = np.arange(n_frames) / self.point_rate
        
        return JointTrajectory(
            name=f"knee_{side}",
            angles=angles,
            time=time,
            rate=self.point_rate,
            units="degrees"
        )
    
    def _calculate_ankle_angles(self, markers: np.ndarray, 
                                marker_indices: List[int], 
                                side: str) -> JointTrajectory:
        """Calculate ankle joint angles."""
        n_frames = markers.shape[2]
        angles = np.zeros((n_frames, 3))
        time = np.arange(n_frames) / self.point_rate
        
        return JointTrajectory(
            name=f"ankle_{side}",
            angles=angles,
            time=time,
            rate=self.point_rate,
            units="degrees"
        )
    
    def _calculate_foot_angles(self, markers: np.ndarray, 
                              marker_indices: List[int], 
                              side: str) -> Optional[JointTrajectory]:
        """Calculate foot angles (progression, etc.)."""
        if not marker_indices:
            return None
        
        n_frames = markers.shape[2]
        angles = np.zeros((n_frames, 3))
        time = np.arange(n_frames) / self.point_rate
        
        return JointTrajectory(
            name=f"foot_{side}",
            angles=angles,
            time=time,
            rate=self.point_rate,
            units="degrees"
        )
    
    def extract_gait_cycle(self, kinematic_data: KinematicData, 
                          heel_strike_frames: List[int]) -> KinematicData:
        """
        Extract gait cycle data based on heel strike events.
        
        Args:
            kinematic_data: Processed kinematic data
            heel_strike_frames: List of heel strike frame indices
            
        Returns:
            KinematicData normalized to gait cycle percentage
        """
        # Normalize each gait cycle to 0-100%
        # This is a placeholder - actual implementation depends on event detection
        
        return kinematic_data
    
    def smooth_trajectories(self, kinematic_data: KinematicData, 
                           window_size: int = 5) -> KinematicData:
        """
        Apply smoothing to joint angle trajectories.
        
        Args:
            kinematic_data: Raw kinematic data
            window_size: Size of smoothing window
            
        Returns:
            Smoothed kinematic data
        """
        smoothed_data = KinematicData(
            pelvis=self._smooth_joint(kinematic_data.pelvis, window_size),
            hip_left=self._smooth_joint(kinematic_data.hip_left, window_size),
            hip_right=self._smooth_joint(kinematic_data.hip_right, window_size),
            knee_left=self._smooth_joint(kinematic_data.knee_left, window_size),
            knee_right=self._smooth_joint(kinematic_data.knee_right, window_size),
            ankle_left=self._smooth_joint(kinematic_data.ankle_left, window_size),
            ankle_right=self._smooth_joint(kinematic_data.ankle_right, window_size),
            foot_left=self._smooth_joint(kinematic_data.foot_left, window_size) if kinematic_data.foot_left else None,
            foot_right=self._smooth_joint(kinematic_data.foot_right, window_size) if kinematic_data.foot_right else None,
        )
        
        return smoothed_data
    
    def _smooth_joint(self, joint: JointTrajectory, 
                     window_size: int) -> JointTrajectory:
        """Apply smoothing to a single joint trajectory."""
        if window_size <= 1:
            return joint
        
        smoothed_angles = np.zeros_like(joint.angles)
        for i in range(3):  # Each angle component
            smoothed_angles[:, i] = signal.medfilt(joint.angles[:, i], window_size)
        
        return JointTrajectory(
            name=joint.name,
            angles=smoothed_angles,
            time=joint.time,
            rate=joint.rate,
            units=joint.units
        )
    
    def calculate_angular_velocities(self, kinematic_data: KinematicData) -> Dict[str, np.ndarray]:
        """
        Calculate angular velocities from joint angle trajectories.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Dictionary mapping joint names to angular velocity arrays
        """
        velocities = {}
        
        for joint_name in ["pelvis", "hip_left", "hip_right", "knee_left", 
                          "knee_right", "ankle_left", "ankle_right"]:
            joint = getattr(kinematic_data, joint_name)
            dt = 1.0 / joint.rate
            
            # Calculate derivative for each angle component
            vel = np.gradient(joint.angles, dt, axis=0)
            velocities[joint_name] = vel
        
        return velocities
    
    def calculate_range_of_motion(self, kinematic_data: KinematicData) -> Dict[str, Dict[str, float]]:
        """
        Calculate range of motion for each joint.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Dictionary with ROM values for each joint and plane
        """
        rom = {}
        
        for joint_name in ["pelvis", "hip_left", "hip_right", "knee_left", 
                          "knee_right", "ankle_left", "ankle_right"]:
            joint = getattr(kinematic_data, joint_name)
            
            joint_rom = {
                "sagittal": float(np.max(joint.angles[:, 0]) - np.min(joint.angles[:, 0])),
                "frontal": float(np.max(joint.angles[:, 1]) - np.min(joint.angles[:, 1])),
                "transverse": float(np.max(joint.angles[:, 2]) - np.min(joint.angles[:, 2])),
            }
            rom[joint_name] = joint_rom
        
        return rom
