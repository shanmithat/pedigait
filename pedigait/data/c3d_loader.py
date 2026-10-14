"""
C3D File Loader Module

Handles loading and preprocessing of C3D motion capture files.
Extracts marker positions, analog signals, and metadata.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import ezc3d
except ImportError:
    warnings.warn("ezc3d not installed. C3D loading will fail.")
    ezc3d = None


class C3DLoader:
    """Load and preprocess C3D motion capture files."""
    
    def __init__(self, c3d_path: str):
        """
        Initialize C3D loader with file path.
        
        Args:
            c3d_path: Path to the C3D file
        """
        self.c3d_path = Path(c3d_path)
        if not self.c3d_path.exists():
            raise FileNotFoundError(f"C3D file not found: {c3d_path}")
        
        self.c = None
        self.points = None
        self.labels = None
        self.analogs = None
        self.analog_labels = None
        self.point_rate = None
        self.analog_rate = None
        self.units = None
        
    def load(self):
        """Load the C3D file and extract basic data."""
        if ezc3d is None:
            raise ImportError("ezc3d is required. Install with: pip install ezc3d")
        
        self.c = ezc3d.c3d(str(self.c3d_path))
        
        # Extract point data
        self.points = np.asarray(self.c["data"]["points"], dtype=float)
        
        # Extract labels
        self.labels = [str(x).strip() for x in 
                      self.c["parameters"]["POINT"]["LABELS"]["value"]]
        
        # Extract analog data if available
        if "analogs" in self.c["data"]:
            self.analogs = np.asarray(self.c["data"]["analogs"], dtype=float)
            self.analog_labels = [str(x).strip() for x in 
                                  self.c["parameters"]["ANALOG"]["LABELS"]["value"]]
        
        # Extract metadata
        self.point_rate = float(self.c["parameters"]["POINT"]["RATE"]["value"][0])
        
        if "ANALOG" in self.c["parameters"]:
            analog_rate = self.c["parameters"]["ANALOG"].get("RATE", {}).get("value", [None])
            self.analog_rate = float(analog_rate[0]) if analog_rate and analog_rate[0] else None
        
        self.units = self.c["parameters"]["POINT"].get("UNITS", {}).get("value", ["unknown"])
        
        return self
    
    def get_marker_positions(self, scale_to_meters: bool = True) -> np.ndarray:
        """
        Get marker positions from C3D file.
        
        Args:
            scale_to_meters: Convert from mm to meters if True
            
        Returns:
            Array of shape (3, n_markers, n_frames) with x, y, z coordinates
        """
        if self.points is None:
            self.load()
        
        xyz = self.points[:3].astype(float)
        
        # Drop markers with no valid observations
        valid_marker = np.isfinite(xyz).all(axis=(0, 2))
        xyz = xyz[:, valid_marker, :]
        
        # Center the coordinates for stable viewing
        xyz = xyz - np.nanmedian(xyz, axis=(1, 2), keepdims=True)
        
        # Scale to meters if needed
        if scale_to_meters and np.nanpercentile(np.abs(xyz), 95) > 10:
            xyz = xyz / 1000.0
        
        return xyz
    
    def get_marker_labels(self) -> List[str]:
        """Get valid marker labels."""
        if self.points is None:
            self.load()
        
        xyz = self.points[:3].astype(float)
        valid_marker = np.isfinite(xyz).all(axis=(0, 2))
        labels = [label for label, keep in zip(self.labels, valid_marker) if keep]
        
        return labels
    
    def get_joint_angles(self, joint_config: Dict) -> Dict[str, np.ndarray]:
        """
        Extract joint angle trajectories from marker positions.
        
        Args:
            joint_config: Dictionary defining joint configurations
            
        Returns:
            Dictionary with joint angle trajectories
        """
        markers = self.get_marker_positions()
        labels = self.get_marker_labels()
        
        joint_angles = {}
        
        # Implement joint angle calculations based on marker positions
        # This is a placeholder - actual implementation depends on marker set
        
        return joint_angles
    
    def get_metadata(self) -> Dict:
        """Get C3D file metadata."""
        return {
            "file": self.c3d_path.name,
            "group": self.c3d_path.parent.name,
            "frames": self.points.shape[2] if self.points is not None else 0,
            "markers": len(self.labels) if self.labels else 0,
            "point_rate_hz": self.point_rate,
            "analog_channels": len(self.analog_labels) if self.analog_labels else 0,
            "units": self.units,
        }
    
    def get_force_platform_data(self) -> Optional[Dict]:
        """Extract force platform data if available."""
        if self.c is None:
            self.load()
        
        fp = self.c["parameters"].get("FORCE_PLATFORM", {})
        if not fp:
            return None
        
        return {
            "used": fp.get("USED", {}).get("value", [0]),
            "type": fp.get("TYPE", {}).get("value", []),
            "channels": fp.get("CHANNEL", {}).get("value", []),
        }


def load_c3d_directory(directory: str) -> List[C3DLoader]:
    """
    Load all C3D files from a directory.
    
    Args:
        directory: Path to directory containing C3D files
        
    Returns:
        List of C3DLoader objects
    """
    c3d_files = sorted(Path(directory).rglob("*.c3d"))
    loaders = []
    
    for c3d_path in c3d_files:
        try:
            loader = C3DLoader(str(c3d_path))
            loader.load()
            loaders.append(loader)
        except Exception as e:
            warnings.warn(f"Failed to load {c3d_path}: {e}")
    
    return loaders


def get_trial_info(c3d_paths: List[str]) -> pd.DataFrame:
    """
    Get metadata for all C3D files.
    
    Args:
        c3d_paths: List of paths to C3D files
        
    Returns:
        DataFrame with trial information
    """
    trial_info = []
    
    for path in c3d_paths:
        try:
            loader = C3DLoader(str(path))
            loader.load()
            info = loader.get_metadata()
            info["path"] = str(path)
            trial_info.append(info)
        except Exception as e:
            trial_info.append({
                "path": str(path),
                "error": str(e)
            })
    
    return pd.DataFrame(trial_info)
