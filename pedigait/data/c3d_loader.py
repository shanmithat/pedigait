"""
C3D File Loader Module

Handles loading and preprocessing of C3D motion capture files.
Extracts marker positions, analog signals, metadata, and pre-calculated joint angles.
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
    
    # Channel identifiers that are NOT 3D physical markers
    NON_MARKER_PATTERNS = {
        "angle", "force", "moment", "power", "normalised",
        "pelo", "pelp", "pela", "pell",
        "lfeo", "lfep", "lfea", "lfel",
        "ltio", "ltip", "ltia", "ltil",
        "lfoo", "lfop", "lfoa", "lfol",
        "rfeo", "rfep", "rfea", "rfel",
        "rtio", "rtip", "rtia", "rtil",
        "rfoo", "rfop", "rfoa", "rfol",
        "ltoo", "ltop", "ltoa", "ltol",
        "rtoo", "rtop", "rtoa", "rtol"
    }

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
        self._cached_markers = None
        self._cached_labels = None
        
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
        
        raw_units = self.c["parameters"]["POINT"].get("UNITS", {}).get("value", ["mm"])
        self.units = str(raw_units[0]) if raw_units else "mm"
        
        return self
    
    def get_kinematic_angles(self) -> Dict[str, np.ndarray]:
        """
        Extract pre-calculated joint angle channels from C3D if available.
        
        Returns:
            Dictionary mapping joint names to angle arrays of shape (n_frames, 3)
        """
        if self.points is None:
            self.load()
        
        angles_dict = {}
        target_map = {
            "knee_left": "LKneeAngles",
            "knee_right": "RKneeAngles",
            "hip_left": "LHipAngles",
            "hip_right": "RHipAngles",
            "ankle_left": "LAnkleAngles",
            "ankle_right": "RAnkleAngles",
            "pelvis": "LPelvisAngles"
        }
        
        label_lower_to_idx = {l.lower(): i for i, l in enumerate(self.labels)}
        
        for joint, target_name in target_map.items():
            if target_name.lower() in label_lower_to_idx:
                idx = label_lower_to_idx[target_name.lower()]
                raw_angles = self.points[:3, idx, :].T  # Shape: (n_frames, 3)
                # Check that it's not all NaNs or zeros
                if np.isfinite(raw_angles).any() and not np.all(raw_angles == 0):
                    df = pd.DataFrame(raw_angles)
                    clean_angles = df.interpolate(method='linear', limit_direction='both').fillna(0).to_numpy()
                    angles_dict[joint] = clean_angles
                    
        return angles_dict

    def get_marker_positions(self, scale_to_meters: bool = True) -> np.ndarray:
        """
        Get filtered and centered 3D anatomical marker positions.
        
        Args:
            scale_to_meters: Convert from mm to meters if True
            
        Returns:
            Array of shape (3, n_markers, n_frames) with x, y, z coordinates
        """
        if self._cached_markers is not None:
            return self._cached_markers
            
        if self.points is None:
            self.load()
        
        pts = self.points[:3].astype(float)
        
        # Filter only physical anatomical optical markers
        chosen_indices = []
        chosen_labels = []
        
        for i, l in enumerate(self.labels):
            l_str = str(l).strip()
            l_low = l_str.lower()
            
            # Exclude uncalibrated/dummy markers or calculation channels
            if l_str.startswith("*") or l_str == "":
                continue
            if any(k in l_low for k in self.NON_MARKER_PATTERNS):
                continue
            
            # Require at least some finite observations
            finite_mask = np.isfinite(pts[:, i, :])
            if finite_mask.sum() > 0 and not np.all(pts[:, i, :] == 0):
                chosen_indices.append(i)
                chosen_labels.append(l_str)
        
        if not chosen_indices:
            # Fallback if filtering was too aggressive
            chosen_indices = list(range(min(30, pts.shape[1])))
            chosen_labels = [self.labels[i] for i in chosen_indices]
            
        coords = pts[:, chosen_indices, :].copy()
        
        # Interpolate NaNs per marker channel across frames
        for m in range(coords.shape[1]):
            for dim in range(3):
                s = pd.Series(coords[dim, m, :])
                coords[dim, m, :] = s.interpolate(method="linear", limit_direction="both").fillna(0).to_numpy()
        
        # Scale to meters if coordinates are in millimeters
        if scale_to_meters and np.nanpercentile(np.abs(coords), 95) > 10.0:
            coords = coords / 1000.0
            self.units = "m"
            
        # Center in X and Y around the subject's mean position
        x_center = float(np.nanmedian(coords[0]))
        y_center = float(np.nanmedian(coords[1]))
        coords[0] -= x_center
        coords[1] -= y_center
        
        # Ground plane adjustment: place lowest foot point near z = 0
        z_floor = float(np.nanpercentile(coords[2], 2))
        coords[2] -= z_floor
        
        self._cached_markers = coords
        self._cached_labels = chosen_labels
        return coords
    
    def get_marker_labels(self) -> List[str]:
        """Get labels corresponding to get_marker_positions()."""
        if self._cached_labels is None:
            self.get_marker_positions()
        return self._cached_labels
    
    def get_metadata(self) -> Dict:
        """Get C3D file metadata."""
        if self.points is None:
            self.load()
        return {
            "file": self.c3d_path.name,
            "group": self.c3d_path.parent.name,
            "frames": self.points.shape[2] if self.points is not None else 0,
            "markers": len(self.get_marker_labels()),
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
    """Load all C3D files from a directory."""
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
    """Get metadata for all C3D files."""
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
