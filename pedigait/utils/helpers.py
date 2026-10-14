"""
Utility Functions Module

Provides helper functions for the pediatric gait analysis package.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import warnings


def normalize_trajectories(trajectories: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize trajectories to 0-100% gait cycle.
    
    Args:
        trajectories: List of trajectories to normalize
        
    Returns:
        List of normalized trajectories
    """
    normalized = []
    
    for traj in trajectories:
        n_frames = len(traj)
        gait_cycle_pct = np.linspace(0, 100, n_frames)
        
        # Interpolate to standard gait cycle
        standard_pct = np.linspace(0, 100, 100)
        interpolated = np.interp(standard_pct, gait_cycle_pct, traj)
        
        normalized.append(interpolated)
    
    return normalized


def detect_heel_strikes(trajectory: np.ndarray, threshold: float = 0.5) -> List[int]:
    """
    Detect heel strike events in a trajectory.
    
    Args:
        trajectory: Heel marker trajectory (n_frames, 3)
        threshold: Threshold for heel strike detection
        
    Returns:
        List of heel strike frame indices
    """
    # Simple threshold-based detection
    # This is a placeholder - actual implementation would use more sophisticated methods
    
    heel_strikes = []
    
    # Look for local minima in the vertical (z) direction
    z_positions = trajectory[:, 2]
    
    for i in range(1, len(z_positions) - 1):
        if z_positions[i] < z_positions[i-1] and z_positions[i] < z_positions[i+1]:
            if z_positions[i] < threshold:
                heel_strikes.append(i)
    
    return heel_strikes


def calculate_gait_parameters(kinematic_data) -> Dict[str, float]:
    """
    Calculate overall gait parameters.
    
    Args:
        kinematic_data: Processed kinematic data
        
    Returns:
        Dictionary of gait parameters
    """
    params = {}
    
    # Calculate stride length (placeholder)
    params["stride_length"] = 0.0
    
    # Calculate stride time
    params["stride_time"] = kinematic_data.knee_left.time[-1] - kinematic_data.knee_left.time[0]
    
    # Calculate cadence (steps per minute)
    params["cadence"] = 0.0
    
    # Calculate walking speed (placeholder)
    params["walking_speed"] = 0.0
    
    return params


def save_results(results: Dict, output_path: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def load_results(input_path: str) -> Dict:
    """
    Load results from JSON file.
    
    Args:
        input_path: Input file path
        
    Returns:
        Results dictionary
    """
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    return results


def create_summary_report(analysis_results: Dict) -> str:
    """
    Create a human-readable summary report.
    
    Args:
        analysis_results: Analysis results dictionary
        
    Returns:
        Formatted summary report string
    """
    report = []
    report.append("=" * 60)
    report.append("PEDIATRIC GAIT ANALYSIS SUMMARY REPORT")
    report.append("=" * 60)
    
    # Overall metrics
    if "overall_metrics" in analysis_results:
        metrics = analysis_results["overall_metrics"]
        report.append("\nOVERALL METRICS:")
        report.append("-" * 40)
        for key, value in metrics.items():
            report.append(f"  {key}: {value:.4f}")
    
    # KSwS metrics
    if "ksws_metrics" in analysis_results:
        ksws = analysis_results["ksws_metrics"]
        report.append("\nKNEE SWING SYMMETRY/SEVERITY (KSwS):")
        report.append("-" * 40)
        report.append(f"  KSwS Index: {ksws.get('ksws_index', 0):.4f}")
        report.append(f"  Classification: {ksws.get('classification', 'N/A')}")
        report.append(f"  Symmetry Index: {ksws.get('symmetry_index', 0):.4f}")
        report.append(f"  Severity Score: {ksws.get('severity_score', 0):.4f}")
    
    # Comparative analysis
    if "comparative_analysis" in analysis_results:
        comp = analysis_results["comparative_analysis"]
        report.append("\nCOMPARATIVE ANALYSIS (TD vs CP):")
        report.append("-" * 40)
        report.append(f"  TD KSwS: {comp.get('td_ksws', {}).get('ksws_index', 0):.4f}")
        report.append(f"  CP KSwS: {comp.get('cp_ksws', {}).get('ksws_index', 0):.4f}")
        report.append(f"  KSwS Difference: {comp.get('differences', {}).get('ksws_index_diff', 0):.4f}")
    
    # Range of motion
    if "range_of_motion" in analysis_results:
        rom = analysis_results["range_of_motion"]
        report.append("\nRANGE OF MOTION:")
        report.append("-" * 40)
        for joint, values in rom.items():
            report.append(f"  {joint}:")
            for plane, value in values.items():
                report.append(f"    {plane}: {value:.2f} degrees")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


def validate_c3d_file(c3d_path: str) -> bool:
    """
    Validate a C3D file.
    
    Args:
        c3d_path: Path to C3D file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import ezc3d
        c = ezc3d.c3d(c3d_path)
        
        # Check if it has point data
        if "points" not in c["data"]:
            return False
        
        # Check if it has valid shape
        points = c["data"]["points"]
        if points.shape[0] < 3 or points.shape[2] < 1:
            return False
        
        return True
    except Exception as e:
        warnings.warn(f"C3D validation failed: {e}")
        return False


def get_file_info(c3d_path: str) -> Dict:
    """
    Get information about a C3D file.
    
    Args:
        c3d_path: Path to C3D file
        
    Returns:
        Dictionary with file information
    """
    try:
        import ezc3d
        c = ezc3d.c3d(c3d_path)
        
        points = c["data"]["points"]
        params = c["parameters"]
        
        info = {
            "file": Path(c3d_path).name,
            "frames": points.shape[2],
            "markers": points.shape[1],
            "point_rate": float(params["POINT"]["RATE"]["value"][0]),
            "units": params["POINT"].get("UNITS", {}).get("value", ["unknown"]),
            "valid": True
        }
        
        return info
    except Exception as e:
        return {
            "file": Path(c3d_path).name,
            "error": str(e),
            "valid": False
        }


def resample_signal(signal: np.ndarray, original_rate: float, 
                   target_rate: float) -> np.ndarray:
    """
    Resample a signal to a target sampling rate.
    
    Args:
        signal: Input signal
        original_rate: Original sampling rate in Hz
        target_rate: Target sampling rate in Hz
        
    Returns:
        Resampled signal
    """
    from scipy import signal as scipy_signal
    
    n_original = len(signal)
    n_target = int(n_original * target_rate / original_rate)
    
    if n_target <= 1:
        return signal
    
    resampled = scipy_signal.resample(signal, n_target)
    
    return resampled


def smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing to a signal.
    
    Args:
        signal: Input signal
        window_size: Size of smoothing window
        
    Returns:
        Smoothed signal
    """
    if window_size <= 1:
        return signal
    
    smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
    
    return smoothed


def find_peaks(signal: np.ndarray, min_distance: int = 10) -> List[int]:
    """
    Find peaks in a signal.
    
    Args:
        signal: Input signal
        min_distance: Minimum distance between peaks
        
    Returns:
        List of peak indices
    """
    from scipy.signal import find_peaks as scipy_find_peaks
    
    peaks, _ = scipy_find_peaks(signal, distance=min_distance)
    
    return peaks.tolist()
