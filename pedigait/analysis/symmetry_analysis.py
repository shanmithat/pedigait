"""
Symmetry and Severity Analysis Module

Analyzes gait symmetry and calculates Knee Swing Symmetry/Severity (KSwS) indices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal, stats


@dataclass
class SymmetryMetrics:
    """Container for symmetry metrics."""
    symmetry_index: float  # 0 = perfect asymmetry, 1 = perfect symmetry
    asymmetry_score: float  # Higher = more asymmetric
    correlation: float  # Correlation between left and right
    mean_difference: float  # Mean absolute difference
    std_difference: float  # Standard deviation of differences


@dataclass
class SeverityMetrics:
    """Container for severity metrics."""
    severity_score: float  # 0 = normal, 1 = highly pathological
    classification: str  # Normal, Mild, Moderate, Severe, Highly Pathological
    percentile: float  # Percentile relative to normative data


@dataclass
class KSwSMetrics:
    """Container for Knee Swing Symmetry/Severity metrics."""
    ksws_index: float
    symmetry: SymmetryMetrics
    severity: SeverityMetrics
    left_knee_metrics: Dict[str, float]
    right_knee_metrics: Dict[str, float]


class SymmetryAnalyzer:
    """Analyze gait symmetry and calculate symmetry indices."""
    
    def __init__(self):
        """Initialize symmetry analyzer."""
        pass
    
    def calculate_symmetry_index(self, left_trajectory: np.ndarray, 
                                right_trajectory: np.ndarray) -> SymmetryMetrics:
        """
        Calculate symmetry index between left and right trajectories.
        
        Args:
            left_trajectory: Left side trajectory (n_frames,)
            right_trajectory: Right side trajectory (n_frames,)
            
        Returns:
            SymmetryMetrics object
        """
        # Ensure same length
        min_len = min(len(left_trajectory), len(right_trajectory))
        left = left_trajectory[:min_len]
        right = right_trajectory[:min_len]
        
        # Calculate symmetry index (1 - normalized RMSE)
        diff = left - right
        rmse = np.sqrt(np.mean(diff ** 2))
        
        # Normalize by the range of the combined signal
        combined = np.concatenate([left, right])
        signal_range = np.max(combined) - np.min(combined)
        
        if signal_range > 0:
            normalized_rmse = rmse / signal_range
            symmetry_index = 1.0 - normalized_rmse
        else:
            symmetry_index = 1.0
        
        # Calculate correlation
        corr, _ = stats.pearsonr(left, right)
        
        # Calculate mean and std of differences
        mean_diff = float(np.mean(np.abs(diff)))
        std_diff = float(np.std(diff))
        
        return SymmetryMetrics(
            symmetry_index=float(symmetry_index),
            asymmetry_score=float(1.0 - symmetry_index),
            correlation=float(corr),
            mean_difference=mean_diff,
            std_difference=std_diff
        )
    
    def analyze_knee_symmetry(self, kinematic_data) -> KSwSMetrics:
        """
        Analyze knee swing symmetry from kinematic data.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            KSwSMetrics object
        """
        # Extract knee trajectories
        left_knee = kinematic_data.knee_left.angles
        right_knee = kinematic_data.knee_right.angles
        
        # Focus on sagittal plane (flexion/extension)
        left_sagittal = left_knee[:, 0]
        right_sagittal = right_knee[:, 0]
        
        # Calculate symmetry metrics
        symmetry = self.calculate_symmetry_index(left_sagittal, right_sagittal)
        
        # Calculate severity metrics
        severity = self._calculate_knee_severity(left_sagittal, right_sagittal)
        
        # Calculate individual knee metrics
        left_metrics = self._calculate_joint_metrics(left_sagittal)
        right_metrics = self._calculate_joint_metrics(right_sagittal)
        
        # Calculate KSwS index (combination of symmetry and severity)
        ksws_index = self._calculate_ksws_index(symmetry, severity)
        
        return KSwSMetrics(
            ksws_index=ksws_index,
            symmetry=symmetry,
            severity=severity,
            left_knee_metrics=left_metrics,
            right_knee_metrics=right_metrics
        )
    
    def _calculate_knee_severity(self, left_sagittal: np.ndarray, 
                                right_sagittal: np.ndarray) -> SeverityMetrics:
        """
        Calculate knee severity metrics.
        
        Args:
            left_sagittal: Left knee sagittal trajectory
            right_sagittal: Right knee sagittal trajectory
            
        Returns:
            SeverityMetrics object
        """
        # Calculate features indicative of pathology
        left_features = self._extract_gait_features(left_sagittal)
        right_features = self._extract_gait_features(right_sagittal)
        
        # Combine features
        combined_features = {}
        for key in left_features:
            combined_features[key] = (left_features[key] + right_features[key]) / 2.0
        
        # Calculate severity score (0-1 scale)
        severity_score = self._calculate_severity_score(combined_features)
        
        # Classify based on score
        classification = self._classify_severity(severity_score)
        
        # Calculate percentile (placeholder - would need normative data)
        percentile = 100.0 * (1.0 - severity_score)
        
        return SeverityMetrics(
            severity_score=severity_score,
            classification=classification,
            percentile=percentile
        )
    
    def _extract_gait_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """
        Extract gait features from a trajectory.
        
        Args:
            trajectory: Joint angle trajectory
            
        Returns:
            Dictionary of gait features
        """
        features = {}
        
        # Range of motion
        features["rom"] = float(np.max(trajectory) - np.min(trajectory))
        
        # Mean absolute value
        features["mean_abs"] = float(np.mean(np.abs(trajectory)))
        
        # Standard deviation
        features["std"] = float(np.std(trajectory))
        
        # Peak values
        features["peak_positive"] = float(np.max(trajectory))
        features["peak_negative"] = float(np.min(trajectory))
        
        # Variability (coefficient of variation)
        if np.mean(np.abs(trajectory)) > 0:
            features["cv"] = float(np.std(trajectory) / np.mean(np.abs(trajectory)))
        else:
            features["cv"] = 0.0
        
        # Jerk (rate of change of acceleration)
        if len(trajectory) > 2:
            vel = np.gradient(trajectory)
            acc = np.gradient(vel)
            jerk = np.gradient(acc)
            features["jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2)))
        else:
            features["jerk_rms"] = 0.0
        
        return features
    
    def _calculate_severity_score(self, features: Dict[str, float]) -> float:
        """
        Calculate severity score from gait features.
        
        Args:
            features: Dictionary of gait features
            
        Returns:
            Severity score (0-1)
        """
        # Normalize features (placeholder - would use normative ranges)
        # For now, use simple heuristics
        
        score = 0.0
        
        # Low ROM can indicate stiffness (pathological)
        if features.get("rom", 0) < 30:  # Threshold in degrees
            score += 0.3
        
        # High jerk indicates lack of smoothness
        if features.get("jerk_rms", 0) > 10:  # Arbitrary threshold
            score += 0.2
        
        # High variability
        if features.get("cv", 0) > 0.5:
            score += 0.2
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _classify_severity(self, score: float) -> str:
        """
        Classify severity based on score.
        
        Args:
            score: Severity score (0-1)
            
        Returns:
            Classification string
        """
        if score < 0.1:
            return "Normal"
        elif score < 0.3:
            return "Mild"
        elif score < 0.6:
            return "Moderate"
        elif score < 0.8:
            return "Severe"
        else:
            return "Highly Pathological"
    
    def _calculate_joint_metrics(self, trajectory: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for a joint trajectory.
        
        Args:
            trajectory: Joint angle trajectory
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        metrics["min"] = float(np.min(trajectory))
        metrics["max"] = float(np.max(trajectory))
        metrics["mean"] = float(np.mean(trajectory))
        metrics["std"] = float(np.std(trajectory))
        metrics["rom"] = float(np.max(trajectory) - np.min(trajectory))
        metrics["rms"] = float(np.sqrt(np.mean(trajectory ** 2)))
        
        # Calculate velocity and acceleration metrics
        if len(trajectory) > 1:
            vel = np.gradient(trajectory)
            metrics["vel_max"] = float(np.max(np.abs(vel)))
            metrics["vel_rms"] = float(np.sqrt(np.mean(vel ** 2)))
            
            if len(trajectory) > 2:
                acc = np.gradient(vel)
                metrics["acc_max"] = float(np.max(np.abs(acc)))
                metrics["acc_rms"] = float(np.sqrt(np.mean(acc ** 2)))
        
        return metrics
    
    def _calculate_ksws_index(self, symmetry: SymmetryMetrics, 
                             severity: SeverityMetrics) -> float:
        """
        Calculate KSwS index from symmetry and severity metrics.
        
        Args:
            symmetry: Symmetry metrics
            severity: Severity metrics
            
        Returns:
            KSwS index (0-1, where 0 = normal, 1 = highly pathological)
        """
        # Weight symmetry and severity components
        symmetry_weight = 0.4
        severity_weight = 0.6
        
        # KSwS index combines asymmetry and severity
        ksws_index = (symmetry_weight * symmetry.asymmetry_score + 
                     severity_weight * severity.severity_score)
        
        return ksws_index
    
    def analyze_all_joints_symmetry(self, kinematic_data) -> Dict[str, SymmetryMetrics]:
        """
        Analyze symmetry for all joint pairs.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Dictionary mapping joint names to symmetry metrics
        """
        symmetry_results = {}
        
        # Define joint pairs
        joint_pairs = [
            ("hip_left", "hip_right"),
            ("knee_left", "knee_right"),
            ("ankle_left", "ankle_right"),
        ]
        
        for left_name, right_name in joint_pairs:
            left_joint = getattr(kinematic_data, left_name)
            right_joint = getattr(kinematic_data, right_name)
            
            # Use sagittal plane for symmetry analysis
            left_sagittal = left_joint.angles[:, 0]
            right_sagittal = right_joint.angles[:, 0]
            
            symmetry = self.calculate_symmetry_index(left_sagittal, right_sagittal)
            symmetry_results[f"{left_name}_vs_{right_name}"] = symmetry
        
        return symmetry_results
    
    def calculate_gait_asymmetry_metrics(self, kinematic_data) -> Dict[str, float]:
        """
        Calculate overall gait asymmetry metrics.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Dictionary of asymmetry metrics
        """
        metrics = {}
        
        # Calculate symmetry for each joint pair
        symmetry_results = self.analyze_all_joints_symmetry(kinematic_data)
        
        # Overall symmetry score (average of all joint symmetries)
        overall_symmetry = np.mean(
            [s.symmetry_index for s in symmetry_results.values()]
        )
        metrics["overall_symmetry_index"] = overall_symmetry
        metrics["overall_asymmetry_score"] = 1.0 - overall_symmetry
        
        # Individual joint asymmetries
        for joint_pair, symmetry in symmetry_results.items():
            metrics[f"{joint_pair}_asymmetry"] = symmetry.asymmetry_score
        
        # Calculate temporal asymmetry (difference in timing)
        temporal_asymmetry = self._calculate_temporal_asymmetry(kinematic_data)
        metrics["temporal_asymmetry"] = temporal_asymmetry
        
        return metrics
    
    def _calculate_temporal_asymmetry(self, kinematic_data) -> float:
        """
        Calculate temporal asymmetry between left and right sides.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Temporal asymmetry score
        """
        # Compare timing of peak flexion/extension
        # This is a placeholder - actual implementation would use event detection
        
        left_knee = kinematic_data.knee_left.angles[:, 0]
        right_knee = kinematic_data.knee_right.angles[:, 0]
        
        # Find peak flexion times
        left_peak_idx = np.argmax(left_knee)
        right_peak_idx = np.argmax(right_knee)
        
        # Calculate time difference
        time_diff = abs(left_peak_idx - right_peak_idx) / kinematic_data.knee_left.rate
        
        # Normalize by gait cycle duration (assuming ~1 second)
        temporal_asymmetry = min(time_diff / 1.0, 1.0)
        
        return temporal_asymmetry


class ComparativeAnalyzer:
    """Analyze and compare multiple gait trials."""
    
    def __init__(self):
        """Initialize comparative analyzer."""
        self.symmetry_analyzer = SymmetryAnalyzer()
    
    def compare_td_vs_cp(self, td_data, cp_data) -> Dict[str, Dict]:
        """
        Compare typically developing (TD) vs cerebral palsy (CP) gait.
        
        Args:
            td_data: TD kinematic data
            cp_data: CP kinematic data
            
        Returns:
            Dictionary with comparative analysis results
        """
        comparison = {}
        
        # Analyze symmetry for both
        td_ksws = self.symmetry_analyzer.analyze_knee_symmetry(td_data)
        cp_ksws = self.symmetry_analyzer.analyze_knee_symmetry(cp_data)
        
        comparison["td_ksws"] = {
            "ksws_index": td_ksws.ksws_index,
            "symmetry_index": td_ksws.symmetry.symmetry_index,
            "severity_score": td_ksws.severity.severity_score,
            "classification": td_ksws.severity.classification
        }
        
        comparison["cp_ksws"] = {
            "ksws_index": cp_ksws.ksws_index,
            "symmetry_index": cp_ksws.symmetry.symmetry_index,
            "severity_score": cp_ksws.severity.severity_score,
            "classification": cp_ksws.severity.classification
        }
        
        # Calculate differences
        comparison["differences"] = {
            "ksws_index_diff": cp_ksws.ksws_index - td_ksws.ksws_index,
            "symmetry_diff": td_ksws.symmetry.symmetry_index - cp_ksws.symmetry.symmetry_index,
            "severity_diff": cp_ksws.severity.severity_score - td_ksws.severity.severity_score
        }
        
        # Calculate range of motion differences
        td_rom = self._calculate_overall_rom(td_data)
        cp_rom = self._calculate_overall_rom(cp_data)
        
        comparison["rom_differences"] = {
            joint: cp_rom.get(joint, {}).get("sagittal", 0) - td_rom.get(joint, {}).get("sagittal", 0)
            for joint in td_rom.keys()
        }
        
        return comparison
    
    def _calculate_overall_rom(self, kinematic_data) -> Dict[str, Dict[str, float]]:
        """
        Calculate overall range of motion for all joints.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Dictionary with ROM for each joint and plane
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
    
    def calculate_deviation_from_norm(self, cp_data, td_data) -> Dict[str, float]:
        """
        Calculate how CP gait deviates from TD (normative) gait.
        
        Args:
            cp_data: CP kinematic data
            td_data: TD (normative) kinematic data
            
        Returns:
            Dictionary with deviation metrics
        """
        deviation = {}
        
        # Compare each joint
        for joint_name in ["pelvis", "hip_left", "hip_right", "knee_left", 
                          "knee_right", "ankle_left", "ankle_right"]:
            cp_joint = getattr(cp_data, joint_name)
            td_joint = getattr(td_data, joint_name)
            
            # Calculate RMSE between trajectories
            for i, plane in enumerate(["sagittal", "frontal", "transverse"]):
                rmse = np.sqrt(np.mean((cp_joint.angles[:, i] - td_joint.angles[:, i]) ** 2))
                deviation[f"{joint_name}_{plane}_rmse"] = rmse
        
        return deviation
