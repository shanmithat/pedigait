"""
Comparative Analysis Module

Provides tools for comparing typically developing (TD) vs cerebral palsy (CP) gait.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Visualization will be limited.")


@dataclass
class ComparativeResult:
    """Container for comparative analysis results."""
    td_data: Dict
    cp_data: Dict
    differences: Dict
    metrics: Dict
    visualizations: Optional[Dict] = None


class ComparativeAnalyzer:
    """Analyze and compare TD vs CP gait patterns."""
    
    def __init__(self):
        """Initialize comparative analyzer."""
        pass
    
    def compare_gait_patterns(self, td_kinematic_data, cp_kinematic_data) -> ComparativeResult:
        """
        Perform comprehensive comparison of TD vs CP gait.
        
        Args:
            td_kinematic_data: TD kinematic data
            cp_kinematic_data: CP kinematic data
            
        Returns:
            ComparativeResult object
        """
        # Extract features from both
        td_features = self._extract_comparative_features(td_kinematic_data)
        cp_features = self._extract_comparative_features(cp_kinematic_data)
        
        # Calculate differences
        differences = self._calculate_differences(td_features, cp_features)
        
        # Calculate comparative metrics
        metrics = self._calculate_comparative_metrics(td_features, cp_features)
        
        return ComparativeResult(
            td_data=td_features,
            cp_data=cp_features,
            differences=differences,
            metrics=metrics
        )
    
    def _extract_comparative_features(self, kinematic_data) -> Dict:
        """
        Extract features for comparative analysis.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Extract joint trajectories
        joints = ["pelvis", "hip_left", "hip_right", "knee_left", 
                "knee_right", "ankle_left", "ankle_right"]
        
        for joint_name in joints:
            joint = getattr(kinematic_data, joint_name)
            
            features[f"{joint_name}_trajectory"] = {
                "sagittal": joint.angles[:, 0],
                "frontal": joint.angles[:, 1],
                "transverse": joint.angles[:, 2],
                "time": joint.time
            }
            
            # Calculate ROM
            features[f"{joint_name}_rom"] = {
                "sagittal": float(np.max(joint.angles[:, 0]) - np.min(joint.angles[:, 0])),
                "frontal": float(np.max(joint.angles[:, 1]) - np.min(joint.angles[:, 1])),
                "transverse": float(np.max(joint.angles[:, 2]) - np.min(joint.angles[:, 2])),
            }
            
            # Calculate statistical features
            features[f"{joint_name}_stats"] = {
                "mean_sagittal": float(np.mean(joint.angles[:, 0])),
                "std_sagittal": float(np.std(joint.angles[:, 0])),
                "mean_frontal": float(np.mean(joint.angles[:, 1])),
                "std_frontal": float(np.std(joint.angles[:, 1])),
                "mean_transverse": float(np.mean(joint.angles[:, 2])),
                "std_transverse": float(np.std(joint.angles[:, 2])),
            }
        
        # Calculate overall gait parameters
        features["gait_parameters"] = self._calculate_gait_parameters(kinematic_data)
        
        return features
    
    def _calculate_differences(self, td_features: Dict, cp_features: Dict) -> Dict:
        """
        Calculate differences between TD and CP features.
        
        Args:
            td_features: TD features
            cp_features: CP features
            
        Returns:
            Dictionary of differences
        """
        differences = {}
        
        # Calculate ROM differences
        joints = ["pelvis", "hip_left", "hip_right", "knee_left", 
                "knee_right", "ankle_left", "ankle_right"]
        
        for joint_name in joints:
            td_rom = td_features.get(f"{joint_name}_rom", {})
            cp_rom = cp_features.get(f"{joint_name}_rom", {})
            
            differences[f"{joint_name}_rom_diff"] = {
                "sagittal": cp_rom.get("sagittal", 0) - td_rom.get("sagittal", 0),
                "frontal": cp_rom.get("frontal", 0) - td_rom.get("frontal", 0),
                "transverse": cp_rom.get("transverse", 0) - td_rom.get("transverse", 0),
            }
        
        # Calculate trajectory RMSE
        for joint_name in joints:
            td_traj = td_features.get(f"{joint_name}_trajectory", {})
            cp_traj = cp_features.get(f"{joint_name}_trajectory", {})
            
            for plane in ["sagittal", "frontal", "transverse"]:
                td_signal = td_traj.get(plane, np.array([]))
                cp_signal = cp_traj.get(plane, np.array([]))
                
                if len(td_signal) > 0 and len(cp_signal) > 0:
                    # Interpolate to same length
                    min_len = min(len(td_signal), len(cp_signal))
                    td_interp = td_signal[:min_len]
                    cp_interp = cp_signal[:min_len]
                    
                    rmse = np.sqrt(np.mean((td_interp - cp_interp) ** 2))
                    differences[f"{joint_name}_{plane}_rmse"] = rmse
        
        # Calculate statistical differences
        for joint_name in joints:
            td_stats = td_features.get(f"{joint_name}_stats", {})
            cp_stats = cp_features.get(f"{joint_name}_stats", {})
            
            for stat in ["mean", "std"]:
                for plane in ["sagittal", "frontal", "transverse"]:
                    key = f"{stat}_{plane}"
                    diff = cp_stats.get(key, 0) - td_stats.get(key, 0)
                    differences[f"{joint_name}_{key}_diff"] = diff
        
        return differences
    
    def _calculate_comparative_metrics(self, td_features: Dict, cp_features: Dict) -> Dict:
        """
        Calculate overall comparative metrics.
        
        Args:
            td_features: TD features
            cp_features: CP features
            
        Returns:
            Dictionary of comparative metrics
        """
        metrics = {}
        
        # Calculate overall ROM difference
        joints = ["pelvis", "hip_left", "hip_right", "knee_left", 
                "knee_right", "ankle_left", "ankle_right"]
        
        total_rom_diff = 0
        count = 0
        
        for joint_name in joints:
            td_rom = td_features.get(f"{joint_name}_rom", {})
            cp_rom = cp_features.get(f"{joint_name}_rom", {})
            
            for plane in ["sagittal", "frontal", "transverse"]:
                diff = abs(cp_rom.get(plane, 0) - td_rom.get(plane, 0))
                total_rom_diff += diff
                count += 1
        
        if count > 0:
            metrics["overall_rom_difference"] = total_rom_diff / count
        
        # Calculate overall trajectory similarity
        total_rmse = 0
        count = 0
        
        for joint_name in joints:
            for plane in ["sagittal", "frontal", "transverse"]:
                rmse = self.differences.get(f"{joint_name}_{plane}_rmse", 0)
                total_rmse += rmse
                count += 1
        
        if count > 0:
            metrics["overall_trajectory_rmse"] = total_rmse / count
        
        # Calculate symmetry differences
        metrics["symmetry_difference"] = self._calculate_symmetry_difference(td_features, cp_features)
        
        return metrics
    
    def _calculate_gait_parameters(self, kinematic_data) -> Dict:
        """
        Calculate overall gait parameters.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Dictionary of gait parameters
        """
        params = {}
        
        # Calculate stride time
        if hasattr(kinematic_data, 'knee_left'):
            params["stride_time"] = float(kinematic_data.knee_left.time[-1] - kinematic_data.knee_left.time[0])
        
        # Calculate cadence (steps per minute) - placeholder
        params["cadence"] = 0.0
        
        # Calculate walking speed - placeholder
        params["walking_speed"] = 0.0
        
        return params
    
    def _calculate_symmetry_difference(self, td_features: Dict, cp_features: Dict) -> float:
        """
        Calculate difference in symmetry between TD and CP.
        
        Args:
            td_features: TD features
            cp_features: CP features
            
        Returns:
            Symmetry difference score
        """
        # This is a placeholder - actual implementation would use symmetry analysis
        return 0.0
    
    def create_comparative_visualizations(self, td_features: Dict, cp_features: Dict) -> Dict:
        """
        Create comparative visualizations.
        
        Args:
            td_features: TD features
            cp_features: CP features
            
        Returns:
            Dictionary of visualization figures
        """
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available. Cannot create visualizations.")
            return {}
        
        visualizations = {}
        
        # Create ROM comparison
        visualizations["rom_comparison"] = self._create_rom_comparison_fig(td_features, cp_features)
        
        # Create trajectory comparison
        visualizations["trajectory_comparison"] = self._create_trajectory_comparison_fig(td_features, cp_features)
        
        # Create statistical comparison
        visualizations["statistical_comparison"] = self._create_statistical_comparison_fig(td_features, cp_features)
        
        return visualizations
    
    def _create_rom_comparison_fig(self, td_features: Dict, cp_features: Dict) -> go.Figure:
        """Create ROM comparison figure."""
        joints = ["pelvis", "hip_left", "hip_right", "knee_left", 
                "knee_right", "ankle_left", "ankle_right"]
        planes = ["sagittal", "frontal", "transverse"]
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=planes,
            horizontal_spacing=0.1
        )
        
        for i, plane in enumerate(planes):
            td_values = []
            cp_values = []
            
            for joint_name in joints:
                td_rom = td_features.get(f"{joint_name}_rom", {})
                cp_rom = cp_features.get(f"{joint_name}_rom", {})
                
                td_values.append(td_rom.get(plane, 0))
                cp_values.append(cp_rom.get(plane, 0))
            
            fig.add_trace(
                go.Bar(x=joints, y=td_values, name="TD", marker_color="blue"),
                row=1, col=i+1
            )
            
            fig.add_trace(
                go.Bar(x=joints, y=cp_values, name="CP", marker_color="red"),
                row=1, col=i+1
            )
            
            fig.update_yaxes(title_text=f"ROM ({plane})", row=1, col=i+1)
        
        fig.update_layout(
            title="Range of Motion Comparison",
            barmode="group",
            height=500
        )
        
        return fig
    
    def _create_trajectory_comparison_fig(self, td_features: Dict, cp_features: Dict) -> go.Figure:
        """Create trajectory comparison figure."""
        # Focus on key joints
        key_joints = ["knee_left", "knee_right", "hip_left", "hip_right"]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=key_joints
        )
        
        for i, joint_name in enumerate(key_joints):
            row = i // 2 + 1
            col = i % 2 + 1
            
            td_traj = td_features.get(f"{joint_name}_trajectory", {})
            cp_traj = cp_features.get(f"{joint_name}_trajectory", {})
            
            time = td_traj.get("time", np.array([]))
            td_sagittal = td_traj.get("sagittal", np.array([]))
            cp_sagittal = cp_traj.get("sagittal", np.array([]))
            
            if len(time) > 0 and len(td_sagittal) > 0 and len(cp_sagittal) > 0:
                fig.add_trace(
                    go.Scatter(x=time, y=td_sagittal, name=f"TD {joint_name}", 
                              line=dict(color="blue", width=2)),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(x=time, y=cp_sagittal, name=f"CP {joint_name}", 
                              line=dict(color="red", width=2, dash="dash")),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, showlegend=True)
        
        return fig
    
    def _create_statistical_comparison_fig(self, td_features: Dict, cp_features: Dict) -> go.Figure:
        """Create statistical comparison figure."""
        joints = ["knee_left", "knee_right"]
        stats = ["mean", "std"]
        planes = ["sagittal"]
        
        fig = go.Figure()
        
        for joint_name in joints:
            for stat in stats:
                for plane in planes:
                    td_value = td_features.get(f"{joint_name}_stats", {}).get(f"{stat}_{plane}", 0)
                    cp_value = cp_features.get(f"{joint_name}_stats", {}).get(f"{stat}_{plane}", 0)
                    
                    fig.add_trace(
                        go.Bar(
                            x=[f"{joint_name}_{stat}_{plane}"],
                            y=[td_value],
                            name=f"TD {joint_name}",
                            marker_color="blue"
                        )
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=[f"{joint_name}_{stat}_{plane}"],
                            y=[cp_value],
                            name=f"CP {joint_name}",
                            marker_color="red"
                        )
                    )
        
        fig.update_layout(
            title="Statistical Comparison",
            barmode="group",
            height=500
        )
        
        return fig


class PathologyClassifier:
    """Classify gait patterns based on pathology."""
    
    def __init__(self):
        """Initialize pathology classifier."""
        self.symmetry_analyzer = None
        try:
            from ..analysis.symmetry_analysis import SymmetryAnalyzer
            self.symmetry_analyzer = SymmetryAnalyzer()
        except ImportError:
            warnings.warn("SymmetryAnalyzer not available")
    
    def classify_pathology(self, kinematic_data) -> Dict:
        """
        Classify gait pathology.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Dictionary with classification results
        """
        classification = {}
        
        # Analyze symmetry
        if self.symmetry_analyzer:
            ksws_metrics = self.symmetry_analyzer.analyze_knee_symmetry(kinematic_data)
            classification["ksws_index"] = ksws_metrics.ksws_index
            classification["classification"] = ksws_metrics.severity.classification
            classification["severity_score"] = ksws_metrics.severity.severity_score
        
        # Add ROM-based classification
        rom_classification = self._classify_by_rom(kinematic_data)
        classification.update(rom_classification)
        
        # Add temporal classification
        temporal_classification = self._classify_by_temporal_features(kinematic_data)
        classification.update(temporal_classification)
        
        return classification
    
    def _classify_by_rom(self, kinematic_data) -> Dict:
        """
        Classify based on range of motion.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Dictionary with ROM-based classification
        """
        classification = {}
        
        # Calculate ROM for key joints
        knee_rom = self._calculate_joint_rom(kinematic_data, "knee_left")
        hip_rom = self._calculate_joint_rom(kinematic_data, "hip_left")
        ankle_rom = self._calculate_joint_rom(kinematic_data, "ankle_left")
        
        # Classify based on ROM thresholds
        if knee_rom["sagittal"] < 30:
            classification["knee_rom_class"] = "Reduced (stiffness)"
        elif knee_rom["sagittal"] > 80:
            classification["knee_rom_class"] = "Excessive (hyperextension)"
        else:
            classification["knee_rom_class"] = "Normal"
        
        return classification
    
    def _calculate_joint_rom(self, kinematic_data, joint_name: str) -> Dict:
        """Calculate ROM for a specific joint."""
        joint = getattr(kinematic_data, joint_name)
        
        return {
            "sagittal": float(np.max(joint.angles[:, 0]) - np.min(joint.angles[:, 0])),
            "frontal": float(np.max(joint.angles[:, 1]) - np.min(joint.angles[:, 1])),
            "transverse": float(np.max(joint.angles[:, 2]) - np.min(joint.angles[:, 2])),
        }
    
    def _classify_by_temporal_features(self, kinematic_data) -> Dict:
        """
        Classify based on temporal features.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Dictionary with temporal classification
        """
        classification = {}
        
        # Calculate variability
        variability = self._calculate_trajectory_variability(kinematic_data)
        
        if variability > 0.5:
            classification["variability_class"] = "High (inconsistent)"
        elif variability < 0.2:
            classification["variability_class"] = "Low (consistent)"
        else:
            classification["variability_class"] = "Normal"
        
        return classification
    
    def _calculate_trajectory_variability(self, kinematic_data) -> float:
        """Calculate overall trajectory variability."""
        joints = ["pelvis", "hip_left", "knee_left", "ankle_left"]
        
        total_variability = 0
        count = 0
        
        for joint_name in joints:
            joint = getattr(kinematic_data, joint_name)
            
            for i in range(3):  # Each plane
                std_val = np.std(joint.angles[:, i])
                mean_val = np.mean(np.abs(joint.angles[:, i]))
                
                if mean_val > 0:
                    cv = std_val / mean_val
                    total_variability += cv
                    count += 1
        
        if count > 0:
            return total_variability / count
        
        return 0.0


class GaitPatternAnalyzer:
    """Analyze characteristic gait patterns."""
    
    def __init__(self):
        """Initialize gait pattern analyzer."""
        pass
    
    def detect_crouch_gait(self, kinematic_data) -> float:
        """
        Detect crouch gait pattern.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Crouch gait score (0-1)
        """
        # Crouch gait is characterized by excessive knee and hip flexion
        knee_rom = self._calculate_joint_rom(kinematic_data, "knee_left")
        hip_rom = self._calculate_joint_rom(kinematic_data, "hip_left")
        
        # Calculate crouch score based on excessive flexion
        crouch_score = 0.0
        
        # Check for excessive knee flexion
        if knee_rom["sagittal"] > 60:
            crouch_score += 0.5
        
        # Check for excessive hip flexion
        if hip_rom["sagittal"] > 50:
            crouch_score += 0.5
        
        return min(crouch_score, 1.0)
    
    def detect_stiff_knee_gait(self, kinematic_data) -> float:
        """
        Detect stiff knee gait pattern.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Stiff knee score (0-1)
        """
        knee_rom = self._calculate_joint_rom(kinematic_data, "knee_left")
        
        # Stiff knee is characterized by reduced knee flexion
        if knee_rom["sagittal"] < 20:
            return 1.0
        elif knee_rom["sagittal"] < 30:
            return 0.7
        elif knee_rom["sagittal"] < 40:
            return 0.4
        else:
            return 0.0
    
    def detect_asymmetric_gait(self, kinematic_data) -> float:
        """
        Detect asymmetric gait pattern.
        
        Args:
            kinematic_data: Processed kinematic data
            
        Returns:
            Asymmetry score (0-1)
        """
        # Calculate symmetry between left and right
        symmetry_analyzer = None
        try:
            from ..analysis.symmetry_analysis import SymmetryAnalyzer
            symmetry_analyzer = SymmetryAnalyzer()
        except ImportError:
            return 0.0
        
        if symmetry_analyzer:
            symmetry_results = symmetry_analyzer.analyze_all_joints_symmetry(kinematic_data)
            
            # Calculate overall asymmetry
            asymmetry_scores = [s.asymmetry_score for s in symmetry_results.values()]
            
            if asymmetry_scores:
                return np.mean(asymmetry_scores)
        
        return 0.0
    
    def _calculate_joint_rom(self, kinematic_data, joint_name: str) -> Dict:
        """Calculate ROM for a specific joint."""
        joint = getattr(kinematic_data, joint_name)
        
        return {
            "sagittal": float(np.max(joint.angles[:, 0]) - np.min(joint.angles[:, 0])),
            "frontal": float(np.max(joint.angles[:, 1]) - np.min(joint.angles[:, 1])),
            "transverse": float(np.max(joint.angles[:, 2]) - np.min(joint.angles[:, 2])),
        }
