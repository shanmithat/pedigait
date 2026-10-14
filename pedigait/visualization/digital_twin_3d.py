"""
3D Digital Twin Visualizer Module

Creates interactive 3D visualizations of skeletal motion from C3D data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. 3D visualization will be limited.")

try:
    from mayavi import mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False
    warnings.warn("Mayavi not available. Advanced 3D visualization disabled.")


class SkeletonBuilder:
    """Build 3D skeleton from marker positions."""
    
    # Standard marker connections for skeleton
    SKELETON_CONNECTIONS = [
        # Pelvis
        ("LASI", "RASI"),
        ("LPSI", "RPSI"),
        ("LASI", "LPSI"),
        ("RASI", "RPSI"),
        
        # Left leg
        ("LASI", "LTHI"),
        ("LTHI", "LKNE"),
        ("LKNE", "LANK"),
        ("LANK", "LHEE"),
        ("LHEE", "LTOE"),
        
        # Right leg
        ("RASI", "RTHI"),
        ("RTHI", "RKNE"),
        ("RKNE", "RANK"),
        ("RANK", "RHEE"),
        ("RHEE", "RTOE"),
        
        # Additional connections
        ("LTHI", "LHFE"),
        ("RTHI", "RHFE"),
        ("LKNE", "LFEME"),
        ("RKNE", "RFEME"),
    ]
    
    def __init__(self, markers: np.ndarray, labels: List[str]):
        """
        Initialize skeleton builder.
        
        Args:
            markers: Array of shape (3, n_markers, n_frames) with x, y, z coordinates
            labels: List of marker labels
        """
        self.markers = markers
        self.labels = labels
        self.marker_to_index = {label: i for i, label in enumerate(labels)}
        
    def get_connections(self) -> List[Tuple[int, int]]:
        """Get skeleton connections as marker indices."""
        connections = []
        
        for marker1, marker2 in self.SKELETON_CONNECTIONS:
            if marker1 in self.marker_to_index and marker2 in self.marker_to_index:
                connections.append((
                    self.marker_to_index[marker1],
                    self.marker_to_index[marker2]
                ))
        
        return connections
    
    def build_skeleton_lines(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build skeleton line segments for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Tuple of (x_coords, y_coords, z_coords) for line segments
        """
        connections = self.get_connections()
        
        x_lines = []
        y_lines = []
        z_lines = []
        
        for i, j in connections:
            # Add a break between segments
            x_lines.extend([self.markers[0, i, frame_idx], 
                           self.markers[0, j, frame_idx],
                           None])
            y_lines.extend([self.markers[1, i, frame_idx], 
                           self.markers[1, j, frame_idx],
                           None])
            z_lines.extend([self.markers[2, i, frame_idx], 
                           self.markers[2, j, frame_idx],
                           None])
        
        return np.array(x_lines), np.array(y_lines), np.array(z_lines)


class DigitalTwinVisualizer:
    """Create 3D digital twin visualizations."""
    
    def __init__(self, markers: np.ndarray, labels: List[str], 
                 point_rate: float = 100.0, units: str = "m"):
        """
        Initialize digital twin visualizer.
        
        Args:
            markers: Array of shape (3, n_markers, n_frames) with x, y, z coordinates
            labels: List of marker labels
            point_rate: Sampling rate in Hz
            units: Coordinate units
        """
        self.markers = markers
        self.labels = labels
        self.point_rate = point_rate
        self.units = units
        self.n_frames = markers.shape[2]
        
        # Calculate stable plot bounds
        self.lim = float(np.nanpercentile(np.abs(markers), 99))
        self.lim = max(self.lim, 0.1)
        
        # Create skeleton builder
        self.skeleton = SkeletonBuilder(markers, labels)
        
    def create_plotly_animation(self, trial_name: str = "trial",
                               n_frames_animation: int = 70) -> go.Figure:
        """
        Create interactive Plotly animation of the digital twin.
        
        Args:
            trial_name: Name for the trial
            n_frames_animation: Number of frames in animation
            
        Returns:
            Plotly Figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for this visualization")
        
        # Select frames for animation
        frame_ids = np.unique(
            np.linspace(0, self.n_frames - 1, min(n_frames_animation, self.n_frames)).astype(int)
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add initial frame
        fig.add_trace(self._create_marker_trace(frame_ids[0]))
        fig.add_trace(self._create_skeleton_trace(frame_ids[0]))
        
        # Add frames for animation
        frames = []
        for frame_idx in frame_ids:
            frame = go.Frame(
                data=[
                    self._create_marker_trace(frame_idx),
                    self._create_skeleton_trace(frame_idx)
                ],
                name=str(int(frame_idx))
            )
            frames.append(frame)
        
        fig.update_layout(
            title=f"Motion-capture digital twin  {trial_name}",
            scene=dict(
                xaxis_title=f"X ({self.units})",
                yaxis_title=f"Y ({self.units})",
                zaxis_title=f"Z ({self.units})",
                aspectmode="data",
                xaxis=dict(range=[-self.lim, self.lim]),
                yaxis=dict(range=[-self.lim, self.lim]),
                zaxis=dict(range=[-self.lim, self.lim]),
            ),
            height=800,
            margin=dict(l=0, r=0, t=60, b=0),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=1.08,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 70, "redraw": True},
                                     "fromcurrent": True}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate"}])
                ]
            )],
            sliders=[dict(
                steps=[
                    dict(method="animate", 
                         args=[[str(int(f))], 
                              {"mode": "immediate",
                               "frame": {"duration": 0, "redraw": True}}],
                         label=str(i))
                    for i, f in enumerate(frame_ids)
                ],
                currentvalue=dict(prefix="Animation frame: ")
            )]
        )
        
        return fig
    
    def _create_marker_trace(self, frame_idx: int) -> go.Scatter3d:
        """Create marker trace for a specific frame."""
        p = self.markers[:, :, frame_idx]
        
        return go.Scatter3d(
            x=p[0],
            y=p[1],
            z=p[2],
            mode="markers+text",
            text=self.labels,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                size=4,
                color=p[2],
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="Z")
            ),
            hovertemplate="%{text}<br>X=%{x:.3f}<br>Y=%{y:.3f}<br>Z=%{z:.3f}<extra></extra>",
            name="Measured markers"
        )
    
    def _create_skeleton_trace(self, frame_idx: int) -> go.Scatter3d:
        """Create skeleton trace for a specific frame."""
        x_lines, y_lines, z_lines = self.skeleton.build_skeleton_lines(frame_idx)
        
        return go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(
                color="white",
                width=2
            ),
            name="Skeleton"
        )
    
    def create_comparative_visualization(self, markers2: np.ndarray, 
                                        labels2: List[str], 
                                        trial_name2: str = "CP") -> go.Figure:
        """
        Create side-by-side comparison of two trials.
        
        Args:
            markers2: Second set of markers for comparison
            labels2: Second set of labels
            trial_name2: Name for second trial
            
        Returns:
            Plotly Figure with comparative visualization
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for this visualization")
        
        from plotly.subplots import make_subplots
        
        # Create subplot figure
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=(f"TD: {self.labels[0]}", f"CP: {labels2[0]}")
        )
        
        # Add first trial
        fig.add_trace(
            go.Scatter3d(
                x=self.markers[0, :, 0],
                y=self.markers[1, :, 0],
                z=self.markers[2, :, 0],
                mode="markers",
                marker=dict(size=3, color="blue"),
                name="TD"
            ),
            row=1, col=1
        )
        
        # Add second trial
        fig.add_trace(
            go.Scatter3d(
                x=markers2[0, :, 0],
                y=markers2[1, :, 0],
                z=markers2[2, :, 0],
                mode="markers",
                marker=dict(size=3, color="red"),
                name="CP"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True
        )
        
        return fig
    
    def save_animation(self, fig: go.Figure, output_path: str):
        """
        Save animation to HTML file.
        
        Args:
            fig: Plotly Figure to save
            output_path: Output file path
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), include_plotlyjs=True)
        print(f"Animation saved to: {output_path}")
    
    def create_matplotlib_visualization(self, frame_idx: int = 0):
        """
        Create static 3D visualization using matplotlib.
        
        Args:
            frame_idx: Frame index to visualize
            
        Returns:
            Matplotlib figure and axes
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot markers
        p = self.markers[:, :, frame_idx]
        ax.scatter(p[0], p[1], p[2], c=p[2], cmap='viridis', s=20)
        
        # Plot skeleton
        connections = self.skeleton.get_connections()
        for i, j in connections:
            ax.plot([p[0, i], p[0, j]], 
                   [p[1, i], p[1, j]], 
                   [p[2, i], p[2, j]], 'k-', linewidth=1)
        
        # Add labels
        for i, label in enumerate(self.labels):
            ax.text(p[0, i], p[1, i], p[2, i], label, fontsize=8)
        
        ax.set_xlabel(f'X ({self.units})')
        ax.set_ylabel(f'Y ({self.units})')
        ax.set_zlabel(f'Z ({self.units})')
        ax.set_title(f'3D Digital Twin - Frame {frame_idx}')
        
        # Set equal aspect
        max_range = np.array([
            p[0].max()-p[0].min(),
            p[1].max()-p[1].min(),
            p[2].max()-p[2].min()
        ]).max() / 2.0
        
        mid_x = (p[0].max()+p[0].min()) * 0.5
        mid_y = (p[1].max()+p[1].min()) * 0.5
        mid_z = (p[2].max()+p[2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        return fig, ax


class GaitCycleVisualizer:
    """Visualize gait cycle trajectories and comparisons."""
    
    def __init__(self):
        """Initialize gait cycle visualizer."""
        pass
    
    def plot_joint_trajectories(self, kinematic_data, 
                                joint_names: List[str] = None) -> go.Figure:
        """
        Plot joint angle trajectories across gait cycle.
        
        Args:
            kinematic_data: Processed kinematic data
            joint_names: List of joint names to plot
            
        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for this visualization")
        
        if joint_names is None:
            joint_names = ["pelvis", "hip_left", "hip_right", 
                          "knee_left", "knee_right", 
                          "ankle_left", "ankle_right"]
        
        from plotly.subplots import make_subplots
        
        # Create subplots
        n_joints = len(joint_names)
        fig = make_subplots(
            rows=n_joints,
            cols=1,
            subplot_titles=joint_names,
            vertical_spacing=0.05
        )
        
        # Plot each joint
        for i, joint_name in enumerate(joint_names):
            joint = getattr(kinematic_data, joint_name)
            if joint is None:
                continue
            
            # Plot each angle component
            for j, plane in enumerate(["Sagittal", "Frontal", "Transverse"]):
                fig.add_trace(
                    go.Scatter(
                        x=joint.time,
                        y=joint.angles[:, j],
                        name=f"{joint_name} - {plane}",
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )
            
            fig.update_yaxes(title_text="Angle (degrees)", row=i+1, col=1)
        
        fig.update_xaxes(title_text="Time (s)", row=n_joints, col=1)
        fig.update_layout(height=300 * n_joints, showlegend=True)
        
        return fig
    
    def plot_comparative_trajectories(self, td_data, cp_data, 
                                     joint_name: str = "knee_left") -> go.Figure:
        """
        Plot comparative trajectories between TD and CP.
        
        Args:
            td_data: Typically Developing kinematic data
            cp_data: Cerebral Palsy kinematic data
            joint_name: Joint name to compare
            
        Returns:
            Plotly Figure with comparative visualization
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for this visualization")
        
        td_joint = getattr(td_data, joint_name)
        cp_joint = getattr(cp_data, joint_name)
        
        fig = go.Figure()
        
        # Plot TD trajectories
        for j, plane in enumerate(["Sagittal", "Frontal", "Transverse"]):
            fig.add_trace(
                go.Scatter(
                    x=td_joint.time,
                    y=td_joint.angles[:, j],
                    name=f"TD - {plane}",
                    line=dict(color="blue", width=2)
                )
            )
        
        # Plot CP trajectories
        for j, plane in enumerate(["Sagittal", "Frontal", "Transverse"]):
            fig.add_trace(
                go.Scatter(
                    x=cp_joint.time,
                    y=cp_joint.angles[:, j],
                    name=f"CP - {plane}",
                    line=dict(color="red", width=2, dash="dash")
                )
            )
        
        fig.update_layout(
            title=f"Comparative {joint_name} trajectories",
            xaxis_title="Time (s)",
            yaxis_title="Angle (degrees)",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_range_of_motion_comparison(self, td_rom: Dict, cp_rom: Dict) -> go.Figure:
        """
        Plot range of motion comparison between TD and CP.
        
        Args:
            td_rom: Range of motion for TD
            cp_rom: Range of motion for CP
            
        Returns:
            Plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for this visualization")
        
        fig = go.Figure()
        
        joints = list(td_rom.keys())
        planes = ["sagittal", "frontal", "transverse"]
        
        for plane in planes:
            td_values = [td_rom[joint][plane] for joint in joints]
            cp_values = [cp_rom[joint][plane] for joint in joints]
            
            fig.add_trace(
                go.Bar(
                    x=joints,
                    y=td_values,
                    name=f"TD - {plane.capitalize()}",
                    marker_color="blue"
                )
            )
            
            fig.add_trace(
                go.Bar(
                    x=joints,
                    y=cp_values,
                    name=f"CP - {plane.capitalize()}",
                    marker_color="red"
                )
            )
        
        fig.update_layout(
            title="Range of Motion Comparison",
            xaxis_title="Joint",
            yaxis_title="Range of Motion (degrees)",
            barmode="group",
            height=500
        )
        
        return fig
