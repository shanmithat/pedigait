"""
3D Digital Twin Visualizer Module

Creates interactive 3D visualizations and animations of skeletal motion from C3D motion capture data.
Renders full-body anatomical wireframe digital twins with realistic biomechanical bone connections.
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


class SkeletonBuilder:
    """Build full-body 3D anatomical skeletal wireframes from marker positions."""
    
    # Anatomically validated marker connections for Plug-in Gait
    SKELETON_CONNECTIONS = [
        # Head
        ("LFHD", "RFHD"), ("RFHD", "RBHD"), ("RBHD", "LBHD"), ("LBHD", "LFHD"),
        ("LFHD", "C7"), ("RFHD", "C7"), ("LBHD", "C7"), ("RBHD", "C7"),
        
        # Spine and Torso
        ("C7", "CLAV"), ("CLAV", "STRN"), ("C7", "T10"), ("STRN", "T10"), 
        ("T10", "SACR"), ("C7", "LBAK"),
        
        # Pelvis Triangle
        ("LASI", "RASI"), ("LASI", "SACR"), ("RASI", "SACR"),
        ("LASI", "LPSI"), ("RASI", "RPSI"), ("LPSI", "RPSI"),
        
        # Left Arm
        ("CLAV", "LSHO"), ("C7", "LSHO"), ("LSHO", "LELB"), 
        ("LELB", "LWRA"), ("LELB", "LWRB"), ("LWRA", "LFIN"), ("LWRB", "LFIN"),
        
        # Right Arm
        ("CLAV", "RSHO"), ("C7", "RSHO"), ("RSHO", "RELB"), 
        ("RELB", "RWRA"), ("RELB", "RWRB"), ("RWRA", "RFIN"), ("RWRB", "RFIN"),
        
        # Left Leg (Pelvis -> Thigh -> Knee -> Shank -> Ankle -> Heel -> Toe)
        ("LASI", "LKNE"), ("LKNE", "LANK"), ("LANK", "LHEE"), 
        ("LHEE", "LTOE"), ("LANK", "LTOE"),
        ("LASI", "LTHI"), ("LTHI", "LKNE"), ("LKNE", "LTIB"), ("LTIB", "LANK"),
        
        # Right Leg (Pelvis -> Thigh -> Knee -> Shank -> Ankle -> Heel -> Toe)
        ("RASI", "RKNE"), ("RKNE", "RANK"), ("RANK", "RHEE"), 
        ("RHEE", "RTOE"), ("RANK", "RTOE"),
        ("RASI", "RTHI"), ("RTHI", "RKNE"), ("RKNE", "RTIB"), ("RTIB", "RANK"),
    ]
    
    def __init__(self, markers: np.ndarray, labels: List[str]):
        """
        Initialize skeleton builder.
        
        Args:
            markers: Array of shape (3, n_markers, n_frames) with x, y, z coordinates
            labels: List of marker labels
        """
        self.markers = markers
        self.labels = [str(l).strip().upper() for l in labels]
        self.marker_to_index = {label: i for i, label in enumerate(self.labels)}
        
    def get_connections(self) -> List[Tuple[int, int]]:
        """Get skeleton connections as marker index pairs."""
        connections = []
        for m1, m2 in self.SKELETON_CONNECTIONS:
            if m1 in self.marker_to_index and m2 in self.marker_to_index:
                connections.append((
                    self.marker_to_index[m1],
                    self.marker_to_index[m2]
                ))
        return connections
    
    def build_skeleton_lines(self, frame_idx: int) -> Tuple[List, List, List]:
        """
        Build skeleton line coordinates for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Tuple of lists (x_coords, y_coords, z_coords) with None separators
        """
        connections = self.get_connections()
        x_lines, y_lines, z_lines = [], [], []
        
        for i, j in connections:
            x_lines.extend([self.markers[0, i, frame_idx], self.markers[0, j, frame_idx], None])
            y_lines.extend([self.markers[1, i, frame_idx], self.markers[1, j, frame_idx], None])
            z_lines.extend([self.markers[2, i, frame_idx], self.markers[2, j, frame_idx], None])
            
        return x_lines, y_lines, z_lines


class DigitalTwinVisualizer:
    """Create interactive 3D digital twin visualizations and animations."""
    
    def __init__(self, markers: np.ndarray, labels: List[str], 
                 point_rate: float = 100.0, units: str = "m"):
        """
        Initialize digital twin visualizer.
        
        Args:
            markers: Array of shape (3, n_markers, n_frames)
            labels: List of marker labels
            point_rate: Sampling rate in Hz
            units: Coordinate units ('m')
        """
        self.markers = markers
        self.labels = [str(l).strip() for l in labels]
        self.point_rate = point_rate if point_rate and point_rate > 0 else 100.0
        self.units = units
        self.n_frames = markers.shape[2]
        
        self.skeleton = SkeletonBuilder(markers, self.labels)
        
        # Calculate human-scale bounding box
        self.x_min = float(np.nanmin(markers[0])) - 0.2
        self.x_max = float(np.nanmax(markers[0])) + 0.2
        self.y_min = float(np.nanmin(markers[1])) - 0.2
        self.y_max = float(np.nanmax(markers[1])) + 0.2
        self.z_min = max(0.0, float(np.nanmin(markers[2])) - 0.05)
        self.z_max = float(np.nanmax(markers[2])) + 0.15
        
    def _create_marker_trace(self, frame_idx: int, color_theme: str = "turbo") -> go.Scatter3d:
        """Create marker scatter trace for a given frame."""
        p = self.markers[:, :, frame_idx]
        
        # Categorize marker colors by anatomy
        colors = []
        for l in self.labels:
            lu = l.upper()
            if lu.startswith("L"):
                colors.append("#00d2ff")  # Electric cyan for Left
            elif lu.startswith("R"):
                colors.append("#ff5252")  # Bright coral for Right
            elif any(k in lu for k in ["HD", "C7"]):
                colors.append("#ffd166")  # Warm gold for Head/Neck
            else:
                colors.append("#06d6a0")  # Emerald for Torso/Spine
                
        return go.Scatter3d(
            x=p[0],
            y=p[1],
            z=p[2],
            mode="markers",
            marker=dict(
                size=6,
                color=colors,
                opacity=0.95,
                line=dict(color="#ffffff", width=1)
            ),
            text=self.labels,
            hovertemplate="<b>%{text}</b><br>X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Z: %{z:.2f} m<extra></extra>",
            name="Anatomical Joints"
        )
    
    def _create_skeleton_trace(self, frame_idx: int) -> go.Scatter3d:
        """Create skeleton wireframe line trace for a given frame."""
        x_lines, y_lines, z_lines = self.skeleton.build_skeleton_lines(frame_idx)
        return go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(color="#ffffff", width=4),
            hoverinfo="skip",
            name="Skeletal Wireframe"
        )
        
    def _create_floor_grid_trace(self) -> go.Scatter3d:
        """Create laboratory runway ground grid at Z = 0."""
        gx, gy, gz = [], [], []
        x_ticks = np.linspace(self.x_min, self.x_max, 9)
        y_ticks = np.linspace(self.y_min, self.y_max, 5)
        
        for xt in x_ticks:
            gx.extend([xt, xt, None])
            gy.extend([self.y_min, self.y_max, None])
            gz.extend([0.0, 0.0, None])
            
        for yt in y_ticks:
            gx.extend([self.x_min, self.x_max, None])
            gy.extend([yt, yt, None])
            gz.extend([0.0, 0.0, None])
            
        return go.Scatter3d(
            x=gx, y=gy, z=gz,
            mode="lines",
            line=dict(color="rgba(100, 120, 150, 0.25)", width=1),
            hoverinfo="skip",
            name="Ground Plane"
        )
        
    def create_plotly_animation(self, trial_name: str = "trial",
                               n_frames_animation: int = 60) -> go.Figure:
        """
        Create an interactive, high-fidelity 3D Plotly animation of the digital twin.
        
        Args:
            trial_name: Name of the motion trial
            n_frames_animation: Target number of sampled animation frames
            
        Returns:
            Plotly Figure object
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for this visualization")
            
        frame_ids = np.unique(
            np.linspace(0, self.n_frames - 1, min(n_frames_animation, self.n_frames)).astype(int)
        )
        
        fig = go.Figure()
        
        # Base traces (Frame 0)
        fig.add_trace(self._create_floor_grid_trace())
        fig.add_trace(self._create_skeleton_trace(frame_ids[0]))
        fig.add_trace(self._create_marker_trace(frame_ids[0]))
        
        # Animation frames
        frames = []
        for idx in frame_ids:
            frame = go.Frame(
                data=[
                    self._create_floor_grid_trace(),
                    self._create_skeleton_trace(idx),
                    self._create_marker_trace(idx)
                ],
                name=str(int(idx))
            )
            frames.append(frame)
            
        fig.frames = frames
        
        # Modern clinical presentation styling
        fig.update_layout(
            title=dict(
                text=f"<b>3D Digital Twin Motion Capture</b> &nbsp;|&nbsp; <i>{trial_name}</i>",
                font=dict(size=18, color="#f8f9fa"),
                x=0.03, y=0.96
            ),
            paper_bgcolor="#11151c",
            plot_bgcolor="#11151c",
            scene=dict(
                xaxis=dict(
                    title="X (Progression / m)",
                    range=[self.x_min, self.x_max],
                    backgroundcolor="#11151c",
                    gridcolor="#283040",
                    showbackground=True,
                    color="#90a4ae"
                ),
                yaxis=dict(
                    title="Y (Lateral / m)",
                    range=[self.y_min, self.y_max],
                    backgroundcolor="#11151c",
                    gridcolor="#283040",
                    showbackground=True,
                    color="#90a4ae"
                ),
                zaxis=dict(
                    title="Z (Height / m)",
                    range=[0.0, self.z_max],
                    backgroundcolor="#11151c",
                    gridcolor="#283040",
                    showbackground=True,
                    color="#90a4ae"
                ),
                aspectratio=dict(
                    x=max(1.0, (self.x_max - self.x_min) / 1.5),
                    y=max(0.6, (self.y_max - self.y_min) / 0.8),
                    z=max(1.0, self.z_max / 1.2)
                ),
                camera=dict(
                    eye=dict(x=1.8, y=-1.8, z=0.9),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            height=650,
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.02,
                xanchor="right",
                x=0.98,
                font=dict(color="#cfd8dc")
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.04,
                y=0.02,
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"frame": {"duration": 65, "redraw": True}, "fromcurrent": True}]),
                    dict(label="❚❚ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ],
                bgcolor="#1e2638",
                bordercolor="#37474f",
                font=dict(color="#ffffff")
            )],
            sliders=[dict(
                steps=[
                    dict(method="animate",
                         args=[[str(int(f))], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                         label=f"{f/self.point_rate:.2f}s")
                    for f in frame_ids
                ],
                currentvalue=dict(prefix="Gait Time: ", font=dict(color="#eceff1")),
                x=0.20, len=0.76, y=0.02,
                font=dict(color="#90a4ae"),
                bgcolor="#1e2638"
            )]
        )
        
        return fig
    
    def create_matplotlib_visualization(self, frame_idx: int = 0):
        """
        Create a clean, focused static 3D visualization using matplotlib.
        
        Args:
            frame_idx: Frame index to visualize
            
        Returns:
            Tuple of (fig, ax)
        """
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(9, 7), facecolor='#11151c')
        ax = fig.add_subplot(111, projection='3d', facecolor='#11151c')
        
        p = self.markers[:, :, frame_idx]
        
        # Color markers by limb
        colors = []
        for l in self.labels:
            lu = l.upper()
            if lu.startswith("L"):
                colors.append('#00d2ff')
            elif lu.startswith("R"):
                colors.append('#ff5252')
            else:
                colors.append('#ffd166')
                
        ax.scatter(p[0], p[1], p[2], c=colors, s=35, edgecolors='white', linewidths=0.5, depthshade=True)
        
        # Draw skeleton connections
        connections = self.skeleton.get_connections()
        for i, j in connections:
            ax.plot([p[0, i], p[0, j]], 
                    [p[1, i], p[1, j]], 
                    [p[2, i], p[2, j]], color='#ffffff', linewidth=2.0, alpha=0.9)
        
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(0.0, self.z_max)
        
        ax.set_xlabel('X (Progression / m)', color='#cfd8dc')
        ax.set_ylabel('Y (Lateral / m)', color='#cfd8dc')
        ax.set_zlabel('Z (Height / m)', color='#cfd8dc')
        ax.tick_params(colors='#90a4ae')
        ax.set_title(f'Pediatric Gait Digital Twin  Frame {frame_idx}', color='#ffffff', fontsize=12)
        ax.view_init(elev=20, azim=-60)
        
        plt.tight_layout()
        return fig, ax

    def save_animation(self, fig: go.Figure, output_path: str):
        """Save animation to standalone HTML file."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), include_plotlyjs=True)
        print(f"Animation saved to: {output_path}")


def create_side_by_side_digital_twin(loader_td: 'C3DLoader', loader_cp: 'C3DLoader',
                                     n_frames: int = 50) -> go.Figure:
    """
    Create a synchronized side-by-side comparative 3D digital twin animation
    comparing typically developing (TD) and cerebral palsy (CP) gait.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required for comparative visualization.")
        
    m_td = loader_td.get_marker_positions()
    l_td = loader_td.get_marker_labels()
    m_cp = loader_cp.get_marker_positions()
    l_cp = loader_cp.get_marker_labels()
    
    skel_td = SkeletonBuilder(m_td, l_td)
    skel_cp = SkeletonBuilder(m_cp, l_cp)
    
    min_frames = min(m_td.shape[2], m_cp.shape[2])
    frame_ids = np.linspace(0, min_frames - 1, n_frames).astype(int)
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(f"TD Gait: {loader_td.c3d_path.stem}", f"CP Gait: {loader_cp.c3d_path.stem}")
    )
    
    # Initial Frame 0 TD
    x_l_td, y_l_td, z_l_td = skel_td.build_skeleton_lines(frame_ids[0])
    fig.add_trace(go.Scatter3d(x=x_l_td, y=y_l_td, z=z_l_td, mode="lines",
                              line=dict(color="#00d2ff", width=4), name="TD Skeleton"), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=m_td[0, :, frame_ids[0]], y=m_td[1, :, frame_ids[0]], z=m_td[2, :, frame_ids[0]],
                              mode="markers", marker=dict(size=5, color="#ffffff"), text=l_td, name="TD Joints"), row=1, col=1)
                              
    # Initial Frame 0 CP
    x_l_cp, y_l_cp, z_l_cp = skel_cp.build_skeleton_lines(frame_ids[0])
    fig.add_trace(go.Scatter3d(x=x_l_cp, y=y_l_cp, z=z_l_cp, mode="lines",
                              line=dict(color="#ff5252", width=4), name="CP Skeleton"), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=m_cp[0, :, frame_ids[0]], y=m_cp[1, :, frame_ids[0]], z=m_cp[2, :, frame_ids[0]],
                              mode="markers", marker=dict(size=5, color="#ffd166"), text=l_cp, name="CP Joints"), row=1, col=2)
                              
    # Synchronized frames
    frames = []
    for idx in frame_ids:
        x_td, y_td, z_td = skel_td.build_skeleton_lines(idx)
        x_cp, y_cp, z_cp = skel_cp.build_skeleton_lines(idx)
        frame = go.Frame(
            data=[
                go.Scatter3d(x=x_td, y=y_td, z=z_td),
                go.Scatter3d(x=m_td[0, :, idx], y=m_td[1, :, idx], z=m_td[2, :, idx]),
                go.Scatter3d(x=x_cp, y=y_cp, z=z_cp),
                go.Scatter3d(x=m_cp[0, :, idx], y=m_cp[1, :, idx], z=m_cp[2, :, idx])
            ],
            name=str(idx)
        )
        frames.append(frame)
        
    fig.frames = frames
    
    fig.update_layout(
        paper_bgcolor="#11151c",
        height=650,
        margin=dict(l=10, r=10, t=50, b=10),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.04, y=0.02,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 70, "redraw": True}, "fromcurrent": True}]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ],
            bgcolor="#1e2638", font=dict(color="#ffffff")
        )]
    )
    return fig
