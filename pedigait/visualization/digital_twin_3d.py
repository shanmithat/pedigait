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


def normalize_mocap_markers(markers: np.ndarray, labels: List[str], n_frames: int = 60) -> Tuple[np.ndarray, List[str]]:
    """
    Extract active walking window, infill tracking dropouts, align facing orientation to +X,
    ground anchor to Z=0.0, and center pelvis horizontally at (0, 0).
    """
    m = markers.copy()
    labels_clean = [str(l).strip().upper() for l in labels]
    
    # 1. Active window detection (frames where >= 12 markers exist)
    valid_counts = (np.abs(m[2]) > 0.005).sum(axis=0)
    valid_frames = np.where(valid_counts >= 12)[0]
    if len(valid_frames) == 0:
        valid_frames = np.arange(m.shape[2])
    f_start, f_end = valid_frames[0], valid_frames[-1]
    if f_end - f_start < 25:
        f_start, f_end = 0, m.shape[2] - 1
        
    # Sample evenly across active window
    sample_indices = np.linspace(f_start, f_end, n_frames).astype(int)
    m_sampled = m[:, :, sample_indices]
    
    # Infill dropped markers within active window; mask completely unrecorded markers as NaN
    for mk in range(m_sampled.shape[1]):
        for dim in range(3):
            vals = m_sampled[dim, mk, :]
            bad = (np.abs(vals) < 0.001) | np.isnan(vals)
            if np.all(bad):
                m_sampled[:, mk, :] = np.nan
                break
            elif np.any(bad):
                good_idx = np.where(~bad)[0]
                m_sampled[dim, mk, bad] = np.interp(np.where(bad)[0], good_idx, vals[good_idx])
                
    # 2. Orientation heading alignment
    sacr_i = labels_clean.index('SACR') if 'SACR' in labels_clean else (labels_clean.index('LPSI') if 'LPSI' in labels_clean else None)
    lasi_i = labels_clean.index('LASI') if 'LASI' in labels_clean else None
    rasi_i = labels_clean.index('RASI') if 'RASI' in labels_clean else None
    
    f_mid = n_frames // 2
    if sacr_i is not None and lasi_i is not None and rasi_i is not None:
        ant_x = (m_sampled[0, lasi_i, f_mid] + m_sampled[0, rasi_i, f_mid])/2.0 - m_sampled[0, sacr_i, f_mid]
        ant_y = (m_sampled[1, lasi_i, f_mid] + m_sampled[1, rasi_i, f_mid])/2.0 - m_sampled[1, sacr_i, f_mid]
        heading_rad = np.arctan2(ant_y, ant_x)
    else:
        heading_rad = 0.0
        
    # Rotate around Z-axis by -heading_rad so anterior points along +X
    rot_angle = -heading_rad
    cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
    x_rot = cos_a * m_sampled[0] - sin_a * m_sampled[1]
    y_rot = sin_a * m_sampled[0] + cos_a * m_sampled[1]
    z_rot = m_sampled[2]
    
    # 3. Ground floor anchoring (lowest foot landmark = 0.0)
    z_valid = z_rot[~np.isnan(z_rot) & (z_rot > 0.001)]
    z_min = float(np.nanmin(z_valid)) if len(z_valid) > 0 else 0.0
    z_rot = np.where(np.isnan(z_rot), np.nan, np.maximum(0.0, z_rot - z_min))
    
    # 4. Pelvis horizontal centering per frame (focal spot centering)
    if lasi_i is not None and rasi_i is not None and sacr_i is not None:
        pelvis_x = (x_rot[lasi_i] + x_rot[rasi_i] + x_rot[sacr_i]) / 3.0
        pelvis_y = (y_rot[lasi_i] + y_rot[rasi_i] + y_rot[sacr_i]) / 3.0
        x_rot -= pelvis_x[np.newaxis, :]
        y_rot -= pelvis_y[np.newaxis, :]
    else:
        x_rot -= np.nanmean(x_rot, axis=1, keepdims=True)
        y_rot -= np.nanmean(y_rot, axis=1, keepdims=True)
        
    # 5. Filter spatial outliers (> 0.95m from horizontal body core)
    dist_xy = np.sqrt(x_rot**2 + y_rot**2)
    bad_dist = dist_xy > 0.95
    x_rot[bad_dist] = np.nan
    y_rot[bad_dist] = np.nan
    z_rot[bad_dist] = np.nan

    m_clean = np.stack([x_rot, y_rot, z_rot], axis=0)
    return m_clean, labels_clean


def create_side_by_side_digital_twin(loader_td: 'C3DLoader', loader_cp: 'C3DLoader',
                                     n_frames: int = 60, camera_mode: str = "sagittal") -> go.Figure:
    """
    Create an orientation-corrected, synchronized, side-by-side comparative 3D digital twin animation
    comparing typically developing (TD) and cerebral palsy (CP) gait with uniform view and positioning.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required for comparative visualization.")
        
    # Extract orientation-corrected, centered, ground-anchored markers
    m_td, l_td = normalize_mocap_markers(loader_td.get_marker_positions(), loader_td.get_marker_labels(), n_frames=n_frames)
    m_cp, l_cp = normalize_mocap_markers(loader_cp.get_marker_positions(), loader_cp.get_marker_labels(), n_frames=n_frames)
    
    skel_td = SkeletonBuilder(m_td, l_td)
    skel_cp = SkeletonBuilder(m_cp, l_cp)
    
    # Marker colors by anatomy
    def get_marker_colors(labels):
        colors = []
        for l in labels:
            if l.startswith("L") and not l.startswith("LB"):
                colors.append("#00f0ff")  # Cyan for Left
            elif l.startswith("R") and not l.startswith("RB"):
                colors.append("#ff4d6d")  # Coral for Right
            elif any(k in l for k in ["HD", "C7"]):
                colors.append("#ffd166")  # Gold for Cranial
            else:
                colors.append("#10b981")  # Emerald for Spine/Pelvis
        return colors

    colors_td = get_marker_colors(l_td)
    colors_cp = get_marker_colors(l_cp)
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(
            f"<b>TD Normative Reference:</b> {loader_td.c3d_path.stem}",
            f"<b>CP Pathological:</b> {loader_cp.c3d_path.stem}"
        ),
        horizontal_spacing=0.04
    )
    
    # Ground floor grid lines
    gx, gy, gz = [], [], []
    for g_pos in np.linspace(-0.6, 0.6, 7):
        gx.extend([g_pos, g_pos, None, -0.6, 0.6, None])
        gy.extend([-0.6, 0.6, None, g_pos, g_pos, None])
        gz.extend([0.0, 0.0, None, 0.0, 0.0, None])
        
    # Trace 0: TD Skeleton lines (Col 1)
    x_l_td, y_l_td, z_l_td = skel_td.build_skeleton_lines(0)
    fig.add_trace(go.Scatter3d(
        x=x_l_td, y=y_l_td, z=z_l_td, mode="lines",
        line=dict(color="#00f0ff", width=4.5), name="TD Skeleton", hoverinfo="skip"
    ), row=1, col=1)
    
    # Trace 1: TD Joints (Col 1)
    fig.add_trace(go.Scatter3d(
        x=m_td[0, :, 0], y=m_td[1, :, 0], z=m_td[2, :, 0], mode="markers",
        marker=dict(size=5.5, color=colors_td, opacity=0.95), text=l_td,
        name="TD Joints", hovertemplate="<b>%{text}</b><br>Z: %{z:.2f} m<extra></extra>"
    ), row=1, col=1)
    
    # Trace 2: TD Floor Grid (Col 1)
    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz, mode="lines",
        line=dict(color="rgba(71, 85, 105, 0.35)", width=1),
        hoverinfo="skip", showlegend=False
    ), row=1, col=1)
    
    # Trace 3: CP Skeleton lines (Col 2)
    x_l_cp, y_l_cp, z_l_cp = skel_cp.build_skeleton_lines(0)
    fig.add_trace(go.Scatter3d(
        x=x_l_cp, y=y_l_cp, z=z_l_cp, mode="lines",
        line=dict(color="#ff4d6d", width=4.5), name="CP Skeleton", hoverinfo="skip"
    ), row=1, col=2)
    
    # Trace 4: CP Joints (Col 2)
    fig.add_trace(go.Scatter3d(
        x=m_cp[0, :, 0], y=m_cp[1, :, 0], z=m_cp[2, :, 0], mode="markers",
        marker=dict(size=5.5, color=colors_cp, opacity=0.95), text=l_cp,
        name="CP Joints", hovertemplate="<b>%{text}</b><br>Z: %{z:.2f} m<extra></extra>"
    ), row=1, col=2)
    
    # Trace 5: CP Floor Grid (Col 2)
    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz, mode="lines",
        line=dict(color="rgba(71, 85, 105, 0.35)", width=1),
        hoverinfo="skip", showlegend=False
    ), row=1, col=2)
    
    # Synchronized frames (simultaneously updating traces [0, 1, 3, 4])
    frames = []
    for f in range(n_frames):
        x_td, y_td, z_td = skel_td.build_skeleton_lines(f)
        x_cp, y_cp, z_cp = skel_cp.build_skeleton_lines(f)
        frame = go.Frame(
            data=[
                go.Scatter3d(x=x_td, y=y_td, z=z_td),
                go.Scatter3d(x=m_td[0, :, f], y=m_td[1, :, f], z=m_td[2, :, f]),
                go.Scatter3d(x=x_cp, y=y_cp, z=z_cp),
                go.Scatter3d(x=m_cp[0, :, f], y=m_cp[1, :, f], z=m_cp[2, :, f])
            ],
            traces=[0, 1, 3, 4],
            name=f"frame_{f}"
        )
        frames.append(frame)
        
    fig.frames = frames
    
    # Camera definition (identical for both subplots)
    if "sagittal" in camera_mode.lower():
        # Lateral view looking along -Y: best for knee flexion & equinus
        cam = dict(eye=dict(x=0.0, y=-2.4, z=0.45), up=dict(x=0, y=0, z=1))
    elif "frontal" in camera_mode.lower():
        # Frontal view looking along +X: best for bilateral asymmetry
        cam = dict(eye=dict(x=2.4, y=0.0, z=0.45), up=dict(x=0, y=0, z=1))
    else:
        # 3D Isometric view
        cam = dict(eye=dict(x=1.7, y=-1.7, z=1.1), up=dict(x=0, y=0, z=1))
        
    scene_spec = dict(
        xaxis=dict(range=[-0.7, 0.7], title="", showgrid=True, gridcolor="#1e293b", zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.7, 0.7], title="", showgrid=True, gridcolor="#1e293b", zeroline=False, showticklabels=False),
        zaxis=dict(range=[0.0, 1.6], title="", showgrid=True, gridcolor="#1e293b", zeroline=True, zerolinecolor="#38bdf8", showticklabels=False),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1.4),
        camera=cam
    )
    
    fig.update_layout(
        paper_bgcolor="#070b14",
        plot_bgcolor="#070b14",
        height=680,
        margin=dict(l=5, r=5, t=50, b=10),
        scene=scene_spec,
        scene2=scene_spec,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5, font=dict(color="#e2e8f0")),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.04, y=0.02,
            buttons=[
                dict(label="▶ Play Synchronized Motion", method="animate",
                     args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ],
            bgcolor="#1e293b", font=dict(color="#ffffff", size=12),
            bordercolor="#38bdf8", borderwidth=1
        )]
    )
    return fig

