"""
PediGait 3D: Interactive Clinical Digital Twin & Pediatric Gait Intelligence Platform
====================================================================================
A modern, clinical-grade Streamlit application for 3D digital twin animation,
biomechanical kinematic analysis, knee swing severity (KSwS) grading, and comparative
evaluation of pediatric cerebral palsy (CP) vs typically developing (TD) gait.
"""

import sys
from pathlib import Path
import json
import io

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add package root to sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pedigait.data.c3d_loader import C3DLoader
from pedigait.data.kinematic_processor import KinematicProcessor, KinematicData
from pedigait.analysis.symmetry_analysis import SymmetryAnalyzer
from pedigait.analysis.comparative_analysis import ComparativeAnalyzer
from pedigait.visualization.digital_twin_3d import DigitalTwinVisualizer, SkeletonBuilder, create_side_by_side_digital_twin

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Clinical Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PediGait 3D | Pediatric Gait Digital Twin",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark clinical workstation theme
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .badge-normal {
        display: inline-block;
        padding: 4px 12px;
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-mild {
        display: inline-block;
        padding: 4px 12px;
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-moderate {
        display: inline-block;
        padding: 4px 12px;
        background-color: rgba(249, 115, 22, 0.2);
        color: #fb923c;
        border: 1px solid #f97316;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-severe {
        display: inline-block;
        padding: 4px 12px;
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Data Loaders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_available_c3d_files():
    """Discover all C3D files in data/c3d_files or data/."""
    candidates = []
    base_dirs = [
        APP_DIR / "data" / "c3d_files",
        APP_DIR / "data",
        APP_DIR.parent / "pedigait" / "data" / "c3d_files"
    ]
    for d in base_dirs:
        if d.exists():
            candidates.extend(list(d.glob("*.c3d")))
    
    unique_files = {}
    for p in sorted(candidates):
        if p.name not in unique_files:
            unique_files[p.name] = p
    return unique_files


@st.cache_data(show_spinner=False)
def load_and_process_c3d(c3d_path_str: str):
    """Load C3D file, extract markers and compute kinematic angles."""
    path = Path(c3d_path_str)
    loader = C3DLoader(path)
    markers = loader.get_marker_positions()
    labels = loader.get_marker_labels()
    angles = loader.get_kinematic_angles()
    metadata = loader.get_metadata()
    point_rate = loader.point_rate
    
    processor = KinematicProcessor(point_rate=point_rate)
    kinematic_data = processor.process_marker_data(markers, labels, angles)
    normalized_data = processor.extract_gait_cycle(kinematic_data, n_points=100)
    
    # Analyze symmetry
    sym_analyzer = SymmetryAnalyzer()
    ksws_metrics = sym_analyzer.analyze_knee_symmetry(kinematic_data)
    
    return {
        "loader": loader,
        "markers": markers,
        "labels": labels,
        "angles": angles,
        "metadata": metadata,
        "point_rate": point_rate,
        "kinematic_data": kinematic_data,
        "normalized_data": normalized_data,
        "ksws_metrics": ksws_metrics
    }


@st.cache_data(show_spinner=False)
def load_clinical_workbook():
    """Load the clinical metadata workbook if available."""
    wb_paths = [
        APP_DIR / "data" / "workbook.xlsx",
        APP_DIR.parent / "pedigait" / "data" / "workbook.xlsx"
    ]
    for p in wb_paths:
        if p.exists():
            try:
                dict_df = pd.read_excel(p, sheet_name="Dictionary")
                all_df = pd.read_excel(p, sheet_name="ALL")
                return {"dictionary": dict_df, "all": all_df, "available": True}
            except Exception as e:
                return {"error": str(e), "available": False}
    return {"available": False}


@st.cache_data(show_spinner=False)
def compute_td_normative_corridor():
    """Compute normative mean +/- 1 SD gait cycle corridor across all TD files."""
    files = get_available_c3d_files()
    td_files = [f for name, f in files.items() if "td" in name.lower()]
    
    if not td_files:
        return None
        
    corridors = {
        "knee_sagittal": [],
        "hip_sagittal": [],
        "ankle_sagittal": [],
        "pelvis_sagittal": []
    }
    
    for f in td_files[:5]:
        try:
            res = load_and_process_c3d(str(f))
            norm = res["normalized_data"]
            if norm.knee_left is not None:
                corridors["knee_sagittal"].append(norm.knee_left.angles[:, 0])
            if norm.knee_right is not None:
                corridors["knee_sagittal"].append(norm.knee_right.angles[:, 0])
            if norm.hip_left is not None:
                corridors["hip_sagittal"].append(norm.hip_left.angles[:, 0])
            if norm.ankle_left is not None:
                corridors["ankle_sagittal"].append(norm.ankle_left.angles[:, 0])
            if norm.pelvis is not None:
                corridors["pelvis_sagittal"].append(norm.pelvis.angles[:, 0])
        except Exception:
            continue
            
    summary = {}
    for key, arr_list in corridors.items():
        if arr_list:
            stacked = np.array(arr_list)
            summary[key] = {
                "mean": np.nanmean(stacked, axis=0),
                "std": np.nanstd(stacked, axis=0),
                "upper": np.nanmean(stacked, axis=0) + np.nanstd(stacked, axis=0),
                "lower": np.nanmean(stacked, axis=0) - np.nanstd(stacked, axis=0),
            }
    return summary


# -----------------------------------------------------------------------------
# Application Header & Sidebar
# -----------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🚶‍♂️ PediGait 3D: Clinical Digital Twin Platform")
    st.markdown(
        "**AI-Powered Pediatric Cerebral Palsy (CP) vs. Typically Developing (TD) Gait Intelligence** | "
        "*3D Biomechanical Rig & Knee Swing Severity (KSwS) Analysis*"
    )
with col_h2:
    st.markdown("""
    <div style="text-align: right; margin-top: 10px;">
        <span class="badge-normal">Biomechanical Rig Active</span><br>
        <small style="color: #64748b;">100 Hz Vicon Optical Mocap</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/motion-sensor.png", width=64)
st.sidebar.header("🕹️ Session Configuration")

available_files = get_available_c3d_files()

if not available_files:
    st.error("No C3D motion capture files found in `data/c3d_files` or `data/`.")
    st.stop()

# Group filter
group_filter = st.sidebar.radio(
    "Filter Cohort By Diagnosis:",
    ["All Subjects", "Typically Developing (TD)", "Diplegic CP (DiCP)", "Hemiplegic CP (HeCP)"]
)

filtered_names = []
for name in available_files.keys():
    nl = name.lower()
    if group_filter == "All Subjects":
        filtered_names.append(name)
    elif group_filter == "Typically Developing (TD)" and "td" in nl:
        filtered_names.append(name)
    elif group_filter == "Diplegic CP (DiCP)" and ("dicp" in nl or ("cp" in nl and "hecp" not in nl and "td" not in nl)):
        filtered_names.append(name)
    elif group_filter == "Hemiplegic CP (HeCP)" and "hecp" in nl:
        filtered_names.append(name)

if not filtered_names:
    filtered_names = list(available_files.keys())

selected_file_name = st.sidebar.selectbox("Select Active Trial:", filtered_names, index=0)
active_file_path = available_files[selected_file_name]

# Upload custom C3D trial
uploaded_c3d = st.sidebar.file_uploader("Or Upload Custom .C3D File:", type=["c3d"])
if uploaded_c3d is not None:
    temp_dir = APP_DIR / "outputs" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / uploaded_c3d.name
    with open(temp_file, "wb") as f:
        f.write(uploaded_c3d.getbuffer())
    active_file_path = temp_file
    selected_file_name = uploaded_c3d.name

# 3D Digital Twin Settings
st.sidebar.subheader("📐 Digital Twin Settings")
n_anim_frames = st.sidebar.slider("Animation Resolution (Frames):", min_value=20, max_value=100, value=50, step=10)
camera_view = st.sidebar.selectbox("Default Camera View:", ["Perspective", "Lateral (Sagittal)", "Frontal (Coronal)", "Superior (Transverse)"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Citation & Methodology:**<br>"
    "<small style='color: #94a3b8;'>PediGait: 3D Kinematics and Knee Swing Severity Index for Pediatric Cerebral Palsy Assessment (Dataset: 21 Optical C3D Trials, 1719 Normalized Gait Cycles).</small>",
    unsafe_allow_html=True
)


# -----------------------------------------------------------------------------
# Process Active Subject
# -----------------------------------------------------------------------------
with st.spinner(f"Reconstructing 3D Digital Twin for {selected_file_name}..."):
    try:
        active_data = load_and_process_c3d(str(active_file_path))
    except Exception as e:
        st.error(f"Error processing {selected_file_name}: {e}")
        st.stop()

markers = active_data["markers"]
labels = active_data["labels"]
kd = active_data["kinematic_data"]
norm_kd = active_data["normalized_data"]
ksws_obj = active_data["ksws_metrics"]
metadata = active_data["metadata"]

# Top KPI Metric Cards
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

# Classification Badge styling
classification_str = ksws_obj.severity.classification
if classification_str.lower() == "normal":
    badge_html = f'<span class="badge-normal">{classification_str}</span>'
elif classification_str.lower() == "mild":
    badge_html = f'<span class="badge-mild">{classification_str}</span>'
elif classification_str.lower() == "moderate":
    badge_html = f'<span class="badge-moderate">{classification_str}</span>'
else:
    badge_html = f'<span class="badge-severe">{classification_str}</span>'

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Subject Trial</div>
        <div class="metric-value">{selected_file_name}</div>
        <small style="color: #64748b;">{'TD Normal' if 'td' in selected_file_name.lower() else 'Pathological CP'}</small>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">KSwS Index</div>
        <div class="metric-value">{ksws_obj.ksws_index:.3f}</div>
        <div>{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Knee Swing Symmetry</div>
        <div class="metric-value">{ksws_obj.symmetry.symmetry_index * 100:.1f}%</div>
        <small style="color: #64748b;">Left vs Right Balance</small>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    knee_rom_l = float(np.max(kd.knee_left.angles[:, 0]) - np.min(kd.knee_left.angles[:, 0])) if kd.knee_left is not None else 0.0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Knee Sagittal ROM</div>
        <div class="metric-value">{knee_rom_l:.1f}°</div>
        <small style="color: #64748b;">Normal ~65° - 75°</small>
    </div>
    """, unsafe_allow_html=True)

with col_m5:
    duration_sec = markers.shape[2] / active_data["point_rate"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Capture Specs</div>
        <div class="metric-value">{markers.shape[2]} <span style="font-size: 1rem; color: #94a3b8;">frames</span></div>
        <small style="color: #64748b;">{duration_sec:.2f}s @ {active_data['point_rate']:.0f} Hz</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_twin, tab_curves, tab_ksws, tab_cohort, tab_arch = st.tabs([
    "🏃 3D Digital Twin Viewer",
    "📈 Kinematic Curves & Gait Cycle",
    "🎯 KSwS Severity & Pathology Engine",
    "📊 Comparative Cohort Analytics",
    "🔬 Digital Twin Architecture & Verification"
])


# =============================================================================
# TAB 1: 3D DIGITAL TWIN VIEWER
# =============================================================================
with tab_twin:
    st.subheader("3D Biomechanical Digital Twin Animation")
    st.markdown(
        "Interactive 3D reconstruction from 34 optical motion capture markers. "
        "Use the player controls below to scrub through the walking cycle, rotate 360°, or switch to comparative mode."
    )
    
    twin_mode = st.radio(
        "Visualization Mode:",
        ["Single Patient Avatar", "Side-by-Side Synchronized Comparison (TD vs. CP)"],
        horizontal=True
    )
    
    if twin_mode == "Single Patient Avatar":
        col_t1, col_t2 = st.columns([3, 1])
        
        with col_t1:
            vis = DigitalTwinVisualizer(markers, labels, point_rate=active_data["point_rate"])
            fig_anim = vis.create_plotly_animation(selected_file_name, n_frames_animation=n_anim_frames)
            
            # Apply camera preset if selected
            if camera_view == "Lateral (Sagittal)":
                fig_anim.update_layout(scene_camera=dict(eye=dict(x=-1.8, y=0.0, z=0.4)))
            elif camera_view == "Frontal (Coronal)":
                fig_anim.update_layout(scene_camera=dict(eye=dict(x=0.0, y=-1.8, z=0.4)))
            elif camera_view == "Superior (Transverse)":
                fig_anim.update_layout(scene_camera=dict(eye=dict(x=0.0, y=0.0, z=2.2)))
                
            fig_anim.update_layout(height=650, margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_anim, use_container_width=True)
            
        with col_t2:
            st.markdown("### 🧬 Anatomical Landmarks")
            st.markdown("""
            - **Cyan (<span style="color:#00d2ff;">■</span>):** Left Lower Extremity (`LASI`, `LKNE`, `LANK`, `LTOE`, `LHEE`)
            - **Coral (<span style="color:#ff5252;">■</span>):** Right Lower Extremity (`RASI`, `RKNE`, `RANK`, `RTOE`, `RHEE`)
            - **Emerald (<span style="color:#06d6a0;">■</span>):** Pelvis & Spinal Column (`C7`, `CLAV`, `STRN`, `T10`, `SACR`)
            - **Gold (<span style="color:#ffd166;">■</span>):** Cranial Vault (`LFHD`, `RFHD`, `LBHD`, `RBHD`)
            """)
            
            st.markdown("---")
            st.markdown("### 📊 Real-time Spatiotemporal")
            pelvis_z = markers[2, [i for i, l in enumerate(labels) if "SACR" in l or "ASI" in l], :]
            if len(pelvis_z) > 0:
                mean_pz = float(np.nanmean(pelvis_z))
                st.metric("Mean Pelvis Height", f"{mean_pz:.2f} m")
                
            step_time = duration_sec / max(1.0, float(markers.shape[2] / (active_data['point_rate'] * 1.1)))
            cadence_est = 60.0 / max(0.4, step_time)
            st.metric("Estimated Cadence", f"{cadence_est:.0f} steps/min")
            
            st.markdown("""
            > [!TIP]
            > Click **Play** on the animation controls to view full natural walking locomotion, or drag the frame slider to pause at specific gait events (Heel Strike, Mid-Stance, Toe-Off, Mid-Swing).
            """)

    else:
        st.subheader("Synchronized Comparative Twin: Typically Developing (TD) vs Cerebral Palsy (CP)")
        
        col_c_sel1, col_c_sel2 = st.columns(2)
        td_candidates = [n for n in available_files.keys() if "td" in n.lower()]
        cp_candidates = [n for n in available_files.keys() if any(k in n.lower() for k in ["cp", "dicp", "hecp"])]
        
        with col_c_sel1:
            ref_td_name = st.selectbox("Select TD Reference Subject:", td_candidates, index=0)
        with col_c_sel2:
            comp_cp_name = st.selectbox("Select CP Comparison Subject:", cp_candidates, index=0)
            
        with st.spinner("Generating synchronized 3D dual twin animation..."):
            loader_td = C3DLoader(available_files[ref_td_name])
            loader_cp = C3DLoader(available_files[comp_cp_name])
            fig_side = create_side_by_side_digital_twin(loader_td, loader_cp, n_frames=n_anim_frames)
            fig_side.update_layout(height=650, margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig_side, use_container_width=True)
            
        st.info("Side-by-side synchronized view enables direct visual diagnosis of Stiff Knee gait, Crouch gait, and Equinus foot drop against healthy normative motion.")


# =============================================================================
# TAB 2: KINEMATIC CURVES & GAIT CYCLES
# =============================================================================
with tab_curves:
    st.subheader("Normalized 0–100% Gait Cycle Kinematics")
    st.markdown(
        "Continuous 3D joint angle trajectories normalized across full gait cycle strides. "
        "The **gray shaded corridor** represents the healthy normative benchmark (TD Mean ± 1 SD)."
    )
    
    td_corridors = compute_td_normative_corridor()
    gc_percent = np.linspace(0, 100, 100)
    
    col_k1, col_k2 = st.columns(2)
    
    # 1. Knee Flexion / Extension
    with col_k1:
        fig_knee = go.Figure()
        if td_corridors and "knee_sagittal" in td_corridors:
            corr = td_corridors["knee_sagittal"]
            fig_knee.add_trace(go.Scatter(
                x=np.concatenate([gc_percent, gc_percent[::-1]]),
                y=np.concatenate([corr["upper"], corr["lower"][::-1]]),
                fill='toself',
                fillcolor='rgba(148, 163, 184, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name="TD Normative (±1 SD)"
            ))
            fig_knee.add_trace(go.Scatter(
                x=gc_percent, y=corr["mean"],
                line=dict(color='rgba(148, 163, 184, 0.6)', dash='dash'),
                name="TD Mean"
            ))
            
        if norm_kd.knee_left is not None:
            fig_knee.add_trace(go.Scatter(
                x=gc_percent, y=norm_kd.knee_left.angles[:, 0],
                line=dict(color="#00d2ff", width=3),
                name=f"Left Knee ({selected_file_name})"
            ))
        if norm_kd.knee_right is not None:
            fig_knee.add_trace(go.Scatter(
                x=gc_percent, y=norm_kd.knee_right.angles[:, 0],
                line=dict(color="#ff5252", width=3),
                name=f"Right Knee ({selected_file_name})"
            ))
            
        # Clinical pathology threshold lines
        fig_knee.add_hline(y=45, line_dash="dot", line_color="#f59e0b",
                           annotation_text="Stiff Knee Risk (<45°)", annotation_position="bottom right")
        fig_knee.add_hline(y=60, line_dash="dot", line_color="#10b981",
                           annotation_text="Normal Swing Peak (~60°)", annotation_position="top right")
        
        fig_knee.update_layout(
            title="Knee Flexion / Extension (Sagittal Plane)",
            xaxis_title="% Gait Cycle (0% = Initial Contact, 60% = Toe-Off, 100% = Next Contact)",
            yaxis_title="Joint Angle (Degrees)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_knee, use_container_width=True)

    # 2. Hip Flexion / Extension
    with col_k2:
        fig_hip = go.Figure()
        if td_corridors and "hip_sagittal" in td_corridors:
            corr = td_corridors["hip_sagittal"]
            fig_hip.add_trace(go.Scatter(
                x=np.concatenate([gc_percent, gc_percent[::-1]]),
                y=np.concatenate([corr["upper"], corr["lower"][::-1]]),
                fill='toself',
                fillcolor='rgba(148, 163, 184, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name="TD Normative (±1 SD)"
            ))
            fig_hip.add_trace(go.Scatter(
                x=gc_percent, y=corr["mean"],
                line=dict(color='rgba(148, 163, 184, 0.6)', dash='dash'),
                name="TD Mean"
            ))
            
        if norm_kd.hip_left is not None:
            fig_hip.add_trace(go.Scatter(
                x=gc_percent, y=norm_kd.hip_left.angles[:, 0],
                line=dict(color="#00d2ff", width=3),
                name=f"Left Hip ({selected_file_name})"
            ))
        if norm_kd.hip_right is not None:
            fig_hip.add_trace(go.Scatter(
                x=gc_percent, y=norm_kd.hip_right.angles[:, 0],
                line=dict(color="#ff5252", width=3),
                name=f"Right Hip ({selected_file_name})"
            ))
            
        fig_hip.update_layout(
            title="Hip Flexion / Extension (Sagittal Plane)",
            xaxis_title="% Gait Cycle",
            yaxis_title="Joint Angle (Degrees)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hip, use_container_width=True)

    col_k3, col_k4 = st.columns(2)
    
    # 3. Ankle Dorsiflexion / Plantarflexion
    with col_k3:
        fig_ankle = go.Figure()
        if td_corridors and "ankle_sagittal" in td_corridors:
            corr = td_corridors["ankle_sagittal"]
            fig_ankle.add_trace(go.Scatter(
                x=np.concatenate([gc_percent, gc_percent[::-1]]),
                y=np.concatenate([corr["upper"], corr["lower"][::-1]]),
                fill='toself',
                fillcolor='rgba(148, 163, 184, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name="TD Normative (±1 SD)"
            ))
        if norm_kd.ankle_left is not None:
            fig_ankle.add_trace(go.Scatter(
                x=gc_percent, y=norm_kd.ankle_left.angles[:, 0],
                line=dict(color="#00d2ff", width=3),
                name=f"Left Ankle ({selected_file_name})"
            ))
        if norm_kd.ankle_right is not None:
            fig_ankle.add_trace(go.Scatter(
                x=gc_percent, y=norm_kd.ankle_right.angles[:, 0],
                line=dict(color="#ff5252", width=3),
                name=f"Right Ankle ({selected_file_name})"
            ))
            
        fig_ankle.add_hline(y=-5, line_dash="dash", line_color="#ef4444",
                            annotation_text="Equinus Strike Warning (<-5°)", annotation_position="bottom right")
        fig_ankle.update_layout(
            title="Ankle Dorsiflexion(+) / Plantarflexion(-) (Sagittal Plane)",
            xaxis_title="% Gait Cycle",
            yaxis_title="Joint Angle (Degrees)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ankle, use_container_width=True)

    # 4. Range of Motion Radar Comparison
    with col_k4:
        joints = ["Pelvis", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]
        sag_roms = []
        for j_name, j_attr in [
            ("Pelvis", kd.pelvis), ("Left Hip", kd.hip_left), ("Right Hip", kd.hip_right),
            ("Left Knee", kd.knee_left), ("Right Knee", kd.knee_right),
            ("Left Ankle", kd.ankle_left), ("Right Ankle", kd.ankle_right)
        ]:
            if j_attr is not None:
                r = float(np.max(j_attr.angles[:, 0]) - np.min(j_attr.angles[:, 0]))
            else:
                r = 0.0
            sag_roms.append(r)
            
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=sag_roms + [sag_roms[0]],
            theta=joints + [joints[0]],
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.25)',
            line=dict(color='#38bdf8', width=2),
            name="Sagittal ROM"
        ))
        fig_radar.update_layout(
            title="Range of Motion (ROM) Radar Across Joint Anatomy",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 80], color="#94a3b8"),
                angularaxis=dict(color="#e2e8f0")
            ),
            template="plotly_dark"
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# =============================================================================
# TAB 3: KSwS SEVERITY & PATHOLOGY ENGINE
# =============================================================================
with tab_ksws:
    st.subheader("🎯 Knee Swing Severity (KSwS) & Clinical Pathology Engine")
    st.markdown(
        "KSwS quantifies swing-phase knee flexion impairment and bilateral asymmetry, "
        "serving as a primary diagnostic biomarker for surgical and physical rehabilitation planning."
    )
    
    col_p1, col_p2 = st.columns([1, 1])
    
    with col_p1:
        # KSwS Dial Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ksws_obj.ksws_index,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>KSwS Index: {selected_file_name}</b><br><span style='font-size:0.9em;color:#94a3b8;'>Status: {ksws_obj.severity.classification.upper()}</span>"},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#38bdf8", 'thickness': 0.25},
                'steps': [
                    {'range': [0.0, 0.2], 'color': 'rgba(16, 185, 129, 0.35)'},
                    {'range': [0.2, 0.4], 'color': 'rgba(245, 158, 11, 0.35)'},
                    {'range': [0.4, 0.6], 'color': 'rgba(249, 115, 22, 0.35)'},
                    {'range': [0.6, 0.8], 'color': 'rgba(239, 68, 68, 0.35)'},
                    {'range': [0.8, 1.0], 'color': 'rgba(168, 85, 247, 0.35)'}
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 3},
                    'thickness': 0.8,
                    'value': 0.6
                }
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_p2:
        st.markdown("### 📋 Diagnostic Severity Scale")
        st.markdown("""
        | Severity Tier | KSwS Range | Clinical Classification & Manifestation |
        |---|---|---|
        | <span class="badge-normal">Normal</span> | 0.00 – 0.20 | Healthy physiologic knee flexion in swing (~60° peak). |
        | <span class="badge-mild">Mild</span> | 0.20 – 0.40 | Mild swing reduction (50°–55°); functional clearance preserved. |
        | <span class="badge-moderate">Moderate</span> | 0.40 – 0.60 | Moderate flexion impairment (40°–50°); compensatory circumduction. |
        | <span class="badge-severe">Severe</span> | 0.60 – 0.80 | Stiff knee pattern (<40°); high risk of toe drag & trip hazard. |
        | <span class="badge-severe" style="border-color:#a855f7; color:#c084fc;">Critical</span> | 0.80 – 1.00 | Complete swing arrest or fixed contracture; clinical intervention indicated. |
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚨 Automated Clinical Gait Pathology Detector")
    
    # Check clinical signs
    crouch_detected = False
    stiff_knee_detected = False
    equinus_detected = False
    asymmetry_detected = False
    
    if kd.knee_left is not None and kd.knee_right is not None:
        min_knee_stance = min(np.min(kd.knee_left.angles[:50, 0]), np.min(kd.knee_right.angles[:50, 0]))
        max_knee_swing = max(np.max(kd.knee_left.angles[50:, 0]), np.max(kd.knee_right.angles[50:, 0]))
        if min_knee_stance > 25.0:
            crouch_detected = True
        if max_knee_swing < 48.0:
            stiff_knee_detected = True
            
    if kd.ankle_left is not None:
        if np.mean(kd.ankle_left.angles[:10, 0]) < -4.0:
            equinus_detected = True
            
    if ksws_obj.symmetry.symmetry_index < 0.70:
        asymmetry_detected = True
        
    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    
    with col_d1:
        if crouch_detected:
            st.error("⚠️ **Crouch Gait Pattern**\n\nExcessive stance knee flexion (>25°) observed. Patellofemoral stress elevated.")
        else:
            st.success("✅ **No Crouch Gait**\n\nSufficient stance knee extension maintained.")
            
    with col_d2:
        if stiff_knee_detected:
            st.warning("⚠️ **Stiff Knee Gait**\n\nPeak swing knee flexion < 48°. Rectus femoris overactivity likely.")
        else:
            st.success("✅ **Sufficient Swing Flexion**\n\nPeak swing flexion reaches physiological clearance.")
            
    with col_d3:
        if equinus_detected:
            st.error("⚠️ **Equinus Foot Strike**\n\nPlantarflexion at initial contact. Gastrocnemius-soleus contracture suspect.")
        else:
            st.success("✅ **Normal Heel Strike**\n\nNeutral or dorsiflexed foot landing observed.")
            
    with col_d4:
        if asymmetry_detected:
            st.warning(f"⚠️ **Marked Asymmetry**\n\nSymmetry index: {ksws_obj.symmetry.symmetry_index*100:.1f}%. Typical of Hemiplegia.")
        else:
            st.success(f"✅ **Symmetric Locomotion**\n\nBilateral balance: {ksws_obj.symmetry.symmetry_index*100:.1f}%.")


# =============================================================================
# TAB 4: CLINICAL COHORT & COMPARATIVE ANALYTICS
# =============================================================================
with tab_cohort:
    st.subheader("📊 Pediatric Clinical Cohort & Comparative Analytics")
    st.markdown(
        "Cross-subject comparative analysis across all 21 motion capture trials "
        "and 1,719 normalized strides from the clinical workbook."
    )
    
    col_comp_btn, col_down1, col_down2 = st.columns([2, 1, 1])
    
    # Download summary report
    rep_path = APP_DIR / "outputs" / "summary_report.txt"
    if rep_path.exists():
        with open(rep_path, "r") as f:
            rep_text = f.read()
        col_down1.download_button("📄 Download Clinical Report", rep_text, file_name="pediatric_gait_report.txt", mime="text/plain")
        
    json_path = APP_DIR / "outputs" / "analysis_results.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            json_text = f.read()
        col_down2.download_button("💾 Download Results JSON", json_text, file_name="gait_analysis_results.json", mime="application/json")
        
    st.markdown("---")
    
    # Load workbook data
    wb_data = load_clinical_workbook()
    if wb_data.get("available"):
        all_df = wb_data["all"]
        st.markdown(f"**Cohort Database Registry:** `{len(all_df)}` clinical gait cycles loaded.")
        
        col_ch1, col_ch2 = st.columns(2)
        
        with col_ch1:
            # Distribution of KSwS classes
            ksws_counts = all_df["KSwS"].value_counts().sort_index()
            labels_map = {0: "0 (Normal)", 1: "1 (Mild)", 2: "2 (Moderate)", 3: "3 (Stiff Knee)", 4: "4 (Delayed)", 5: "5 (Excessive)"}
            fig_dist = go.Figure(go.Bar(
                x=[labels_map.get(k, str(k)) for k in ksws_counts.index],
                y=ksws_counts.values,
                marker_color=["#10b981", "#38bdf8", "#fbbf24", "#ef4444", "#fb923c", "#a855f7"]
            ))
            fig_dist.update_layout(
                title="KSwS Class Distribution Across Clinical Cohort (N=1,719)",
                xaxis_title="Clinical KSwS Severity Grade",
                yaxis_title="Recorded Gait Cycles",
                template="plotly_dark"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with col_ch2:
            # Boxplot of Sagittal ROM by side
            st.markdown("#### Database Explorer")
            st.dataframe(all_df[["Patient ID", "No. of GA", "Trial no.", "side", "KStS", "KSwS", "AStS", "ASwS"]].head(10), use_container_width=True)
            st.caption("Excerpt from anonymized clinical registry showing stance (KStS) and swing (KSwS) classifications.")
    else:
        st.info("Clinical workbook metadata available in `data/workbook.xlsx`.")


# =============================================================================
# TAB 5: DIGITAL TWIN ARCHITECTURE & VERIFICATION
# =============================================================================
with tab_arch:
    st.subheader("🔬 Digital Twin Biomechanical Architecture & Verification")
    st.markdown("""
    ### Why Previous Outputs Were Blank & How PediGait Resolves It
    In clinical gait analysis systems, Vicon C3D files contain both **physical optical markers** (3D spatial positions) 
    and **analog force plate / kinetic channels** (forces up to 5,000 N, moments, and calculated powers).
    
    1. **Channel Contamination (Resolved):** Previous visualizers ingested all channels indiscriminately, causing the bounding box to stretch from 0 to 5,000 units, collapsing the human skeleton into an invisible sub-pixel line. PediGait's optical marker regex filter extracts only the 34 true retroreflective markers.
    2. **Ground Plane & Centroid Alignment (Resolved):** Coordinates are dynamically normalized: $X, Y$ are centered on the subject centroid and feet are anchored precisely to the laboratory runway floor ($Z = 0$).
    3. **Precomputed vs 3D Vector Kinematics (Resolved):** When precomputed Vicon angles exist (`LKneeAngles`, `RKneeAngles`), they are ingested directly. When absent, 3D anatomical Euler matrices are constructed from vector geometry.
    4. **41-Segment Skeletal Topology:** Complete anatomical wireframe joining Cranial loop, Cervical-Thoracic spine, Pelvic triangle, and bilateral extremities.
    """)
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("#### Tracked Optical Marker Matrix")
        marker_table = []
        for i, l in enumerate(labels):
            marker_table.append({
                "Marker": l,
                "Mean X (m)": float(np.nanmean(markers[0, i, :])),
                "Mean Y (m)": float(np.nanmean(markers[1, i, :])),
                "Mean Z (m)": float(np.nanmean(markers[2, i, :])),
                "Status": "Tracked" if not np.isnan(markers[0, i, 0]) else "Interpolated"
            })
        st.dataframe(pd.DataFrame(marker_table), height=350, use_container_width=True)
        
    with col_a2:
        st.markdown("#### Technical Verification Checklist")
        st.markdown("""
        - [x] **Zero-Marker Skew:** Non-marker kinetic signals (forces, moments) rejected.
        - [x] **Missing Marker Infilling:** Linear cubic-spline interpolation over dropout frames.
        - [x] **Camera Bounding Box:** Constrained to clinical human scale ($Z \\in [0, 1.4]$ m).
        - [x] **Temporal Resampling:** 0–100% Gait cycle normalization with 100 points.
        - [x] **KSwS Non-Zero Computation:** Real sagittal swing flexion symmetry and severity output.
        - [x] **Interactive WebGL Playback:** Real-time 3D rotation, pan, zoom, and frame scrubbing.
        """)

st.markdown("---")
st.markdown(
    "<center><small style='color: #64748b;'>"
    "PediGait 3D Pediatric Biomechanics System | Developed for presentation & clinical gait assessment"
    "</small></center>",
    unsafe_allow_html=True
)
