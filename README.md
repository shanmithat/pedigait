# PediGait: 3D Digital Twin & Biomechanical Workstation

PediGait is an end-to-end computational framework for pediatric gait analysis using 3D optical motion-capture data. The project combines C3D processing, lower-limb kinematic analysis, interactive 3D digital-twin visualization, gait symmetry assessment, Knee Swing Symmetry/Severity (KSwS) analysis, TD-versus-CP comparison, and machine-learning-based gait classification.

> Research prototype: PediGait is intended for engineering research, biomechanical analysis, visualization, and decision-support research. It is not a clinically validated diagnostic device and should not be used as a substitute for assessment by a qualified clinician.

## Live Application

https://shanmithat.github.io/pedigait/

## Project Overview

PediGait provides a unified workflow for transforming motion-capture data into interpretable biomechanical information.

The platform supports:

- C3D motion-capture file processing
- Lower-limb and pelvis kinematic analysis
- Gait-cycle normalization
- Interactive 3D skeletal visualization
- Bilateral gait symmetry analysis
- Knee Swing Symmetry/Severity (KSwS) assessment
- TD versus CP comparative analysis
- Gait-pattern and pathology flagging
- Machine-learning-based gait classification
- Cohort-level analysis
- Exportable analysis results

## System Workflow

```text
C3D Motion-Capture Data
          |
          v
     C3D Loader
          |
          v
Marker Identification and Processing
          |
          v
   Kinematic Processor
          |
     +----+----+----------------+
     |         |                |
     v         v                v
Kinematics  Symmetry/KSwS   Marker Data
     |         |                |
     +---------+----------------+
               |
        +------+------+
        |             |
        v             v
  ML Classification  3D Digital Twin
        |             |
        +------+------+
               |
               v
       Interactive Analysis
```

## Key Features

### 1. Kinematic Analysis

The processing pipeline extracts and analyzes:

- Pelvis motion
- Hip motion
- Knee motion
- Ankle motion
- Foot motion
- Joint trajectories
- Joint angles
- Range of motion
- Angular kinematic information where available

The system can use pre-calculated joint-angle channels contained in C3D files and can also calculate biomechanical quantities from marker geometry.

### 2. C3D Motion-Capture Processing

PediGait uses `ezc3d` to process standard biomechanical C3D files.

The loader extracts:

- Optical marker positions
- Marker labels
- Point sampling frequency
- Available analog channels
- Measurement units
- Pre-calculated joint-angle channels when present

The processing pipeline distinguishes optical marker trajectories from non-marker channels such as forces, moments, powers, and calculated biomechanical signals.

### 3. 3D Digital Twin

The project provides an interactive 3D representation of recorded movement using optical motion-capture markers.

The digital twin supports:

- Interactive Plotly visualization
- Skeletal wireframe reconstruction
- Marker labels
- Frame-by-frame playback
- Interactive camera controls
- Multiple viewing orientations
- Side-by-side TD versus CP comparison
- Exportable HTML animations

### 4. Gait-Cycle Analysis

Joint trajectories can be normalized to a common 0-100% gait-cycle representation.

This allows gait patterns with different recording durations to be compared on a common temporal scale.

The application supports analysis of:

- Sagittal-plane motion
- Frontal-plane motion
- Transverse-plane motion
- Bilateral trajectories
- Range of motion
- Gait-cycle patterns

### 5. Symmetry Analysis

PediGait evaluates left-right gait symmetry using quantitative trajectory comparisons.

The analysis includes:

- Symmetry Index
- Asymmetry measures
- Pearson correlation
- Mean absolute differences
- Normalized trajectory error

The implemented symmetry calculations compare corresponding left and right gait trajectories.

### 6. Knee Swing Symmetry/Severity (KSwS)

KSwS is used to quantify knee-swing characteristics and bilateral gait differences.

The application provides a continuous KSwS index and severity classification.

The current classification bands implemented by the application are:

| KSwS Index | Classification |
|---:|---|
| 0.00-0.20 | Normal |
| 0.20-0.40 | Mild |
| 0.40-0.60 | Moderate |
| 0.60-0.80 | Severe |
| 0.80-1.00 | Critical |

The system also evaluates gait patterns including:

- Crouch gait
- Stiff-knee gait
- Asymmetric gait
- Jump gait
- Apparent equinus

These automated flags are research and decision-support indicators rather than independent clinical diagnoses.

## Machine Learning

PediGait includes supervised machine-learning experiments for gait-pattern and KSwS classification.

### Temporal CNN-BiLSTM

The main temporal model combines one-dimensional temporal convolution with a bidirectional LSTM.

```text
Input gait sequence
       |
       v
Temporal Conv1D
       |
Batch Normalization
       |
GELU Activation
       |
Dropout
       |
Temporal Conv1D
       |
Batch Normalization
       |
GELU Activation
       |
Dropout
       |
Bidirectional LSTM
       |
Fully Connected Layer
       |
Classification Output
```

The implementation uses temporal convolution layers followed by a two-layer bidirectional LSTM and fully connected classification layers.

### Research Evaluation

The project notebook contains a patient-separated 10-fold evaluation of the temporal model.

Reported dataset:

| Parameter | Value |
|---|---:|
| Gait cycles | 1,719 |
| Patients | 356 |
| Cross-validation folds | 10 |
| Classification task | 6-class KSwS |

Reported overall Temporal CNN results:

| Metric | Score |
|---|---:|
| Accuracy | 59.86% |
| Balanced Accuracy | 62.74% |
| Macro F1 | 58.89% |

A Logistic Regression baseline reported in the research notebook achieved:

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 55.38% | 54.86% | 53.02% |
| Temporal CNN | 59.86% | 62.74% | 58.89% |

These results represent research-stage model evaluation and should not be interpreted as clinical validation.

## TD vs CP Comparative Analysis

PediGait includes a dedicated comparative analysis pipeline for typically developing (TD) and cerebral palsy (CP) gait.

The analysis can compare:

- Joint trajectories
- Sagittal range of motion
- Frontal range of motion
- Transverse range of motion
- Mean joint angles
- Variability
- Bilateral symmetry
- Trajectory similarity

This provides quantitative comparison of gait patterns in addition to visual inspection.

## Interactive Streamlit Workstation

The main application is implemented in `app.py` using Streamlit.

The workstation provides access to:

1. 3D digital-twin visualization
2. Kinematic trajectories and gait-cycle analysis
3. KSwS severity and gait-pattern analysis
4. Comparative cohort analytics
5. Biomechanical processing and verification

The application also supports uploading a custom C3D trial for analysis.

## Installation

### Clone the repository

```bash
git clone https://github.com/shanmithat/pedigait.git
cd pedigait
```

### Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install PediGait in development mode

```bash
pip install -e .
```

## Running the Application

Launch the Streamlit workstation:

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

## Command-Line Interface

The package also provides command-line workflows.

Run the complete analysis pipeline:

```bash
python -m pedigait --input data/c3d_files --output outputs --all
```

Process a single file:

```bash
python -m pedigait --input data/c3d_files --sample-file TD1a.c3d --output outputs --visualize
```

Run comparative analysis:

```bash
python -m pedigait --input data/c3d_files --output outputs --comparative --visualize
```

Run classification:

```bash
python -m pedigait --input data/c3d_files --output outputs --classify
```

## Python API

Core components can also be used directly from Python:

```python
from pedigait.data.c3d_loader import C3DLoader
from pedigait.data.kinematic_processor import KinematicProcessor
from pedigait.visualization.digital_twin_3d import DigitalTwinVisualizer
from pedigait.analysis.symmetry_analysis import SymmetryAnalyzer

loader = C3DLoader("data/TD1a.c3d")
loader.load()

processor = KinematicProcessor(loader.point_rate)

kinematic_data = processor.process_marker_data(
    loader.get_marker_positions(),
    loader.get_marker_labels()
)

symmetry_analyzer = SymmetryAnalyzer()
ksws_metrics = symmetry_analyzer.analyze_knee_symmetry(kinematic_data)

print(f"KSwS Index: {ksws_metrics.ksws_index:.4f}")
print(f"Classification: {ksws_metrics.severity.classification}")

visualizer = DigitalTwinVisualizer(
    loader.get_marker_positions(),
    loader.get_marker_labels(),
    loader.point_rate
)

fig = visualizer.create_plotly_animation("TD1a")
fig.show()

visualizer.save_animation(
    fig,
    "outputs/TD1a_animation.html"
)
```

## Data Requirements

PediGait accepts standard biomechanical C3D motion-capture files.

The current skeletal reconstruction uses markers including:

```text
Pelvis:
LASI, RASI, LPSI, RPSI

Lower Limb:
LTHI, RTHI
LKNE, RKNE
LTIB, RTIB
LANK, RANK
LHEE, RHEE
LTOE, RTOE
```

Additional markers may be used for upper-body and full-skeleton visualization.

A typical data directory can be organized as:

```text
data/
├── c3d_files/
│   ├── TD1a.c3d
│   ├── TD1b.c3d
│   ├── DiCP1.c3d
│   ├── HeCP1.c3d
│   └── ...
└── workbook.xlsx
```

Clinical or research datasets are not included in the public repository unless explicitly provided.

## Project Structure

```text
pedigait/
├── pedigait/
│   ├── __init__.py
│   ├── __main__.py
│   ├── data/
│   │   ├── c3d_loader.py
│   │   └── kinematic_processor.py
│   ├── visualization/
│   │   └── digital_twin_3d.py
│   ├── analysis/
│   │   ├── symmetry_analysis.py
│   │   └── comparative_analysis.py
│   ├── classification/
│   │   └── temporal_cnn.py
│   └── utils/
│
├── assets/
├── docs/
├── app.py
├── generate_report_data_and_figures.py
├── pedigait.ipynb
├── index.html
├── mocap_data.js
├── mocap_data.json
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Output Files

The analysis pipeline can generate:

```text
analysis_results.json
summary_report.txt
{trial}_animation.html
{trial}_static.png
classification_results.json
comparative_analysis.json
```

These outputs contain processed biomechanical metrics, visualization files, classification results, and comparative analysis results.

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Application | Streamlit |
| Motion Capture | C3D / ezc3d |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Signal Processing | SciPy |
| Visualization | Plotly, Matplotlib |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| Web Visualization | HTML / JavaScript |
| Package Management | setuptools |

## Research Status and Limitations

PediGait is currently a research and engineering prototype.

The project demonstrates:

- Automated C3D processing
- Biomechanical kinematic extraction
- Interactive 3D digital-twin visualization
- Bilateral symmetry analysis
- KSwS-based severity analysis
- TD versus CP comparison
- Supervised gait classification
- Patient-separated model evaluation

The current model results should not be interpreted as clinical validation.

Important areas for future validation include:

- Larger and more diverse datasets
- External validation cohorts
- Independent clinical validation
- Improved class-balance handling
- Robustness across motion-capture laboratories
- Prospective evaluation
- Explainability of machine-learning predictions
- Comparison against expert clinical assessment
- Integration of additional kinetic and anthropometric information

## Future Development

Potential extensions include:

- Physics-informed biomechanical modeling
- Inverse dynamics
- Ground-reaction-force estimation
- Patient-specific digital twins
- Anthropometric personalization
- Real-time motion capture
- Markerless pose estimation
- Automated gait-event detection
- Explainable gait classification
- Longitudinal patient tracking
- Multi-modal sensor fusion
- Automated clinical report generation

## Contributing

Contributions are welcome.

Create a feature branch:

```bash
git checkout -b feature/your-feature
```

Commit your changes:

```bash
git add .
git commit -m "Describe your changes"
```

Push the branch:

```bash
git push origin feature/your-feature
```

Then open a Pull Request.

For bugs, feature requests, or questions, use the GitHub issue tracker.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use PediGait in academic or research work, please cite:

```bibtex
@misc{pedigait,
  author       = {Shanmitha T},
  title        = {PediGait: Pediatric Gait Digital Twin and Biomechanical Analysis Platform},
  year         = {2026},
  howpublished = {GitHub},
  url          = {https://github.com/shanmithat/pedigait}
}
```

## Project

PediGait is a computational biomechanics and machine-learning research project focused on pediatric gait analysis, digital-twin visualization, and quantitative gait assessment.

Repository:
https://github.com/shanmithat/pedigait

Live application:
https://shanmithat.github.io/pedigait/
