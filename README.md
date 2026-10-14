# Pediatric Gait Digital Twin Analysis

A comprehensive Python package for analyzing pediatric gait patterns using motion capture data (C3D files). This package provides tools for kinematic analysis, 3D visualization, locomotion classification, and comparative analysis between typically developing (TD) and cerebral palsy (CP) gait.

## Features

### 1. **Kinematic Analysis**
- Extract joint motion trajectories (Pelvis, Hip, Knee, Ankle, Foot)
- Calculate joint angles, angular velocities, and ranges of motion
- Process gait cycle data from C3D motion capture files

### 2. **3D Digital Twin Visualizer**
- Create interactive 3D skeletal visualizations
- Wireframe connections between key markers (ASIS, PSIS, Thigh, Knee, Tibia, Ankle, Heel, Toe)
- Side-by-side comparisons of TD vs CP gait patterns
- Export animations to HTML for interactive viewing

### 3. **Locomotion Classification**
- Temporal CNN for classifying gait patterns
- Knee Swing Symmetry/Severity (KSwS) index calculation
- Classification into: Normal, Mild, Moderate, Severe, Highly Pathological
- Tabular baseline classifiers for comparison

### 4. **Comparative Analysis**
- Quantitative comparison between TD and CP gait
- Range of motion deviations
- Symmetry analysis
- Trajectory similarity metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/shanforpresident/pedigait.git
cd pedigait

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Using the Command-Line Interface

```bash
# Run complete analysis pipeline
python -m pedigait --input data/c3d_files --output outputs --all

# Process a single file
python -m pedigait --input data/ --sample-file TD1a.c3d --output outputs --visualize

# Run comparative analysis
python -m pedigait --input data/ --output outputs --comparative --visualize

# Run classification
python -m pedigait --input data/ --output outputs --classify
```

### Using the Python API

```python
from pathlib import Path
from pedigait.data.c3d_loader import C3DLoader
from pedigait.data.kinematic_processor import KinematicProcessor
from pedigait.visualization.digital_twin_3d import DigitalTwinVisualizer
from pedigait.analysis.symmetry_analysis import SymmetryAnalyzer

# Load C3D file
loader = C3DLoader("data/TD1a.c3d")
loader.load()

# Process kinematic data
processor = KinematicProcessor(loader.point_rate)
kinematic_data = processor.process_marker_data(
    loader.get_marker_positions(),
    loader.get_marker_labels()
)

# Analyze symmetry
symmetry_analyzer = SymmetryAnalyzer()
ksws_metrics = symmetry_analyzer.analyze_knee_symmetry(kinematic_data)

print(f"KSwS Index: {ksws_metrics.ksws_index:.4f}")
print(f"Classification: {ksws_metrics.severity.classification}")

# Create 3D visualization
visualizer = DigitalTwinVisualizer(
    loader.get_marker_positions(),
    loader.get_marker_labels(),
    loader.point_rate
)

# Create interactive animation
fig = visualizer.create_plotly_animation("TD1a")
fig.show()

# Save animation
visualizer.save_animation(fig, "outputs/TD1a_animation.html")
```

## Project Structure

```
pedigait/
├── pedigait/
│   ├── __init__.py           # Package initialization
│   ├── __main__.py           # Command-line interface
│   ├── data/
│   │   ├── c3d_loader.py      # C3D file loading
│   │   └── kinematic_processor.py  # Kinematic data processing
│   ├── visualization/
│   │   └── digital_twin_3d.py # 3D visualization
│   ├── analysis/
│   │   ├── symmetry_analysis.py    # Symmetry and KSwS analysis
│   │   └── comparative_analysis.py # TD vs CP comparison
│   ├── classification/
│   │   └── temporal_cnn.py    # CNN classification
│   └── utils/
│       └── helpers.py         # Utility functions
├── requirements.txt          # Dependencies
├── README.md                # This file
├── data/                    # Sample data (not included)
└── outputs/                 # Output directory
```

## Data Requirements

### Input Data
- **C3D Files**: Standard biomechanical motion capture files
- **Marker Set**: Should include at least:
  - Pelvis: LASI, RASI, LPSI, RPSI
  - Hip: LTHI, RTHI, LHFE, RHFE
  - Knee: LKNE, RKNE, LFEME, RFEME
  - Ankle: LANK, RANK, LHEE, RHEE
  - Foot: LTOE, RTOE

### Sample Data Structure
```
data/
├── c3d_files/
│   ├── TD1a.c3d
│   ├── TD1b.c3d
│   ├── DiCP1.c3d
│   ├── HeCP1.c3d
│   └── ...
└── workbook.xlsx        # Optional: Clinical kinematic parameters
```

## Key Metrics

### Knee Swing Symmetry/Severity (KSwS) Index
- **Range**: 0 (normal) to 1 (highly pathological)
- **Classification**:
  - 0.0-0.1: Normal
  - 0.1-0.3: Mild
  - 0.3-0.6: Moderate
  - 0.6-0.8: Severe
  - 0.8-1.0: Highly Pathological

### Symmetry Index
- **Range**: 0 (perfect asymmetry) to 1 (perfect symmetry)
- **Calculation**: 1 - (normalized RMSE between left and right trajectories)

### Range of Motion (ROM)
- Sagittal plane (flexion/extension)
- Frontal plane (abduction/adduction)
- Transverse plane (rotation)

## Visualization

### 3D Digital Twin
- Interactive Plotly animations
- Skeleton wireframe connections
- Marker positions with labels
- Color-coded by height (Z-coordinate)

### Comparative Visualizations
- Side-by-side TD vs CP comparisons
- Range of motion bar charts
- Trajectory overlay plots
- Statistical comparison plots

## Classification Models

### Temporal CNN Architecture
```
Input: [batch, 100, 18]  # 100 timesteps, 18 features
├── Conv1d: 64 filters, kernel_size=3
├── BatchNorm1d
├── GELU activation
├── Dropout: 0.3
├── Conv1d: 64 filters, kernel_size=3
├── BatchNorm1d
├── GELU activation
├── Dropout: 0.3
├── BiLSTM: 128 hidden units, 2 layers
├── Linear: 64 units
├── GELU activation
├── Dropout: 0.3
└── Linear: n_classes (output)
```

### Tabular Baseline
- Random Forest classifier
- Uses extracted features from kinematic data
- For comparison with CNN performance

## Pathology Patterns

### Detected Gait Patterns
1. **Crouch Gait**: Excessive knee and hip flexion
2. **Stiff Knee Gait**: Reduced knee flexion
3. **Asymmetric Gait**: Significant left-right asymmetry
4. **Jump Gait**: Excessive knee extension
5. **Apparent Equinus**: Reduced ankle dorsiflexion

## Output Files

The analysis pipeline generates:
- `analysis_results.json`: Complete analysis results
- `summary_report.txt`: Human-readable summary
- `{trial}_animation.html`: Interactive 3D animations
- `{trial}_static.png`: Static 3D visualizations
- `classification_results.json`: Classification outputs
- `comparative_analysis.json`: TD vs CP comparison

## Dependencies

### Core Dependencies
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn

### C3D Processing
- ezc3d

### 3D Visualization
- plotly
- mayavi (optional, for advanced 3D)
- vtk (optional)

### Machine Learning
- pytorch
- torchvision

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
@misc{pedigait,
  author = {Pediatric Gait Analysis Team},
  title = {Pediatric Gait Digital Twin Analysis},
  year = {2024},
  howpublished = {\url{https://github.com/shanforpresident/pedigait}},
}
```

## Contact

For questions or support, please open an issue on the GitHub repository.

## Acknowledgments

This project was developed for analyzing pediatric gait patterns in clinical and research settings. Special thanks to all contributors and testers.
