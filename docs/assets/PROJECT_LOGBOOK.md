# PediGait Project Logbook (Second-Half Progress Workflow)
**Project Title:** Automated Pediatric Gait Digital Twin and Biomechanical Analysis System for Cerebral Palsy Locomotion Classification and Knee Swing Severity (KSwS) Grading  
**Department:** School of Mechanical Engineering, Vellore Institute of Technology, Chennai  
**Student Name / Reg No:** Shanmitha T  
**Duration:** 24th August 2026 to 21st October 2026 (Monday to Friday, 43 Working Days)  
**Context:** Continuation Phase (commencing post-model training and initial digital twin render: comprehensive validation, 3D kinematics, KSwS diagnostic rule engine, registry benchmarking, interactive clinical workstation, publication figures, and viva defense).  
**Format:** Single Explanatory Sentence per Day, Implied 3rd Person Point of View.  

---

## Weekly Milestone Overview

| Week | Date Range | Core Focus / Major Milestone | Working Days | Status |
|---|---|---|:---:|:---:|
| **Week 1** | 24 Aug – 28 Aug 2026 | Post-Training Model Evaluation & 41-Link Skeletal Rig Refinement | 5 Days | Completed |
| **Week 2** | 31 Aug – 04 Sep 2026 | 3D Joint Angle Kinematic Extraction & Normative Corridors | 5 Days | Completed |
| **Week 3** | 07 Sep – 11 Sep 2026 | Bilateral Symmetry Analysis & Continuous KSwS Index | 5 Days | Completed |
| **Week 4** | 14 Sep – 18 Sep 2026 | Automated Clinical Diagnostic Rules Engine | 5 Days | Completed |
| **Week 5** | 21 Sep – 25 Sep 2026 | Clinical Registry Ingestion (1,719 Strides) & Tabular ML Benchmarking | 5 Days | Completed |
| **Week 6** | 28 Sep – 02 Oct 2026 | Multi-Class Error Analysis & Binary Pathology Modeling | 5 Days | Completed |
| **Week 7** | 05 Oct – 09 Oct 2026 | Interactive WebGL 3D Animation & Comparative Digital Twin | 5 Days | Completed |
| **Week 8** | 12 Oct – 16 Oct 2026 | Streamlit Workstation Deployment & Performance Optimization | 5 Days | Completed |
| **Week 9** | 19 Oct – 21 Oct 2026 | Experimental Reconciliation, Final Report & Defense Sign-Off | 3 Days | Completed |

---

## Detailed Day-by-Day Logbook Table

| S.No | Week | Date & Day | Activity (One Explanatory Sentence, Implied 3rd Person) | Remarks / Status |
|---|---|---|---|:---:|
| 1 | **Week 1** | 24/08/2026 (Monday) | Reviewed evaluation metrics of the pre-trained Temporal CNN model and formulated the second-half project roadmap with the project guide. | Verified |
| 2 | **** | 25/08/2026 (Tuesday) | Audited out-of-fold classification predictions from the deep learning checkpoint to analyze error distributions across validation folds. | Verified |
| 3 | **** | 26/08/2026 (Wednesday) | Upgraded the initial 3D digital twin skeletal rig by expanding 34 discrete optical markers into 41 interconnected anatomical link segments. | Verified |
| 4 | **** | 27/08/2026 (Thursday) | Verified anatomical segment link definitions across head, spine, clavicle, pelvis, thighs, shanks, and feet against clinical Vicon Plug-in Gait standards. | Verified |
| 5 | **** | 28/08/2026 (Friday) | Formulated anatomical local coordinate systems (ACS) for lower limb segments following International Society of Biomechanics (ISB) conventions. | Verified |
| 6 | **Week 2** | 31/08/2026 (Monday) | Implemented Davis regression equations and epicondylar midpoint algorithms to compute 3D hip, knee, and ankle joint center trajectories. | Verified |
| 7 | **** | 01/09/2026 (Tuesday) | Constructed 3D Cardan/Euler rotation matrix solvers to compute relative joint angles across sagittal, coronal, and transverse planes. | Verified |
| 8 | **** | 02/09/2026 (Wednesday) | Programmed automated heel-strike and toe-off detection routines to segment continuous joint angle curves into normalized 0-100% gait cycles. | Verified |
| 9 | **** | 03/09/2026 (Thursday) | Extracted full-stride sagittal kinematic trajectories for Pelvis, Hip, Knee, and Ankle across all 21 optical C3D clinical trials. | Verified |
| 10 | **** | 04/09/2026 (Friday) | Validated computed joint angle waveforms of healthy control trials against established normative pediatric kinematic corridors (Mean +/- 1 SD). | Verified |
| 11 | **Week 3** | 07/09/2026 (Monday) | Formulated mathematical algorithms for bilateral gait asymmetry, including the Robinson Symmetry Index (SI) and cross-correlation lag analysis. | Verified |
| 12 | **** | 08/09/2026 (Tuesday) | Developed the SymmetryAnalyzer module to quantify left-versus-right angular divergence and trajectory root-mean-square error (RMSE). | Verified |
| 13 | **** | 09/09/2026 (Wednesday) | Formulated the continuous Knee Swing Severity (KSwS) mathematical index (0.0 to 1.0) integrating peak swing flexion deficits, timing delays, and stance contractures. | Verified |
| 14 | **** | 10/09/2026 (Thursday) | Computed continuous KSwS indices and symmetry percentages across Typically Developing, Diplegic CP, and Hemiplegic CP cohorts. | Verified |
| 15 | **** | 11/09/2026 (Friday) | Established normative clinical cutoffs for the continuous KSwS index (< 0.20 for healthy gait) and reviewed validation results with the project guide. | Verified |
| 16 | **Week 4** | 14/09/2026 (Monday) | Formulated quantitative threshold criteria for spastic gait patterns based on sagittal kinematic parameters and symmetry scores. | Verified |
| 17 | **** | 15/09/2026 (Tuesday) | Implemented clinical decision rules to automatically detect Crouch Gait when stance-phase knee flexion exceeds 25 degrees. | Verified |
| 18 | **** | 16/09/2026 (Wednesday) | Programmed automated diagnostic screening for Stiff-Knee Gait when peak swing-phase knee flexion fails to reach 48 degrees. | Verified |
| 19 | **** | 17/09/2026 (Thursday) | Integrated automated detection for Equinus foot strike when initial contact ankle dorsiflexion drops below -4 degrees. | Verified |
| 20 | **** | 18/09/2026 (Friday) | Verified the complete rule-based clinical diagnostic engine across all 21 C3D trials, confirming 100% consistency with recorded clinical diagnoses. | Verified |
| 21 | **Week 5** | 21/09/2026 (Monday) | Ingested the large-scale clinical gait registry workbook (workbook.xlsx) comprising 1,719 normalized stride cycles across 356 pediatric patients. | Verified |
| 22 | **** | 22/09/2026 (Tuesday) | Audited the registry feature taxonomy, cataloging 185 documented discrete parameters and 643 continuous waveform time points across 6 KSwS classes. | Verified |
| 23 | **** | 23/09/2026 (Wednesday) | Configured a stratified 80/20 train/test partition (N_train=1375, N_test=344) to evaluate machine learning models on the raw 828-feature tabular dataset. | Verified |
| 24 | **** | 24/09/2026 (Thursday) | Trained baseline tabular classifiers (Random Forest, Linear SVM, and Multi-Layer Perceptron) for fine-grained 6-class KSwS severity grading. | Verified |
| 25 | **** | 25/09/2026 (Friday) | Evaluated tabular baseline models against the random chance threshold of 16.67%, recording 32.85% multi-class accuracy for Random Forest and SVM. | Verified |
| 26 | **Week 6** | 28/09/2026 (Monday) | Analyzed the 6-class normalized confusion matrix, identifying high misclassification rates between clinically adjacent severity grades. | Verified |
| 27 | **** | 29/09/2026 (Tuesday) | Performed top-k classification analysis on the multi-class model, demonstrating that top-2 and top-3 accuracy reach 56.98% and 76.74% due to continuous severity overlap. | Verified |
| 28 | **** | 30/09/2026 (Wednesday) | Formulated binary pathology classification (Typically Developing Class 0 vs Cerebral Palsy Classes 1-5) across the 1,719 clinical stride cycles. | Verified |
| 29 | **** | 01/10/2026 (Thursday) | Trained and validated binary classification models, achieving 94.14% sensitivity in capturing pathological cerebral palsy strides (209 true positives out of 222). | Verified |
| 30 | **** | 02/10/2026 (Friday) | Compiled the comparative machine learning performance report and reviewed model trade-offs with the project guide. | Verified |
| 31 | **Week 7** | 05/10/2026 (Monday) | Integrated a high-performance Plotly WebGL graphics pipeline to elevate the initial static digital twin into an interactive 3D motion animator. | Verified |
| 32 | **** | 06/10/2026 (Tuesday) | Programmed multi-camera perspective controls allowing clinicians to dynamically view the 3D twin in Sagittal, Coronal, Transverse, and Isometric angles. | Verified |
| 33 | **** | 07/10/2026 (Wednesday) | Engineered dynamic time-slider scrubbing and variable playback speed controls (0.25x to 2.0x) for frame-by-frame kinematic inspection. | Verified |
| 34 | **** | 08/10/2026 (Thursday) | Developed the synchronized side-by-side comparative digital twin rendering normative TD and CP diplegic motion capture in a single viewport. | Verified |
| 35 | **** | 09/10/2026 (Friday) | Implemented real-time color-coded segment tracking highlighting pathological joint deviations dynamically during comparative digital twin playback. | Verified |
| 36 | **Week 8** | 12/10/2026 (Monday) | Developed the unified Streamlit clinical workstation (app.py) integrating Mocap Ingestion, 3D Digital Twin, Kinematics, and Pathology Classification tabs. | Verified |
| 37 | **** | 13/10/2026 (Tuesday) | Diagnosed and resolved a critical PickleError caching exception by transitioning cache decorators from st.cache_data to st.cache_resource for C3D binary objects. | Verified |
| 38 | **** | 14/10/2026 (Wednesday) | Overhauled the workstation UI/UX with professional clinical dark styling, high-contrast metric badges, and sanitized git repository commits. | Verified |
| 39 | **** | 15/10/2026 (Thursday) | Developed the automated evaluation script (generate_report_data_and_figures.py) to extract comprehensive performance tables and serialize metrics into JSON format. | Verified |
| 40 | **** | 16/10/2026 (Friday) | Rendered high-resolution 300 DPI publication figures (flowcharts, skeletal rigs, kinematic corridors, ROM radars, and confusion matrices) into the report directory. | Verified |
| 41 | **Week 9** | 19/10/2026 (Monday) | Audited experimental metrics and resolved legacy documentation discrepancies across binary screening (94.7%), patient-clustered waveforms (60.15%), and tabular baselines (32.85%). | Verified |
| 42 | **** | 20/10/2026 (Tuesday) | Compiled comprehensive project report chapters and formatted quantitative biomechanical tables in accordance with university capstone guidelines. | Verified |
| 43 | **** | 21/10/2026 (Wednesday) | Finalized viva defense presentation slides, rehearsed live digital twin workstation demonstrations, and obtained formal guide review sign-off. | Verified |
