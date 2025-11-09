Trust-Gated Hybrid VANET IDS
Lightweight OBU Screening with RSU-Supervised Evidence Fusion (v7)

Authors

Mohammad Abbas Alkifaee¹ (ORCID: 0009-0004-3731-0886)

Supervisor: Dr. Fahad Ghalib Abdulkadhim² (ORCID: 0000-0002-4922-0878)

Department of Computer Science and Mathematics, University of Kufa, Najaf, Iraq

Abstract
This repository hosts the artifacts for the study “Trust-Gated Hybrid VANET IDS: Lightweight OBU Screening with RSU-Supervised Evidence Fusion.”
We implement an RSU-centric intrusion-detection and trust-gating workflow that couples: (i) fast physics- and protocol-plausibility flags that accumulate trust evidence, (ii) short-horizon temporal descriptors per sender, (iii) a calibrated gradient-boosting screener for the binary decision, (iv) optional family-specific heads—including a BiLSTM module targeting replay/staleness—and (v) a logistic stacking layer for decision fusion.
Per-sender operating points are adapted via a Beta-trust posterior, allowing sensitivity to track the amount and quality of evidence. A PyQt5 application is included to select datasets, train models, and visualize evaluation outputs such as AUC, precision–recall profiles, and confusion matrices.

Data Availability
Dataset DOI/URL: [10.5281/zenodo.17167970]

Update this placeholder with the canonical link to your dataset repository (e.g., Zenodo, OSF, Kaggle).
Expected CSV schema


Required: sender_pseudo, t_curr, label (0 = benign, 1 = attack)


Recommended: x_curr, y_curr, speed_curr, acc_curr, heading_curr, dt, dist, mb_version, scenario_id, attack_id
Missing fields are imputed safely; richer fields enable stronger temporal and kinematic features.



Software Artifact
Single-file implementation: rsu_trainer_gui_v7.py
Core modules


Plausibility & trust evidence: physics/protocol checks and rolling anomaly counters.


Temporal features: short rolling windows per sender (rates, CUSUM, dead-reckoning residuals, freeze ratios, circular variance).


Base screener: LightGBM with isotonic calibration.


Family-specific heads (optional): pos_speed, replay_stale (BiLSTM), dos (+IsolationForest anomaly channel), sybil, disruptive.


Decision fusion: logistic meta-learner stacking base and heads.


Trust gating: Beta-trust process to adjust the per-sender decision threshold.


GUI: PyQt5 interface for dataset selection, training, and metric visualization.



Installation
Python 3.9+ is recommended.
pip install numpy pandas scikit-learn lightgbm "tensorflow~=2.15.0" pyqt5 matplotlib

Run
python rsu_trainer_gui_v7.py

Notes


GPU support for TensorFlow is optional and not required unless training the BiLSTM head at scale.


If your dataset columns differ, use the GUI mapping dialog or adapt the loader section inside rsu_trainer_gui_v7.py.



Citation
If you build on this work, please cite the study and this repository. A BibTeX entry will be added once the public DOI is finalized.
