# Trust-Gated Hybrid Intrusion Detection for VANETs
**Lightweight OBU Screening with RSU Supervised Fusion (v7)**

**Authors**  
Mohammad Abbas Alkifaee¹ (ORCID: [0009-0004-3731-0886](https://orcid.org/0009-0004-3731-0886))  
Supervisor: Dr. Fahad Ghalib Abdulkadhim² (ORCID: [0000-0002-4922-0878](https://orcid.org/0000-0002-4922-0878))

¹ Affiliation: *[Department of Computer Science and Mathematics, University of Kufa, Najaf, Iraq]*  
² Affiliation: *[Department of Computer Science and Mathematics, University of Kufa, Najaf, Iraq]*  

---

## Abstract
This repository accompanies the study **“Trust-Gated Hybrid Intrusion Detection for VANETs: Lightweight OBU Screening with RSU Supervised Fusion.”** We present an end-to-end RSU-side intrusion detection and trust gating pipeline combining physics/protocol consistency checks, short-window temporal feature engineering, calibrated gradient boosting for binary screening, optional family-specific heads (including a BiLSTM replay/stale detector), and a logistic meta-learner. An adaptive, per-sender decision threshold is derived from a Beta-trust model to modulate sensitivity under varying evidence. A PyQt5 GUI is included for dataset selection, training, and evaluation (AUC, PR curves, confusion matrices).

---

## Data Availability
**Dataset DOI/URL:** **[INSERT YOUR DATASET LINK OR DOI HERE]**  
*(Replace this placeholder with your actual dataset link, e.g., Zenodo/Kaggle/OSF DOI.)*

**Expected CSV schema**
- Required: `sender_pseudo`, `t_curr`, `label` (0=normal, 1=attack)
- Optional but recommended: `x_curr`, `y_curr`, `speed_curr`, `acc_curr`, `heading_curr`, `dt`, `dist`, `mb_version`, `scenario_id`, `attack_id`  
Missing fields are safely imputed; additional fields enable stronger features.

---

## Software Artifact
Single file implementation: `rsu_trainer_gui_v7.py`  
Key components:
- **Quick checks & trust evidence:** physics/protocol flags and anomaly counters.
- **Temporal features:** short rolling windows per sender (rates, CUSUM, dead-reckoning residuals, freeze ratios, circular variance).
- **Base detector:** LightGBM with isotonic calibration.
- **Family heads (optional):** `pos_speed`, `replay_stale` (BiLSTM), `dos` (+IsolationForest anomaly channel), `sybil`, `disruptive`.
- **Stacking:** logistic meta-learner over base and heads.
- **Trust gating:** Beta-trust adaptive threshold per sender.
- **GUI:** PyQt5 interface for training and metrics visualization.

---

## Installation
Python 3.9+ is recommended.
```bash
pip install numpy pandas scikit-learn lightgbm "tensorflow~=2.15.0" pyqt5 matplotlib
