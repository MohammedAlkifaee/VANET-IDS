Mohammad Abbas Alkifaee¹ (ORCID: 0009-0004-3731-0886)  
Supervisor: Dr. Fahad Ghalib Abdulkadhim² (ORCID: 0000-0002-4922-0878)  
Department of Computer Science and Mathematics, University of Kufa, Najaf, Iraq  

<p align="center">
  <img src="https://i.imgur.com/tv2A0aw.png" alt="VANET IDS v2 — Menu Screenshot" width="900">
</p>

# Trust-Gated Hybrid IDS for VANETs (Practical Implementation) — University of Kufa

This repository contains the practical (implementation) side of a Master’s thesis project: a **Trust-Gated Hybrid Intrusion Detection System (IDS)** for Vehicular Ad-Hoc Networks (VANETs).

Core idea:
- **OBU lightweight screening** (fast plausibility/consistency checks)
- **RSU model inference** (ML-based detection)
- **Trust-gated fusion** (adaptive decision based on trust)

> Research/academic use only.

---

## Repository files

- **`vanet_ids_v2.py`** — Main entry point (interactive menu + CLI modes)
- **`main.py`** — Live IDS Dashboard (PyQt5 GUI)
- **`rsu_trainer_all_in_one_v7.py`** — RSU Trainer (all-in-one GUI)
- **`add_attack_id.py`** — Helper to append `attack_id` after `label`

---

## Libraries used

Core:
- `numpy`, `pandas`
- `scikit-learn`, `joblib`

GUI & visualization:
- `PyQt5`
- `pyqtgraph`
- `matplotlib`

Training / ML (depending on enabled heads and configuration):
- `lightgbm`
- `tensorflow` (for sequence models if enabled)

Utilities:
- `argparse`, `pathlib`, `json`, `logging` (standard library)

---

## Installation

### Option A — Conda (recommended)
```bash
conda create -n vanet-ids python=3.10 -y
conda activate vanet-ids

pip install numpy pandas scikit-learn joblib pyqt5 pyqtgraph matplotlib lightgbm tensorflow
