import os, json, argparse, math, warnings, hashlib, time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os, shutil, subprocess, sys, datetime
import shlex


from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    auc,
    roc_auc_score,
)
from sklearn.utils.validation import check_is_fitted
import joblib


warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(suppress=True)

import os, sys, shutil, subprocess

CONDA_SH = "/home/instantf2md/miniconda/etc/profile.d/conda.sh"
CONDA_ENV = "base"


def _open_new_terminal(cmd: str, title: str = "VANET"):
    DEVNULL = subprocess.DEVNULL
    if shutil.which("gnome-terminal"):
        subprocess.Popen(
            ["gnome-terminal", "--title", title, "--", "bash", "-lc", cmd],
            start_new_session=True,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        return
    if shutil.which("x-terminal-emulator"):
        subprocess.Popen(
            ["x-terminal-emulator", "-e", "bash", "-lc", cmd],
            start_new_session=True,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        return
    subprocess.Popen(
        ["bash", "-lc", cmd], start_new_session=True, stdout=DEVNULL, stderr=DEVNULL
    )


def launch_rsu_trainer(rsu_path: str):
    logs_dir = "/home/instantf2md/Desktop/VANET_project/logs"
    os.makedirs(logs_dir, exist_ok=True)
    log = os.path.join(logs_dir, "rsu_trainer.log")
    open(log, "a").close()

    workdir = os.path.dirname(rsu_path)


    cmd = (
        f'cd "{workdir}" && '
        f'echo "[I] workdir={workdir}" >> "{log}" && '
        f'source "{CONDA_SH}" && conda activate "{CONDA_ENV}" && '
        f'echo "[I] python=$(which python)" >> "{log}" && python --version >> "{log}" && '
        f'python "{rsu_path}" >> "{log}" 2>&1; '
        f'echo "[I] EXIT=$?" >> "{log}"; '
        f'echo "[I] Log: {log}"; exec bash'
    )
    _open_new_terminal(cmd, title="RSU Trainer")


def launch_detached(cmd, log_path, cwd=None):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", buffering=1) as logf:
        logf.write(f"\n[I] started {datetime.datetime.now()} cmd={cmd}\n")
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    return p.pid


def log(msg: str):
    print(msg, flush=True)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = []
    for c in df.columns:
        c2 = str(c).replace("\ufeff", "").strip()
        if len(c2) >= 2 and c2[0] == c2[-1] and c2[0] in ("'", '"'):
            c2 = c2[1:-1].strip()
        cleaned.append(c2)
    df.columns = cleaned
    return df


def sanitize_entity_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("sender_pseudo", "receiver_pseudo", "mb_version"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.strip("'\"")
    return df


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def robust_percentile_abs(x: np.ndarray, q: float) -> float:
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    return float(np.percentile(np.abs(x), q))


def mad_threshold(x: np.ndarray, k: float = 4.0) -> float:
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12

    sigma = 1.4826 * mad
    return float(abs(med) + k * sigma)


def minmax01(a: np.ndarray) -> np.ndarray:
    mn, mx = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        return np.zeros_like(a, dtype=float)
    return (a - mn) / (mx - mn)


DERIVED_FEATURES = [

    "accel_prev",
    "accel_curr",
    "jerk",
    "heading_rate",
]

BASE_FEATURES = [

    "dt",
    "speed_prev",
    "speed_curr",
    "delta_speed",
    "dv",

    "accel_prev",
    "accel_curr",
    "dacc",
    "jerk",

    "heading_prev",
    "heading_curr",
    "heading_rate",
    "dtheta",

    "dist",
    "dx",
    "dy",

    "rate_msgs_per_s",

    "pos_conf_x_curr",
    "pos_conf_y_curr",
    "spd_conf_x_curr",
    "spd_conf_y_curr",
    "acc_conf_x_curr",
    "acc_conf_y_curr",
    "head_conf_x_curr",
    "head_conf_y_curr",

    "anom_score_iforest",
]


def _safe_col(df: pd.DataFrame, col: str, default: float = 0.0):
    if col not in df.columns:
        df[col] = default
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sort_key = "t_curr" if "t_curr" in df.columns else None
    group_keys = [c for c in ["receiver_pseudo", "sender_pseudo"] if c in df.columns]


    for c in ["dt", "speed_prev", "speed_curr", "heading_prev", "heading_curr"]:
        if c not in df.columns:
            df[c] = 0.0


    if "acc_curr" in df.columns and "accel_curr" not in df.columns:
        df["accel_curr"] = df["acc_curr"]
    if "acc_prev" in df.columns and "accel_prev" not in df.columns:
        df["accel_prev"] = df["acc_prev"]
    if "jerk" not in df.columns:
        df["jerk"] = np.nan
    if "heading_rate" not in df.columns:
        df["heading_rate"] = np.nan
    if "delta_speed" not in df.columns:
        df["delta_speed"] = df["speed_curr"] - df["speed_prev"]


    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            if g["accel_curr"].isna().any():
                g["accel_curr"] = (g["speed_curr"] - g["speed_prev"]) / g["dt"].replace(
                    0, np.nan
                )
            if g["accel_prev"].isna().any():
                g["accel_prev"] = g["accel_curr"].shift(1)
            if g["jerk"].isna().any():
                g["jerk"] = (g["accel_curr"] - g["accel_prev"]) / g["dt"].replace(
                    0, np.nan
                )
            if g["heading_rate"].isna().any():
                g["heading_rate"] = (g["heading_curr"] - g["heading_prev"]) / g[
                    "dt"
                ].replace(0, np.nan)
        return g

    if group_keys:
        if sort_key and sort_key in df.columns:
            df = df.sort_values(group_keys + [sort_key]).reset_index(drop=True)
        else:
            df = df.sort_values(group_keys).reset_index(drop=True)

        df = df.groupby(group_keys, group_keys=False).apply(
            _per_group, include_groups=True
        )
    else:
        df = _per_group(df)


    for c in ["accel_prev", "accel_curr", "jerk", "heading_rate", "delta_speed"]:
        df[c] = df[c].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return df


@dataclass
class TrainArtifacts:
    features: List[str]
    rsu_model_path: str
    rsu_info_path: str
    iforest_path: Optional[str]
    obu_thr_path: str
    trust_sender_path: str
    trust_pair_path: str
    config_path: str


class IDS_Trainer:
    def __init__(self, output_dir="release_v2"):
        self.output_dir = output_dir
        ensure_dir(self.output_dir)
        log(f"📁 Training outputs will be saved to: {self.output_dir}")

    def calculate_obu_thresholds(self, df_normal: pd.DataFrame) -> Dict[str, float]:
        log("\n--- 🔬 Step 1: Compute OBU thresholds (robust) ---")
        thr = {}
        df = df_normal.copy()

        dt_max_used = 2.0
        df = df[df["dt"] <= dt_max_used].copy()

        for col, q in [("jerk", 99.9), ("heading_rate", 99.9)]:
            p = robust_percentile_abs(df[col].values, q)
            m = mad_threshold(df[col].values, k=4.0)
            thr[f"{col}_abs_max"] = float(max(p, m))

        ds = (df["speed_curr"] - df["speed_prev"]).values
        p_ds = robust_percentile_abs(ds, 99.9)
        m_ds = mad_threshold(ds, k=4.0)
        thr["delta_speed_abs_max"] = float(max(p_ds, m_ds))

        thr["dt_max_used"] = dt_max_used

        path = os.path.join(self.output_dir, "obu_thresholds_v2.json")
        with open(path, "w") as f:
            json.dump(thr, f, indent=2)
        log(f"✅ OBU thresholds saved: {path}\n{json.dumps(thr, indent=2)}")
        return thr

    def _train_iforest(self, df_normal: pd.DataFrame) -> Tuple[IsolationForest, str]:
        log("\n--- 🌲 Step 2a: Train IsolationForest (optional unsupervised score) ---")
        use_cols = [
            "dt",
            "speed_prev",
            "speed_curr",
            "accel_curr",
            "jerk",
            "heading_rate",
            "dx",
            "dy",
            "dist",
            "rate_msgs_per_s",
        ]
        use_cols = [c for c in use_cols if c in df_normal.columns]
        Xn = df_normal[use_cols].fillna(0.0).values

        iforest = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination=0.02,
            random_state=42,
            n_jobs=-1,
        )
        iforest.fit(Xn)
        path = os.path.join(self.output_dir, "iforest_v2.joblib")
        joblib.dump({"model": iforest, "features": use_cols}, path)
        log(f"✅ IsolationForest saved: {path}")
        return iforest, path

    def _compute_iforest_score(
        self, iforest_bundle: Dict, df: pd.DataFrame
    ) -> np.ndarray:
        try:
            model = iforest_bundle["model"]
            feats = iforest_bundle["features"]
            X = df[feats].fillna(0.0).values

            score_normal = model.decision_function(X)
            risk = 1.0 - minmax01(score_normal)
            return risk
        except Exception:
            return np.zeros(len(df), dtype=float)

    def train_rsu_model(
        self, df_labeled: pd.DataFrame, iforest_bundle: Optional[Dict]
    ) -> Tuple[RandomForestClassifier, Dict, float]:
        log("\n--- 🧠 Step 2b: Train RSU RandomForest ---")

        df = df_labeled.copy()
        df = compute_derived_features(df)


        if iforest_bundle is not None:
            df["anom_score_iforest"] = self._compute_iforest_score(iforest_bundle, df)
        else:
            _safe_col(df, "anom_score_iforest", 0.0)


        features = [f for f in BASE_FEATURES if f in df.columns]
        log(f"Using {len(features)} features: {features}")

        X_all = df[features].fillna(0.0).replace([np.inf, -np.inf], 0.0).values
        y_all = df["label"].astype(int).values


        if "sender_pseudo" in df.columns:
            groups = df["sender_pseudo"].values
            log("Grouping by sender_pseudo (no leakage).")
        else:
            groups = np.arange(len(df))
            log("sender_pseudo missing; fallback grouping may leak.")


        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        (train_idx, test_idx) = next(skf.split(X_all, y_all, groups=groups))
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        log(f"Train size: {len(train_idx)} | Test size: {len(test_idx)}")

        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)


        proba_test = clf.predict_proba(X_test)[:, 1]

        prec, rec, thr = precision_recall_curve(y_test, proba_test)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_i = int(np.argmax(f1s))
        best_thr = float(thr[max(0, best_i - 1)]) if len(thr) > 0 else 0.5
        rsu_soft = float(
            max(0.1, min(best_thr * 0.8, best_thr))
        )

        y_pred = (proba_test >= best_thr).astype(int)
        report = classification_report(
            y_test, y_pred, target_names=["Normal (0)", "Attack (1)"]
        )
        try:
            roc = roc_auc_score(y_test, proba_test)
        except Exception:
            roc = float("nan")
        pr_auc = (
            auc(rec, prec) if np.all(np.isfinite([rec, prec])).all() else float("nan")
        )

        log("\n--- RSU Holdout Evaluation ---")
        log(report)
        log(
            f"ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f} | Tuned Threshold: {best_thr:.3f} | RSU soft gate: {rsu_soft:.3f}"
        )


        model_path = os.path.join(self.output_dir, "rsu_rf_v2.joblib")
        info_path = os.path.join(self.output_dir, "rsu_rf_v2_info.json")

        joblib.dump(clf, model_path)
        info = {
            "features": features,
            "metrics": {
                "roc_auc": roc,
                "pr_auc": pr_auc,
                "best_thr_f1": best_thr,
                "class_report": report,
            },
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        log(f"✅ RSU model saved: {model_path}")
        log(f"✅ RSU info  saved: {info_path}")
        return clf, info, best_thr

    def build_trust_database(self, df: pd.DataFrame) -> Tuple[str, str]:
        log("\n--- 📈 Step 3: Build initial trust tables ---")

        alpha, beta = 1.0, 1.0
        s = df.groupby("sender_pseudo", as_index=False).agg(
            msgs=("label", "size"), anomalies=("label", "sum")
        )
        s["non_anom"] = s["msgs"] - s["anomalies"]
        s["trust_raw"] = (alpha + s["non_anom"]) / (alpha + beta + s["msgs"])
        w = np.log1p(s["msgs"])
        s["trust_blend"] = s["trust_raw"] * (w / w.max())
        s = s.sort_values("trust_blend", ascending=True)


        p = df.groupby(["receiver_pseudo", "sender_pseudo"], as_index=False).agg(
            count=("label", "size"), anomalies=("label", "sum")
        )
        p["non_anom"] = p["count"] - p["anomalies"]
        p["trust_raw"] = (alpha + p["non_anom"]) / (alpha + beta + p["count"])
        w2 = np.log1p(p["count"])
        p["trust_blend"] = p["trust_raw"] * (w2 / w2.max())
        p = p.sort_values("trust_blend", ascending=True)

        sender_path = os.path.join(self.output_dir, "trust_sender_v2.csv")
        pair_path = os.path.join(self.output_dir, "trust_pair_v2.csv")
        s.to_csv(sender_path, index=False)
        p.to_csv(pair_path, index=False)
        log(f"✅ Trust tables saved: {sender_path} | {pair_path}")
        return sender_path, pair_path

    def write_final_config(
        self, rsu_thr: float, rsu_soft: float, iforest_path: Optional[str]
    ) -> str:
        config = {
            "fusion_mode": "OR_trust_gated_v2",
            "params": {
                "rsu_thr": float(round(rsu_thr, 4)),
                "rsu_soft": float(round(rsu_soft, 4)),
                "tsender_cut": 0.965,
                "tpair_cut": 0.85,
            },
            "models": {
                "obu_thresholds": "obu_thresholds_v2.json",
                "rsu_supervised": "rsu_rf_v2.joblib",
                "rsu_info": "rsu_rf_v2_info.json",
                "iforest": "iforest_v2.joblib" if iforest_path else None,
            },
            "artifacts": {
                "sender_trust": "trust_sender_v2.csv",
                "pair_trust": "trust_pair_v2.csv",
            },
        }
        path = os.path.join(self.output_dir, "fusion_config_final_v2.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        log(f"\n✅ Final fusion config saved: {path}")
        return path

    def write_manifest(self) -> None:
        manifest = {
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "numpy_version": np.__version__,
        }
        path = os.path.join(self.output_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"🧾 Manifest saved: {path}")

    def run_training_pipeline(self, labeled_csv_path: str) -> TrainArtifacts:
        log(f"\n============== TRAINING: {labeled_csv_path} ==============")

        df = pd.read_csv(labeled_csv_path, engine="python", quotechar="'")
        df = sanitize_columns(df)
        df = sanitize_entity_values(df)


        req = [
            "label",
            "sender_pseudo",
            "receiver_pseudo",
            "dt",
            "speed_curr",
            "speed_prev",
            "heading_prev",
            "heading_curr",
        ]
        for c in req:
            if c not in df.columns:
                raise ValueError(f"Missing required column: {c}")


        df = compute_derived_features(df)


        df_normal = df[df["label"] == 0].copy()
        self.calculate_obu_thresholds(df_normal)


        iforest_bundle = None
        if len(df_normal) > 500:
            iforest, iforest_path = self._train_iforest(df_normal)
            iforest_bundle = {
                "model": iforest,
                "features": joblib.load(iforest_path)["features"],
            }
        else:
            iforest_path = None
            log("ℹ️ Skipping IsolationForest (not enough normal samples).")


        clf, info, best_thr = self.train_rsu_model(df, iforest_bundle)
        rsu_soft = max(0.1, min(best_thr * 0.8, best_thr))


        sender_path, pair_path = self.build_trust_database(df)


        config_path = self.write_final_config(best_thr, rsu_soft, iforest_path)
        self.write_manifest()

        artifacts = TrainArtifacts(
            features=info["features"],
            rsu_model_path=os.path.join(self.output_dir, "rsu_rf_v2.joblib"),
            rsu_info_path=os.path.join(self.output_dir, "rsu_rf_v2_info.json"),
            iforest_path=(
                os.path.join(self.output_dir, "iforest_v2.joblib")
                if iforest_path
                else None
            ),
            obu_thr_path=os.path.join(self.output_dir, "obu_thresholds_v2.json"),
            trust_sender_path=sender_path,
            trust_pair_path=pair_path,
            config_path=config_path,
        )
        log("\n🎉 --- Training completed successfully! --- 🎉")
        return artifacts


class VanetIDS:
    def __init__(self, models_dir="release_v2"):
        self.models_dir = models_dir
        config_path = os.path.join(models_dir, "fusion_config_final_v2.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config not found: {config_path}. Run training first."
            )
        log("🚀 Initializing VANET IDS (detect mode)...")
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.params = self.config.get("params", {})
        self._load_dependencies()
        log("✅ Initialization complete.")

    def _load_dependencies(self):
        log("🔍 Loading models and artifacts...")

        with open(
            os.path.join(self.models_dir, self.config["models"]["obu_thresholds"]), "r"
        ) as f:
            self.obu_thr = json.load(f)


        self.rsu_model = joblib.load(
            os.path.join(self.models_dir, self.config["models"]["rsu_supervised"])
        )
        rsu_info_path = os.path.join(self.models_dir, self.config["models"]["rsu_info"])
        with open(rsu_info_path, "r") as f:
            self.rsu_info = json.load(f)


        if self.config["models"].get("iforest"):
            try:
                self.iforest_bundle = joblib.load(
                    os.path.join(self.models_dir, self.config["models"]["iforest"])
                )
            except Exception:
                self.iforest_bundle = None
        else:
            self.iforest_bundle = None


        self.sender_trust_db = pd.read_csv(
            os.path.join(self.models_dir, self.config["artifacts"]["sender_trust"])
        )
        self.pair_trust_db = pd.read_csv(
            os.path.join(self.models_dir, self.config["artifacts"]["pair_trust"])
        )
        log("👍 All components loaded.")

    def _ensure_features_for_rsu(self, df: pd.DataFrame) -> pd.DataFrame:
        df = compute_derived_features(df)

        if "anom_score_iforest" in self.rsu_info["features"]:
            if self.iforest_bundle is not None:
                try:
                    model = self.iforest_bundle["model"]
                    use_feats = self.iforest_bundle["features"]
                    X = df[use_feats].fillna(0.0).values
                    score_normal = model.decision_function(X)
                    df["anom_score_iforest"] = 1.0 - minmax01(score_normal)
                except Exception:
                    df["anom_score_iforest"] = 0.0
            else:
                df["anom_score_iforest"] = 0.0

        for c in self.rsu_info["features"]:
            _safe_col(df, c, 0.0)
        return df

    def run_obu_detector(self, df_input: pd.DataFrame) -> pd.DataFrame:
        log("\n--- 🛡️  Phase 1: OBU detector ---")
        df = df_input.copy()
        thr = self.obu_thr
        dt_ok = df["dt"] <= float(thr.get("dt_max_used", 2.0))
        delta_speed = (
            df["delta_speed"]
            if "delta_speed" in df.columns
            else (df["speed_curr"] - df["speed_prev"])
        )

        v_jerk = np.abs(df.get("jerk", 0.0)) > float(thr.get("jerk_abs_max", 1e9))
        v_hr = np.abs(df.get("heading_rate", 0.0)) > float(
            thr.get("heading_rate_abs_max", 1e9)
        )
        v_ds = np.abs(delta_speed) > float(thr.get("delta_speed_abs_max", 1e9))

        df["obu_inrange"] = dt_ok.astype(int)
        violations = (
            (dt_ok & v_jerk).astype(int)
            + (dt_ok & v_hr).astype(int)
            + (dt_ok & v_ds).astype(int)
        )
        df["obu_anom"] = (violations > 0).astype(int)
        df["obu_risk"] = (violations / 3.0).astype(float)

        keep = ["row_id", "receiver_pseudo", "sender_pseudo", "dt"]
        for c in keep:
            _safe_col(df, c, 0)
        out = df[keep + ["obu_inrange", "obu_anom", "obu_risk"]].copy()
        log(f"OBU anomalies: {int(out['obu_anom'].sum())}")
        return out

    def run_rsu_detector(self, df_input: pd.DataFrame) -> pd.DataFrame:
        log("\n--- 🧠  Phase 2: RSU detector ---")
        df = self._ensure_features_for_rsu(df_input.copy())
        X = (
            df[self.rsu_info["features"]]
            .fillna(0.0)
            .replace([np.inf, -np.inf], 0.0)
            .values
        )
        proba = self.rsu_model.predict_proba(X)[:, 1]
        decision_thr = float(self.params.get("rsu_thr", 0.5))
        pred = (proba >= decision_thr).astype(int)

        keep = ["row_id", "receiver_pseudo", "sender_pseudo", "dt"]
        for c in keep:
            _safe_col(df, c, 0)
        out = df[keep].copy()
        out["rsu_score"] = proba
        out["rsu_anom"] = pred
        log(f"RSU anomalies: {int(out['rsu_anom'].sum())}")
        return out

    def run_trust_gated_fusion(
        self, obu_df: pd.DataFrame, rsu_df: pd.DataFrame
    ) -> pd.DataFrame:
        log("\n--- 🧩  Phase 3: Trust-gated fusion ---")
        merged = pd.merge(
            obu_df,
            rsu_df,
            on=["row_id", "receiver_pseudo", "sender_pseudo", "dt"],
            how="inner",
        )

        S = self.sender_trust_db[["sender_pseudo", "trust_blend"]].rename(
            columns={"trust_blend": "trust_sender"}
        )
        P = self.pair_trust_db[
            ["receiver_pseudo", "sender_pseudo", "trust_blend"]
        ].rename(columns={"trust_blend": "trust_pair"})
        merged = merged.merge(S, on="sender_pseudo", how="left").merge(
            P, on=["receiver_pseudo", "sender_pseudo"], how="left"
        )

        merged["trust_sender"] = merged["trust_sender"].fillna(0.5)
        merged["trust_pair"] = merged["trust_pair"].fillna(0.5)

        rsu_soft = float(self.params.get("rsu_soft", 0.35))
        tsender_cut = float(self.params.get("tsender_cut", 0.965))
        tpair_cut = float(self.params.get("tpair_cut", 0.85))

        rsu_hit = merged["rsu_score"] >= rsu_soft
        high_trust = (merged["trust_sender"] >= tsender_cut) | (
            merged["trust_pair"] >= tpair_cut
        )
        obu_only = (merged["obu_anom"] == 1) & (~rsu_hit)
        suppress = obu_only & high_trust

        merged["fused_anom"] = (
            rsu_hit | ((merged["obu_anom"] == 1) & (~suppress))
        ).astype(int)
        merged["fused_risk"] = np.maximum(
            merged["rsu_score"].astype(float), merged["obu_risk"].astype(float)
        )
        out_cols = [
            "row_id",
            "receiver_pseudo",
            "sender_pseudo",
            "dt",
            "obu_anom",
            "obu_risk",
            "rsu_anom",
            "rsu_score",
            "trust_sender",
            "trust_pair",
            "fused_anom",
            "fused_risk",
        ]
        res = merged[out_cols].copy()
        log(f"Final fused anomalies: {int(res['fused_anom'].sum())}")
        return res

    def process_bsm_logfile(self, input_csv_path: str, output_csv_path: str):
        log(f"\n============== DETECT: {input_csv_path} ==============")

        df = pd.read_csv(input_csv_path, engine="python", quotechar="'")
        df = sanitize_columns(df)
        df = sanitize_entity_values(df)


        required = [
            "dt",
            "speed_prev",
            "speed_curr",
            "sender_pseudo",
            "receiver_pseudo",
            "heading_prev",
            "heading_curr",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s) in detection input: {missing}"
            )


        if df.empty:
            log("⚠️ Warning: empty file after dt filtering. No output produced.")
            return

        df = df.reset_index(drop=True)
        df["row_id"] = np.arange(len(df), dtype=int)

        obu = self.run_obu_detector(df)
        rsu = self.run_rsu_detector(df)
        fused = self.run_trust_gated_fusion(obu, rsu)
        fused.to_csv(output_csv_path, index=False)

        summary = {
            "total_messages": int(len(fused)),
            "final_anomalies": int(fused["fused_anom"].sum()),
        }
        log("\n--- 📊 Summary ---")
        log(json.dumps(summary, indent=2, ensure_ascii=False))
        log(f"✅ Saved results: {output_csv_path}")


def generate_synthetic_dataset(n_cars=60, n_msgs=6000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    senders = [f"car_{i:03d}" for i in range(n_cars)]
    receivers = [f"rsu_{i%6}" for i in range(n_cars)]
    sender = rng.choice(senders, size=n_msgs)
    receiver = rng.choice(receivers, size=n_msgs)

    dt = rng.uniform(0.05, 1.5, size=n_msgs)
    speed_prev = rng.normal(18.0, 4.0, size=n_msgs).clip(0, 45)
    accel = rng.normal(0.0, 0.6, size=n_msgs)
    speed_curr = (speed_prev + accel * dt).clip(0, 45)
    heading_prev = rng.uniform(-math.pi, math.pi, size=n_msgs)
    delta_heading = rng.normal(0.0, 0.06, size=n_msgs)
    heading_curr = heading_prev + delta_heading
    dist_roadelem = np.abs(rng.normal(0.8, 0.5, size=n_msgs))
    dx = rng.normal(0.0, 2.0, size=n_msgs)
    dy = rng.normal(0.0, 2.0, size=n_msgs)


    label = np.zeros(n_msgs, dtype=int)
    atk_idx = rng.choice(n_msgs, size=int(0.12 * n_msgs), replace=False)
    label[atk_idx] = 1


    jerk_boost = rng.normal(5.0, 2.0, size=atk_idx.size)
    speed_curr[atk_idx] += rng.normal(10.0, 5.0, size=atk_idx.size)
    delta_heading[atk_idx] += rng.normal(0.6, 0.2, size=atk_idx.size)

    t_curr = np.arange(n_msgs) * 0.1

    df = pd.DataFrame(
        {
            "receiver_pseudo": receiver,
            "sender_pseudo": sender,
            "t_curr": t_curr,
            "dt": dt,
            "speed_prev": speed_prev,
            "speed_curr": speed_curr,
            "heading_prev": heading_prev,
            "heading_curr": heading_curr,
            "dist_roadelem": dist_roadelem,
            "dx": dx,
            "dy": dy,
            "label": label,
        }
    )
    return compute_derived_features(df)


def run_selftest(models_dir: str):
    ensure_dir(models_dir)
    trainer = IDS_Trainer(output_dir=models_dir)
    df = generate_synthetic_dataset()
    tmp_csv = os.path.join(models_dir, "synthetic_train.csv")
    df.to_csv(tmp_csv, index=False)
    trainer.run_training_pipeline(tmp_csv)


    ids = VanetIDS(models_dir=models_dir)
    det_df = generate_synthetic_dataset(n_cars=40, n_msgs=4000, seed=19).drop(
        columns=["label"]
    )
    det_in = os.path.join(models_dir, "synthetic_detect.csv")
    det_out = os.path.join(models_dir, "synthetic_out.csv")
    det_df.to_csv(det_in, index=False)
    ids.process_bsm_logfile(det_in, det_out)


ascii_art = r"""
****************************************************************************************************
*  _   _ _   _ _____     _______ ____  ____ ___ _______   __   ___  _____   _  ___   _ _____ _     *
* | | | | \ | |_ _\ \   / / ____|  _ \/ ___|_ _|_   _\ \ / /  / _ \|  ___| | |/ / | | |  ___/ \    *
* | | | |  \| || | \ \ / /|  _| | |_) \___ \| |  | |  \ V /  | | | | |_    | ' /| | | | |_ / _ \   *
* | |_| | |\  || |  \ V / | |___|  _ < ___) | |  | |   | |   | |_| |  _|   | . \| |_| |  _/ ___ \  *
*  \___/|_| \_|___|  \_/  |_____|_| \_\____/___| |_|   |_|    \___/|_|     |_|\_\\___/|_|/_/   \_\ *
****************************************************************************************************
                         by Mohammad Abbas Shareef  & Dr.Fahad Ghalib
"""


DEFAULT_F2MD_DIR = "/home/instantf2md/F2MD"
DEFAULT_RESULTS_DIR = "/home/instantf2md/F2MD/f2md-results"
DEFAULT_EXTRACT_PY = os.path.join(DEFAULT_F2MD_DIR, "extract1_intermsg.py")
DEFAULT_ATTACK_ID_PY = os.path.join(os.path.dirname(__file__), "add_attack_id.py")
DEFAULT_LIVE_IDS_DIR = "/home/instantf2md/Desktop/VANET-IRAQ Live IDS Dashboard"
DEFAULT_RSU_TRAINER_PY = (
    "/home/instantf2md/Desktop/VANET_project/rsu_trainer_all_in_one_v7.py"
)
DEFAULT_BAGHDAD_OMNETPP = (
    "/home/instantf2md/F2MD/veins-f2md/f2md-networks/BaghdadScenario/omnetpp.ini"
)


ATTACK_TYPES = [
    (0, "Genuine"),
    (1, "ConstPos"),
    (2, "ConstPosOffset"),
    (3, "RandomPos"),
    (4, "RandomPosOffset"),
    (5, "ConstSpeed"),
    (6, "ConstSpeedOffset"),
    (7, "RandomSpeed"),
    (8, "RandomSpeedOffset"),
    (9, "EventualStop"),
    (10, "Disruptive"),
    (11, "DataReplay"),
    (12, "StaleMessages"),
    (13, "DoS"),
    (14, "DoSRandom"),
    (15, "DoSDisruptive"),
    (16, "GridSybil"),
    (17, "DataReplaySybil"),
    (18, "DoSRandomSybil"),
    (19, "DoSDisruptiveSybil"),
]


def log_cmd(cmd: str):
    log(f"🧰 CMD: {cmd}")


def run_blocking(cmd_list: list[str], cwd: str | None = None):
    log_cmd(" ".join(shlex.quote(c) for c in cmd_list))
    subprocess.run(cmd_list, cwd=cwd, check=True)


def open_in_new_terminal(cmd: str, cwd: str | None = None, title: str = "VANET job"):
    qterm = shutil.which("qterminal")
    if not qterm:
        raise RuntimeError(
            "qterminal not found. Install it: sudo apt-get install qterminal"
        )
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "xcb")
    env.setdefault("DISPLAY", ":0")
    workdir = cwd or os.getcwd()
    cmd_str = cmd + "; exec bash"
    argv = [qterm, "--title", title, "--workdir", workdir, "-e", "bash", "-lc", cmd_str]
    subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def cmd_launch_daemon(f2md_dir: str) -> str:
    return f"cd {shlex.quote(f2md_dir)} && ./launchSumoTraciDaemon"


def cmd_run_scenario(f2md_dir: str) -> str:
    return f"cd {shlex.quote(f2md_dir)} && ./runScenario"


def run_extract_intermsg(
    results_dir: str,
    root: str,
    out_csv: str,
    version: str = "v2",
    py: str | None = None,
):
    script = os.path.join(results_dir, "extract1_intermsg.py")
    if not os.path.exists(script):

        script = DEFAULT_EXTRACT_PY
    if py is None:

        py = sys.executable if sys.executable else "python3"
    cmd = [py, script, "--root", root, "--out", out_csv, "--version", version]
    run_blocking(cmd, cwd=results_dir)


def print_attack_types():
    print("\nAttack types (ID: name):")
    for aid, name in ATTACK_TYPES:
        print(f"  {aid}: {name}")
    print()


def prompt_attack_id(default: int = 0) -> int:
    valid_ids = {aid for aid, _ in ATTACK_TYPES}
    while True:
        print_attack_types()
        raw = input(f"Enter attack type ID [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print("⚠️ Please enter a number from the list.")
            continue
        if val in valid_ids:
            return val
        print("⚠️ Invalid ID. Choose one of the numbers shown above.")


def add_attack_id_to_csv(
    csv_path: str,
    attack_id: int,
    output_path: str | None = None,
    script_path: str = DEFAULT_ATTACK_ID_PY,
) -> str:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"add_attack_id.py not found at {script_path}. Please place it next to vanet_ids_v2.py."
        )

    py = sys.executable if sys.executable else "python3"
    cmd = [py, script_path, "--input", csv_path, "--attack-id", str(int(attack_id))]
    if output_path and output_path != csv_path:
        cmd += ["--output", output_path]
    else:
        cmd.append("--inplace")
        output_path = csv_path

    print(f"➕ Adding attack_id={attack_id} to {csv_path} ...")
    run_blocking(cmd, cwd=os.path.dirname(script_path) or None)
    return output_path


def run_verify(labels: str, detect: str, models_dir: str, outdir: str):
    script = "summarize_attacks_vs_model.py"
    if not os.path.exists(script):
        log(
            "⚠️ summarize_attacks_vs_model.py غير موجود في هذا المجلد. سأحاول استدعاءه على أي حال..."
        )
    py = sys.executable if sys.executable else "python3"
    cmd = [
        py,
        script,
        "--labels",
        labels,
        "--detect",
        detect,
        "--models_dir",
        models_dir,
        "--outdir",
        outdir,
    ]
    run_blocking(cmd, cwd=os.getcwd())


def run_live_ids_dashboard(
    live_dir: str = DEFAULT_LIVE_IDS_DIR, conda_env: str = "base"
):
    if not os.path.isdir(live_dir):
        raise FileNotFoundError(f"Live IDS directory not found: {live_dir}")
    main_py = os.path.join(live_dir, "main.py")
    if not os.path.exists(main_py):
        raise FileNotFoundError(f"main.py not found in {live_dir}")

    conda = shutil.which("conda")
    if not conda:
        raise RuntimeError("conda not found in PATH داخل هذا التيرمنال")

    env = os.environ.copy()
    env.setdefault("DISPLAY", env.get("DISPLAY", ":0"))
    env.setdefault("QT_QPA_PLATFORM", "xcb")


    env.setdefault("QT_OPENGL", "software")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

    log_path = os.path.join(live_dir, "live_ids.log")
    logf = open(log_path, "ab", buffering=0)

    subprocess.Popen(
        [conda, "run", "--no-capture-output", "-n", conda_env, "python", "main.py"],
        cwd=live_dir,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )

    print(f"🚀 Live IDS Dashboard launched detached (conda env: {conda_env})")
    print(f"📝 If it doesn't open, check: {log_path}")


def run_rsu_trainer(trainer_path: str = DEFAULT_RSU_TRAINER_PY):
    if not os.path.exists(trainer_path):
        raise FileNotFoundError(f"Trainer script not found: {trainer_path}")
    py = sys.executable if sys.executable else "python3"
    trainer_dir = os.path.dirname(trainer_path) or "."
    launch_qterminal_command(
        [py, trainer_path],
        workdir=trainer_dir,
        title="VANET — RSU Trainer",
    )


def list_models(models_root: str):
    print("\n📦 Available model directories under:", os.path.abspath(models_root))
    found = False
    for name in sorted(os.listdir(models_root)):
        path = os.path.join(models_root, name)
        if os.path.isdir(path) and os.path.exists(
            os.path.join(path, "fusion_config_final_v2.json")
        ):
            print("  -", path)
            found = True
    if not found:
        print("  (no ready model directories found)")


def _ask(prompt: str, default: str | None = None, required: bool = False) -> str:
    if default:
        q = f"{prompt} [{default}]: "
    else:
        q = f"{prompt}: "
    while True:
        v = input(q).strip()
        if v:
            return os.path.expanduser(v)
        if default is not None:
            return os.path.expanduser(default)
        if not required:
            return ""
        print("⚠️ This value is required.")


def open_config_in_editor(path: str):
    print(f"\n📝 Opening Baghdad scenario config: {path}")
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    editor_env = os.environ.get("EDITOR")
    if editor_env:
        cmd = shlex.split(editor_env) + [path]
        try:
            subprocess.run(cmd, check=False)
            return
        except Exception as e:
            print(f"⚠️ Failed to open with EDITOR ({editor_env}): {e}")

    xdg = shutil.which("xdg-open")
    if xdg:
        try:
            subprocess.Popen(
                [xdg, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            print("💡 Using system default editor; close it after saving changes.")
            return
        except Exception as e:
            print(f"⚠️ xdg-open failed: {e}")

    for candidate in ("gedit", "xed", "nano", "vi"):
        editor_path = shutil.which(candidate)
        if not editor_path:
            continue
        try:
            subprocess.run([editor_path, path], check=False)
            return
        except Exception as e:
            print(f"⚠️ Failed to open with {candidate}: {e}")

    print("❌ No text editor found. Please install one or set the EDITOR env var.")


DEFAULT_MODELS_DIR = "/home/instantf2md/Desktop/VANET_project/release_v2"


def interactive_menu():

    while True:
        print("\nChoose an action:")
        print("  1) Start SUMO/TraCI Daemon (in a new terminal)")
        print("  2) Run scenario (in a new terminal)")
        print("  3) Extract BSM messages to CSV")
        print("  4) Launch Live IDS Dashboard")
        print("  5) Train RSU model (all-in-one)")
        print("  6) Edit Baghdad scenario config (omnetpp.ini)")
        print("  7) Exit")
        choice = input("\nEnter choice number: ").strip()

        if choice == "1":
            f2md_dir = DEFAULT_F2MD_DIR
            launch_qterminal_detached(
                script_path=os.path.join(f2md_dir, "run_daemon_in_term.sh"),
                workdir=f2md_dir,
                title="VANET — Daemon",
            )

        elif choice == "2":
            f2md_dir = DEFAULT_F2MD_DIR
            launch_qterminal_detached(
                script_path=os.path.join(f2md_dir, "run_scenario_in_term.sh"),
                workdir=f2md_dir,
                title="VANET — Scenario",
            )

        elif choice == "3":
            results_dir = DEFAULT_RESULTS_DIR
            default_root = os.path.join(results_dir, "LuSTNanoScenario-ITSG5")
            root = _ask("BSM root directory (parent of MDBsms_* folders)", default_root)
            out_csv = _ask(
                "Output CSV path",
                os.path.join(results_dir, "features_intermessage_v2.csv"),
            )
            version = _ask("BSM version (v1/v2)", "v2").lower() or "v2"
            try:
                run_extract_intermsg(
                    results_dir=results_dir, root=root, out_csv=out_csv, version=version
                )
                print(f"✅ Extraction finished: {out_csv}")


                attack_id = prompt_attack_id(default=0)
                overwrite_ans = _ask(
                    "Overwrite extracted CSV with attack_id? [Y/n]", "Y"
                ).lower()
                if overwrite_ans.startswith("n"):
                    out_attack = _ask(
                        "Save attack_id version to",
                        f"{os.path.splitext(out_csv)[0]}_with_attackid.csv",
                    )
                else:
                    out_attack = out_csv

                try:
                    out_path = add_attack_id_to_csv(
                        csv_path=out_csv, attack_id=attack_id, output_path=out_attack
                    )
                    print(
                        f"✅ Added attack_id={attack_id} ({dict(ATTACK_TYPES).get(attack_id, 'unknown')}) -> {out_path}"
                    )
                except Exception as e:
                    print(f"⚠️ attack_id step failed: {e}")
            except Exception as e:
                print(f"❌ Extraction failed: {e}")

        elif choice == "4":

            try:
                run_live_ids_dashboard(DEFAULT_LIVE_IDS_DIR)
                print("🚀 Live IDS Dashboard started in a new terminal.")
            except Exception as e:
                print(f"❌ Failed to start Live IDS Dashboard: {e}")

        elif choice == "5":
            project_dir = "/home/instantf2md/Desktop/VANET_project"
            rsu_path = (
                "/home/instantf2md/Desktop/VANET_project/rsu_trainer_all_in_one_v7.py"
            )
            log_path = os.path.join(project_dir, "logs", "rsu_trainer_menu.log")

            pid = launch_detached([sys.executable, rsu_path], log_path, cwd=project_dir)
            print(f"🧠 RSU trainer launched (PID={pid}). Log: {log_path}")

        elif choice == "6":
            open_config_in_editor(DEFAULT_BAGHDAD_OMNETPP)

        elif choice == "7" or choice.lower() in ("q", "quit", "exit"):
            print("👋 Bye.")
            return

        else:
            print("⚠️ Invalid choice. Please select a number from 1 to 7.")


def launch_qterminal_detached(
    script_path: str, workdir: str, title: str = "VANET job"
) -> None:
    qterm = shutil.which("qterminal")
    if not qterm:
        raise RuntimeError(
            "qterminal not found. Install it: sudo apt-get install qterminal"
        )

    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "xcb")
    env.setdefault("DISPLAY", ":0")

    argv = [qterm, "--title", title, "--workdir", workdir, "-e", script_path]


    subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def launch_qterminal_command(
    cmd_list: list[str], workdir: str, title: str = "VANET job"
) -> None:
    qterm = shutil.which("qterminal")
    if not qterm:
        raise RuntimeError(
            "qterminal not found. Install it: sudo apt-get install qterminal"
        )

    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "xcb")
    env.setdefault("DISPLAY", ":0")

    cmd_str = " ".join(shlex.quote(c) for c in cmd_list)
    cmd_str = (
        f"cd {shlex.quote(workdir)} && {cmd_str} 2>&1 | tee -a live_ids.log; exec bash"
    )

    argv = [qterm, "--title", title, "-e", "bash", "-lc", cmd_str]

    subprocess.Popen(argv, env=env, start_new_session=True, close_fds=True)


def main():
    print(ascii_art)


    if len(sys.argv) == 1:
        interactive_menu()
        return


    parser = argparse.ArgumentParser(
        description="VANET IDS v2: training & detection pipeline (OBU + RSU + Trust) + interactive menu.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=[
            "menu",
            "train",
            "detect",
            "selftest",
            "start-daemon",
            "run-scenario",
            "live-ids",
            "train-rsu",
            "extract",
            "verify",
            "list-models",
        ],
        help=(
            "menu: show quick actions (also default when no args)\n"
            "train/detect/selftest: pipeline modes\n"
            "start-daemon/run-scenario: run F2MD helpers in new terminal\n"
            "live-ids: launch live IDS dashboard (main.py)\n"
            "train-rsu: launch rsu_trainer_all_in_one_v7.py in new terminal\n"
            "extract: build features CSV\n"
            "verify: LABEL vs model summary\n"
            "list-models: list ready model dirs\n"
        ),
    )
    parser.add_argument("--models_dir", default="release_v2")
    parser.add_argument("--input")
    parser.add_argument("--output")

    parser.add_argument("--f2md_dir", default=DEFAULT_F2MD_DIR)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--live_ids_dir", default=DEFAULT_LIVE_IDS_DIR)
    parser.add_argument("--trainer_py", default=DEFAULT_RSU_TRAINER_PY)

    parser.add_argument("--extract_root")
    parser.add_argument("--extract_out")
    parser.add_argument("--extract_version", default="v2")
    parser.add_argument(
        "--attack_id",
        type=int,
        help="(extract) Attack type ID (0-19) to append as attack_id; omit to skip.",
    )
    parser.add_argument(
        "--attack_out",
        help="(extract) Optional output CSV for attack_id step (defaults to overwrite extract_out).",
    )

    parser.add_argument("--labels")
    parser.add_argument("--detect_csv")
    parser.add_argument("--outdir", default="analysis_output")

    args = parser.parse_args()

    if args.mode == "menu":
        interactive_menu()
        return

    if args.mode == "start-daemon":
        open_in_new_terminal(cmd_launch_daemon(args.f2md_dir), cwd=args.f2md_dir)
        return
    if args.mode == "run-scenario":
        open_in_new_terminal(cmd_run_scenario(args.f2md_dir), cwd=args.f2md_dir)
        return
    if args.mode == "live-ids":
        run_live_ids_dashboard(args.live_ids_dir)
        return
    if args.mode == "train-rsu":
        run_rsu_trainer(args.trainer_py)
        return
    if args.mode == "extract":
        root = args.extract_root or os.path.join(
            args.results_dir, "LuSTNanoScenario-ITSG5"
        )
        out_csv = args.extract_out or os.path.join(
            args.results_dir, "features_intermessage_v2.csv"
        )
        run_extract_intermsg(
            results_dir=args.results_dir,
            root=root,
            out_csv=out_csv,
            version=args.extract_version,
        )
        if args.attack_id is not None:
            out_attack = args.attack_out or out_csv
            try:
                add_attack_id_to_csv(
                    csv_path=out_csv,
                    attack_id=args.attack_id,
                    output_path=out_attack,
                )
                print(
                    f"✅ Added attack_id={args.attack_id} to {out_attack} (source: {out_csv})"
                )
            except Exception as e:
                print(f"⚠️ attack_id step failed: {e}")
        return
    if args.mode == "verify":
        if not args.labels or not args.detect_csv:
            raise ValueError("In verify mode, --labels and --detect_csv are required.")
        run_verify(
            labels=args.labels,
            detect=args.detect_csv,
            models_dir=args.models_dir,
            outdir=args.outdir,
        )
        return
    if args.mode == "list-models":
        list_models(args.models_dir if os.path.isdir(args.models_dir) else ".")
        return

    if args.mode == "train":
        if not args.input or not args.input.endswith(".csv"):
            raise ValueError("In train mode, --input must be provided and be a CSV.")
        trainer = IDS_Trainer(output_dir=args.models_dir)
        trainer.run_training_pipeline(labeled_csv_path=args.input)
        return

    if args.mode == "detect":
        if not args.input or not args.output:
            raise ValueError("In detect mode, --input and --output are required.")
        ids = VanetIDS(models_dir=args.models_dir)
        ids.process_bsm_logfile(input_csv_path=args.input, output_csv_path=args.output)
        return

    if args.mode == "selftest":
        run_selftest(args.models_dir)
        return


if __name__ == "__main__":
    main()
