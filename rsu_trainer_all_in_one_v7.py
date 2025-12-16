from __future__ import annotations

eps_small = 1e-6
eps = 1e-6

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
import lightgbm as lgb


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)




ATTACK_FAMILIES = {
    "pos_speed": {1,2,3,4,6,7,9},
    "replay_stale": {11,12},
    "dos": {13,14,15},
    "sybil": {16,18,19},
    "disruptive": {10}
}

ALL_FAMILIES = list(ATTACK_FAMILIES.keys())




def quick_checks_add_evidence_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()


    def S(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        else:
            return pd.Series(default, index=out.index, dtype=float)

    speed = S("speed_curr")
    acc   = S("acc_curr")
    hr    = S("heading_rate")
    dt    = S("dt")
    dist  = S("dist")


    out["flag_speed_phys"] = (speed > 80.0).astype(int)
    out["flag_acc_phys"]   = (acc.abs() > 12.0).astype(int)
    out["flag_hr_phys"]    = (hr.abs() > 2.0).astype(int)


    consistency = (dist - speed*dt).abs()
    out["consistency_err"] = consistency
    out["flag_consistency"] = (consistency > np.maximum(1.0, 0.25*speed*dt)).astype(int)


    cols_to_check = [c for c in ["dt","speed_curr","acc_curr","heading_curr"] if c in out.columns]
    if cols_to_check:
        out["flag_proto_nan"] = out[cols_to_check].replace([np.inf, -np.inf], np.nan).isna().any(axis=1).astype(int)
    else:
        out["flag_proto_nan"] = 0
    out["flag_dt_nonpos"] = (dt <= 0).astype(int)

    proto_cols = ["flag_proto_nan","flag_dt_nonpos"]
    out["proto_anom_count"] = out[proto_cols].sum(axis=1)

    return out




def feature_engineering(df: pd.DataFrame, sender_col='sender_pseudo', time_col='t_curr',
                        window_size=15, eps=1e-6) -> pd.DataFrame:
    df = df.copy()
    time_col = time_col if time_col in df.columns else ("time" if "time" in df.columns else time_col)
    df = df.sort_values([sender_col, time_col]).reset_index(drop=True)
    g = df.groupby(sender_col, sort=False)


    if "dt" not in df.columns:
        df["dt"] = g[time_col].diff().fillna(0.0)


    if "dist" not in df.columns:
        if {"x_curr","y_curr"}.issubset(df.columns):
            df["dx"] = g["x_curr"].diff().fillna(0.0)
            df["dy"] = g["y_curr"].diff().fillna(0.0)
            df["dist"] = np.hypot(df["dx"], df["dy"])
        else:
            df["dx"] = df.get("dx", 0.0)
            df["dy"] = df.get("dy", 0.0)
            df["dist"] = df.get("dist", 0.0)

    df["dr_dx"] = 0.0; df["dr_dy"] = 0.0
    if {"x_curr","y_curr"}.issubset(df.columns):
        x_prev = g["x_curr"].shift(1).fillna(df["x_curr"])
        y_prev = g["y_curr"].shift(1).fillna(df["y_curr"])
        cosh = np.cos(df["heading_prev"].fillna(0.0))
        sinh = np.sin(df["heading_prev"].fillna(0.0))
        vx = df["speed_prev"] * cosh
        vy = df["speed_prev"] * sinh
        x_pred = x_prev + vx * df["dt"]
        y_pred = y_prev + vy * df["dt"]
        df["dr_dx"] = df["x_curr"] - x_pred
        df["dr_dy"] = df["y_curr"] - y_pred

    df["dr_angle"] = np.arctan2(df["dr_dy"], df["dr_dx"])
    df["sin_a"] = np.sin(df["dr_angle"])
    df["cos_a"] = np.cos(df["dr_angle"])
    mcos = g["cos_a"].rolling(window=window_size, min_periods=2).mean().reset_index(level=0, drop=True)
    msin = g["sin_a"].rolling(window=window_size, min_periods=2).mean().reset_index(level=0, drop=True)

    df[f"dr_angle_var_w{window_size}"] = 1.0 - np.sqrt(mcos**2 + msin**2)


    if "speed_curr" not in df.columns:
        df["speed_curr"] = df["dist"]/np.clip(df["dt"], eps, None)
    df["speed_prev"] = g["speed_curr"].shift(1).fillna(df["speed_curr"])
    df["dv"] = df["speed_curr"] - df["speed_prev"]
    df["acc_curr"] = df["dv"]/np.clip(df["dt"], eps, None)
    df["acc_prev"] = g["acc_curr"].shift(1).fillna(df["acc_curr"])
    df["dacc_jerk"] = (df["acc_curr"]-df["acc_prev"])/np.clip(df["dt"], eps, None)
    df["neg_acc_flag"]   = (df["acc_curr"] < -0.30).astype(int)
    df["low_speed_flag"] = (df["speed_curr"] < 0.50).astype(int)

    df[f"neg_acc_ratio_w{window_size}"] = (
        g["neg_acc_flag"].rolling(window=window_size, min_periods=2).mean()
        .reset_index(level=0, drop=True)
    )
    df[f"low_speed_ratio_w{window_size}"] = (
        g["low_speed_flag"].rolling(window=window_size, min_periods=2).mean()
        .reset_index(level=0, drop=True)
    )


    if "heading_curr" not in df.columns:
        df["heading_curr"] = df.get("heading_curr", 0.0)
    df["heading_prev"] = g["heading_curr"].shift(1).fillna(df["heading_curr"])
    df["dtheta"] = df["heading_curr"] - df["heading_prev"]
    df["heading_rate"] = df["dtheta"]/np.clip(df["dt"], eps, None)


    eff = int(window_size) - 1 if int(window_size) > 1 else 1
    span_series = g[time_col].transform(lambda s: s - s.shift(eff))
    df['rate_msgs_per_s'] = eff / np.clip(span_series.astype(float), 1e-6, None)
    df['rate_msgs_per_s'] = g['rate_msgs_per_s'].transform(lambda s: s.bfill().ffill())


    roll_cols = ["dv","dacc_jerk","heading_rate","dist","dt"]
    stats = g[roll_cols].rolling(window=window_size, min_periods=2).agg(["mean","std","max"])
    stats.columns = [f"{c}_{s}_w{window_size}" for c,s in stats.columns]
    df = pd.concat([df, stats.reset_index(level=0, drop=True)], axis=1)


    df[f"dt_jitter_w{window_size}"] = g["dt"].rolling(window=window_size, min_periods=2).std().reset_index(level=0, drop=True)
    def freeze_ratio(series, thr):
        return g.apply(lambda x: (np.abs(series.loc[x.index]) < thr).rolling(window=window_size, min_periods=2).mean()
                      ).reset_index(level=0, drop=True)
    df[f"freeze_ratio_dv_w{window_size}"]   = freeze_ratio(df["dv"], 1e-4)
    df[f"freeze_ratio_dist_w{window_size}"] = freeze_ratio(df["dist"], 1e-3)
    df[f"freeze_ratio_hr_w{window_size}"]   = freeze_ratio(df["heading_rate"], 1e-4)


    df["consistency_err"] = (df["dist"] - df["speed_curr"]*df["dt"]).abs()
    df[f"consistency_err_mean_w{window_size}"] = g["consistency_err"].rolling(window=window_size, min_periods=2).mean().reset_index(level=0, drop=True)







    def _state_hash(row):
        return (
            round(row.get("x_curr", 0.0), 3),
            round(row.get("y_curr", 0.0), 3),
            round(row.get("speed_curr", 0.0), 2),
            round(row.get("heading_curr", 0.0), 2),
        )


    df["state_hash"] = df.apply(_state_hash, axis=1)
    g = df.groupby(sender_col, sort=False)


    df["state_code"] = g["state_hash"].transform(lambda s: pd.factorize(s)[0].astype("int64"))


    def _dup_ratio_np(w):
        w = w.astype("int64", copy=False)
        return 1.0 - (np.unique(w).size / max(1, w.size))

    df["state_dup_ratio_w"] = g["state_code"].transform(
        lambda s: s.rolling(window=window_size, min_periods=2).apply(_dup_ratio_np, raw=True)
    )


    med_dt = g["dt"].transform(lambda s: s.rolling(window=window_size, min_periods=2).median())
    mad_dt = g["dt"].transform(
        lambda s: s.rolling(window=window_size, min_periods=2).apply(
            lambda w: np.median(np.abs(w - np.median(w))) if len(w) else 0.0, raw=True
        ) + 1e-6
    )
    df["dt_z"] = (df["dt"] - med_dt) / mad_dt

    mu_dt = g["dt"].transform(lambda s: s.rolling(window=window_size, min_periods=2).mean())
    sd_dt = g["dt"].transform(lambda s: s.rolling(window=window_size, min_periods=2).std())
    df["dt_cv_w"] = sd_dt / (mu_dt + 1e-6)

    df["dt_z"] = (df["dt"] - med_dt) / mad_dt


    mu_dt = g["dt"].transform(lambda s: s.rolling(window=window_size, min_periods=2).mean())
    sd_dt = g["dt"].transform(lambda s: s.rolling(window=window_size, min_periods=2).std())
    df["dt_cv_w"] = sd_dt / (mu_dt + 1e-6)

    alpha = 0.3
    df["rate_ewma"] = g["rate_msgs_per_s"].transform(lambda s: s.ewm(span=window_size, adjust=False).mean())
    r = df["rate_msgs_per_s"] - df["rate_ewma"]
    df["rate_cusum_pos"] = g.apply(lambda x: np.maximum.accumulate(np.maximum(0, r.loc[x.index].cumsum()))).reset_index(level=0, drop=True)
    df["rate_cusum_neg"] = g.apply(lambda x: np.maximum.accumulate(np.maximum(0, (-r.loc[x.index]).cumsum()))).reset_index(level=0, drop=True)


    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df




def build_lstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), input_shape=input_shape),
        LayerNormalization(),
        Dropout(0.2),
        Bidirectional(LSTM(64, activation='tanh')),
        LayerNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model

def _fit_replay_head(self, X_df_scaled: pd.DataFrame, y: pd.Series, groups: pd.Series):
    X_seq, y_seq, g_seq, idx_last = make_sequences_per_sender(X_df_scaled, y, groups, seq_len=self.seq_len)
    if len(X_seq)==0:
        self.lstm = None; self.lstm_shape=None;
        return np.zeros(len(X_df_scaled)), np.array([], dtype=int)

    self.lstm_shape = (X_seq.shape[1], X_seq.shape[2])


    pos = float(y_seq.mean() + 1e-9)
    class_weight = {0: 1.0, 1: max(1.0, (1.0-pos)/pos)}


    oof = pd.Series(0.0, index=X_df_scaled.index, dtype=float)

    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in gkf.split(X_seq, y_seq, g_seq):
        m = build_lstm(self.lstm_shape)
        es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        m.fit(X_seq[tr], y_seq[tr], epochs=40, batch_size=256,
              validation_data=(X_seq[va], y_seq[va]),
              callbacks=[es], verbose=0, class_weight=class_weight)
        p = m.predict(X_seq[va], verbose=0).ravel()
        oof.loc[idx_last[va]] = p


    self.lstm = build_lstm(self.lstm_shape)
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    self.lstm.fit(X_seq, y_seq, epochs=40, batch_size=256, validation_split=0.2,
                  callbacks=[es], verbose=0, class_weight=class_weight)


    oof_sm = oof.copy()
    for sid, idxs in X_df_scaled.groupby(groups, sort=False).groups.items():
        idxs = list(sorted(list(idxs)))
        vals = oof_sm.loc[idxs].values
        if len(vals) >= 3:
            vals = pd.Series(vals).ewm(alpha=0.3, adjust=False).mean().values
        oof_sm.loc[idxs] = vals

    return oof_sm.values, idx_last

def make_sequences_per_sender(X_df: pd.DataFrame, y: pd.Series, groups: pd.Series, seq_len=20):
    feats = X_df.columns.tolist()
    Xs, ys, gs, idx_last = [], [], [], []
    for sid, idxs in X_df.groupby(groups, sort=False).groups.items():
        idxs = np.array(sorted(list(idxs)))
        Xi = X_df.loc[idxs, feats].values
        yi = y.loc[idxs].values
        if len(Xi) < seq_len: continue
        for s in range(0, len(Xi)-seq_len+1):
            e = s+seq_len
            Xs.append(Xi[s:e]); ys.append(yi[e-1]); gs.append(sid); idx_last.append(idxs[e-1])
    if not Xs:
        return np.zeros((0,seq_len,len(feats))), np.zeros((0,)), np.array([]), np.array([])
    return np.stack(Xs), np.array(ys), np.array(gs), np.array(idx_last)




class TrustManager:
    def __init__(self, alpha0=1.0, beta0=1.0):
        self.map: Dict[str, Tuple[float,float]] = {}
        self.alpha0, self.beta0 = alpha0, beta0

    def _get(self, sid):
        return self.map.get(sid, (self.alpha0, self.beta0))

    def update(self, sid, is_malicious: bool, w: float=1.0):
        a,b = self._get(sid)
        if is_malicious: b += w
        else: a += w
        self.map[sid] = (a,b)

    def trust(self, sid) -> float:
        a,b = self._get(sid); return a/(a+b)

    def from_quick_evidence(self, row, w=0.5) -> Tuple[float,float]:
        reward = 0.0; penal = 0.0
        for c in ["flag_speed_phys","flag_acc_phys","flag_hr_phys","flag_consistency","flag_proto_nan","flag_dt_nonpos"]:
            if c in row and row[c]==1: penal += w

        if penal==0.0: reward += 0.25*w
        return reward, penal

    def adaptive_threshold(self, sid, base_thr=0.5, sensitivity=0.55, floor=0.35, ceil=0.85):
        t = self.trust(sid)
        adj = sensitivity*(t-0.5)*2.0
        thr = float(np.clip(base_thr+adj, floor, ceil))
        return thr





class RSUTrainer:
    def __init__(self, train_family: str = "binary", seq_len: int = 20, window_size: int = 25):
        assert train_family in {"binary","pos_speed","replay_stale","dos","sybil","disruptive","all"}
        self.train_family = train_family
        self.seq_len = seq_len
        self.window_size = int(window_size)


        self.scaler = StandardScaler()
        self.bin_clf = lgb.LGBMClassifier( objective='binary', class_weight='balanced', n_estimators=800, learning_rate=0.03, num_leaves=192, min_data_in_leaf=20, feature_fraction=0.85, random_state=42 , min_gain_to_split=1e-12, verbosity=-1)
        self.bin_calib: Optional[CalibratedClassifierCV] = None


        self.head_pos: Optional[lgb.LGBMClassifier] = None
        self.head_pos_calib: Optional[CalibratedClassifierCV] = None

        self.head_dos: Optional[lgb.LGBMClassifier] = None
        self.head_dos_calib: Optional[CalibratedClassifierCV] = None
        self.iforest = IsolationForest(n_estimators=300, contamination=0.02, random_state=42)
        self.if_min, self.if_max = 0.0, 1.0

        self.head_sybil: Optional[lgb.LGBMClassifier] = None
        self.head_sybil_calib: Optional[CalibratedClassifierCV] = None

        self.head_disr: Optional[lgb.LGBMClassifier] = None
        self.head_disr_calib: Optional[CalibratedClassifierCV] = None

        self.lstm: Optional[tf.keras.Model] = None
        self.lstm_shape: Optional[Tuple[int,int]] = None


        self.meta = LogisticRegression(class_weight='balanced', max_iter=300, random_state=42)


        self.sender_col = "sender_pseudo"
        self.time_col = "t_curr"
        self.label_col = "label"
        self.attack_col = "attack_id"


    @staticmethod
    def _family_from_ids(ids: List[int]) -> Optional[str]:
        present = set(ids)
        for fam, s in ATTACK_FAMILIES.items():
            if present & s:
                return fam
        return None

    @staticmethod
    def _build_family_labels(df: pd.DataFrame, fam: str) -> pd.Series:
        if "attack_id" not in df.columns:
            return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        ids = pd.to_numeric(df["attack_id"], errors="coerce").fillna(-1).astype(int)
        return ids.isin(ATTACK_FAMILIES[fam]).astype(int)

    def _dos_features_cols(self, Xcols: List[str]) -> List[str]:
        return [c for c in Xcols if c.startswith("rate_") or c.startswith("dt_jitter") or c.endswith("_dt")]

    def _sybil_features_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["window_id"] = (out[self.time_col] // 5).astype(int)

        grp = out.groupby("window_id")


        out["sybil_unique_ids_5s"] = grp[self.sender_col].transform("nunique")


        win_entropy = grp[self.sender_col].apply(
            lambda s: -(s.value_counts(normalize=True).apply(lambda p: p*np.log(p+1e-9))).sum()
        )
        out["sybil_sender_entropy_5s"] = out["window_id"].map(win_entropy)


        win_sets = grp[self.sender_col].apply(lambda s: set(s.values))
        jacc = win_sets.index.to_series().map(
            lambda w: (len(win_sets.get(w, set()) & win_sets.get(w-1, set())) /
                    max(1, len(win_sets.get(w, set()) | win_sets.get(w-1, set()))))
        ).fillna(0.0)
        out["sybil_jaccard_ids_5s"] = out["window_id"].map(jacc)


        first_win = out.groupby(self.sender_col)["window_id"].transform("min")
        out["sybil_new_id_flag"] = (out["window_id"] == first_win).astype(int)
        rate_by_win = grp["sybil_new_id_flag"].mean()
        ewma = rate_by_win.ewm(span=8, adjust=False).mean()
        z_like = (rate_by_win - ewma).fillna(0.0)
        out["sybil_new_ids_rate"] = out["window_id"].map(rate_by_win)
        out["sybil_new_ids_burst"] = out["window_id"].map(z_like)


        out.drop(columns=["sybil_new_id_flag"], inplace=True, errors="ignore")
        return out


    def _prepare(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        d0 = quick_checks_add_evidence_features(df_raw)
        d1 = feature_engineering(d0, sender_col=self.sender_col, time_col=self.time_col,
                                 window_size=self.window_size)
        d2 = self._sybil_features_cols(d1)
        return d2


    def _fit_binary(self, Xs: np.ndarray, y: pd.Series):
        self.bin_calib = CalibratedClassifierCV(self.bin_clf, method='isotonic', cv=5)
        self.bin_calib.fit(Xs, y)

    def _fit_pos_head(self, Xs: np.ndarray, y_fam: pd.Series):
        self.head_pos = lgb.LGBMClassifier( objective='binary', class_weight='balanced', n_estimators=800, learning_rate=0.05, num_leaves=128, min_data_in_leaf=40, feature_fraction=0.9, random_state=42 , min_gain_to_split=1e-12, verbosity=-1)
        self.head_pos_calib = CalibratedClassifierCV(self.head_pos, method='isotonic', cv=5)
        self.head_pos_calib.fit(Xs, y_fam)

    def _fit_dos_head(self, Xs: np.ndarray, y_fam: pd.Series, dos_cols_mask: np.ndarray):
        X_dos = Xs[:, dos_cols_mask] if dos_cols_mask.sum()>0 else Xs
        self.head_dos = lgb.LGBMClassifier( objective='binary', class_weight='balanced', n_estimators=800, learning_rate=0.05, num_leaves=128, min_data_in_leaf=40, feature_fraction=0.9, random_state=42 , min_gain_to_split=1e-12, verbosity=-1)
        self.head_dos_calib = CalibratedClassifierCV(self.head_dos, method='isotonic', cv=5)
        self.head_dos_calib.fit(X_dos, y_fam)

        normal_mask = (y_fam.values==0)
        if normal_mask.sum()>10:
            self.iforest.fit(X_dos[normal_mask])
            sco = -self.iforest.score_samples(X_dos)
            self.if_min, self.if_max = np.percentile(sco,[1,99])
            if self.if_max<=self.if_min: self.if_max = self.if_min+1e-6

    def _fit_sybil_head(self, Xs: np.ndarray, y_fam: pd.Series, X_cols: List[str]):

        sel = [i for i,c in enumerate(X_cols) if c.startswith("sybil_") or c=="rate_msgs_per_s"]
        X_sy = Xs[:, sel] if sel else Xs
        self.head_sybil = lgb.LGBMClassifier( objective='binary', class_weight='balanced', n_estimators=800, learning_rate=0.05, num_leaves=128, min_data_in_leaf=40, feature_fraction=0.9, random_state=42 , min_gain_to_split=1e-12, verbosity=-1)
        self.head_sybil_calib = CalibratedClassifierCV(self.head_sybil, method='isotonic', cv=5)
        self.head_sybil_calib.fit(X_sy, y_fam)

    def _fit_disr_head(self, Xs: np.ndarray, y_fam: pd.Series, X_cols: List[str]):

        sel = [i for i,c in enumerate(X_cols) if c.startswith("flag_") or c=="proto_anom_count"]
        X_di = Xs[:, sel] if sel else Xs
        self.head_disr = lgb.LGBMClassifier( objective='binary', class_weight='balanced', n_estimators=600, learning_rate=0.05, num_leaves=96, min_data_in_leaf=30, feature_fraction=0.9, random_state=42 , min_gain_to_split=1e-12, verbosity=-1)
        self.head_disr_calib = CalibratedClassifierCV(self.head_disr, method='isotonic', cv=5)
        self.head_disr_calib.fit(X_di, y_fam)

def _fit_replay_head(self, X_df_scaled: pd.DataFrame, y: pd.Series, groups: pd.Series):
    X_seq, y_seq, g_seq, idx_last = make_sequences_per_sender(X_df_scaled, y, groups, seq_len=self.seq_len)
    if len(X_seq)==0:
        self.lstm = None; self.lstm_shape=None; return np.zeros(len(X_df_scaled)), np.array([], dtype=int)
    self.lstm_shape = (X_seq.shape[1], X_seq.shape[2])


    oof = pd.Series(0.0, index=X_df_scaled.index, dtype=float)

    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in gkf.split(X_seq, y_seq, g_seq):
        m = build_lstm(self.lstm_shape)
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        m.fit(X_seq[tr], y_seq[tr], epochs=30, batch_size=256, validation_data=(X_seq[va], y_seq[va]), callbacks=[es], verbose=0)
        p = m.predict(X_seq[va], verbose=0).ravel()

        oof.loc[idx_last[va]] = p


    self.lstm = build_lstm(self.lstm_shape)
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    self.lstm.fit(X_seq, y_seq, epochs=30, batch_size=256, validation_split=0.2, callbacks=[es], verbose=0)
    return oof.values, idx_last



    def fit_evaluate(self, df_raw: pd.DataFrame) -> Dict:
        df = self._prepare(df_raw)

        drop_cols = [self.sender_col, self.time_col, self.label_col, self.attack_col]
        X_cols = [c for c in df.columns if c not in drop_cols]
        y_bin = df[self.label_col].astype(int) if self.label_col in df.columns else pd.Series(np.zeros(len(df),dtype=int), index=df.index)
        groups = df[self.sender_col]


        fam = self.train_family
        if fam=="binary":
            pass
        elif fam=="all":
            pass
        else:

            if self.attack_col in df.columns:
                present_ids = pd.to_numeric(df[self.attack_col], errors="coerce").dropna().astype(int).unique().tolist()
                guessed = self._family_from_ids(present_ids)
                if guessed and guessed!=fam:
                    print(f"[INFO] Detected family in data: {guessed} (you set {fam}). Proceeding with {fam}.")
            else:
                print("[WARN] attack_id not found; family heads will use zeros.")


        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        tr_idx, te_idx = next(gkf.split(df, y_bin, groups))
        train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]


        X_train = train_df[X_cols]; X_test = test_df[X_cols]
        y_train_bin = y_bin.iloc[tr_idx]; y_test_bin = y_bin.iloc[te_idx]
        groups_train = groups.iloc[tr_idx]; groups_test = groups.iloc[te_idx]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)


        self._fit_binary(X_train_s, y_train_bin)
        p_bin_train = self.bin_calib.predict_proba(X_train_s)[:,1]
        p_bin_test  = self.bin_calib.predict_proba(X_test_s)[:,1]



        family_probs_train = {}
        family_probs_test  = {}

        fams_to_train = ALL_FAMILIES if self.train_family=="all" else ([self.train_family] if self.train_family!="binary" else [])


        dos_mask = np.array([c in self._dos_features_cols(X_cols) for c in X_cols])


        X_train_df_s = pd.DataFrame(X_train_s, index=X_train.index, columns=X_cols)
        X_test_df_s  = pd.DataFrame(X_test_s,  index=X_test.index,  columns=X_cols)

        for fam_name in fams_to_train:
            y_fam = self._build_family_labels(train_df, fam_name)
            y_fam_test = self._build_family_labels(test_df, fam_name)

            if fam_name=="pos_speed":
                self._fit_pos_head(X_train_s, y_fam)
                family_probs_train[fam_name] = self.head_pos_calib.predict_proba(X_train_s)[:,1]
                family_probs_test[fam_name]  = self.head_pos_calib.predict_proba(X_test_s)[:,1]

            elif fam_name=="replay_stale":
                oof_train, idx_last = self._fit_replay_head(X_train_df_s, y_fam, groups_train)
                family_probs_train[fam_name] = oof_train

                if self.lstm is not None:
                    X_seq_t, y_seq_t, g_seq_t, idx_last_t = make_sequences_per_sender(X_test_df_s, y_fam_test, groups_test, seq_len=self.seq_len)
                    p_t = np.zeros(len(X_test_df_s))
                    if len(X_seq_t)>0:
                        p_seq = self.lstm.predict(X_seq_t, verbose=0).ravel()
                        p_t[idx_last_t] = p_seq
                    family_probs_test[fam_name] = p_t
                else:
                    family_probs_test[fam_name] = np.zeros(len(X_test_df_s))

            elif fam_name=="dos":
                self._fit_dos_head(X_train_s, y_fam, dos_mask)
                Xtr_d = X_train_s[:, dos_mask] if dos_mask.sum()>0 else X_train_s
                Xte_d = X_test_s[:,  dos_mask] if dos_mask.sum()>0 else X_test_s
                family_probs_train[fam_name] = self.head_dos_calib.predict_proba(Xtr_d)[:,1]
                family_probs_test[fam_name]  = self.head_dos_calib.predict_proba(Xte_d)[:,1]

                sco_tr = -self.iforest.score_samples(Xtr_d)
                sco_te = -self.iforest.score_samples(Xte_d)
                def norm(s): return np.clip((s - self.if_min)/(self.if_max - self.if_min + 1e-9), 0, 1)
                family_probs_train["dos_iforest"] = norm(sco_tr)
                family_probs_test["dos_iforest"]  = norm(sco_te)

            elif fam_name=="sybil":
                self._fit_sybil_head(X_train_s, y_fam, X_cols)
                sel = [i for i,c in enumerate(X_cols) if c.startswith("sybil_") or c=="rate_msgs_per_s"]
                Xtr = X_train_s[:, sel] if sel else X_train_s
                Xte = X_test_s[:,  sel] if sel else X_test_s
                family_probs_train[fam_name] = self.head_sybil_calib.predict_proba(Xtr)[:,1]
                family_probs_test[fam_name]  = self.head_sybil_calib.predict_proba(Xte)[:,1]

            elif fam_name=="disruptive":
                self._fit_disr_head(X_train_s, y_fam, X_cols)
                sel = [i for i,c in enumerate(X_cols) if c.startswith("flag_") or c=="proto_anom_count"]
                Xtr = X_train_s[:, sel] if sel else X_train_s
                Xte = X_test_s[:,  sel] if sel else X_test_s
                family_probs_train[fam_name] = self.head_disr_calib.predict_proba(Xtr)[:,1]
                family_probs_test[fam_name]  = self.head_disr_calib.predict_proba(Xte)[:,1]


        def stack_build(pbin, fam_probs: Dict[str,np.ndarray]) -> np.ndarray:
            feats = [pbin]
            for k in ["pos_speed","replay_stale","dos","sybil","disruptive","dos_iforest"]:
                if k in fam_probs: feats.append(fam_probs[k])
            return np.vstack(feats).T

        meta_X_tr = stack_build(p_bin_train, family_probs_train)
        meta_X_te = stack_build(p_bin_test,  family_probs_test)
        self.meta.fit(meta_X_tr, y_train_bin)


        p_final = self.meta.predict_proba(meta_X_te)[:,1]
        prec, rec, th = precision_recall_curve(y_test_bin, p_final)
        f1 = (2*prec*rec)/(prec+rec+1e-9)
        best_thr = th[np.nanargmax(f1)] if len(th)>0 else 0.5


        y_pred_fixed = (p_final >= best_thr).astype(int)
        report = {
            "best_threshold": float(best_thr),
            "confusion_matrix_fixed": confusion_matrix(y_test_bin, y_pred_fixed),
            "cls_report_fixed": classification_report(y_test_bin, y_pred_fixed, digits=4, output_dict=True),
        }
        try:
            report["roc_auc"] = float(roc_auc_score(y_test_bin, p_final))
        except Exception:
            report["roc_auc"] = None


        tm = TrustManager()
        test_df_loc = test_df.copy()
        test_df_loc["p_final"] = p_final
        test_df_loc = test_df_loc.sort_values([self.sender_col, self.time_col])
        y_pred_adapt = []
        for _, row in test_df_loc.iterrows():
            rwd, pnl = tm.from_quick_evidence(row)
            sid = row[self.sender_col]
            if pnl>0: tm.update(sid, True, pnl)
            elif rwd>0: tm.update(sid, False, rwd)
            thr = tm.adaptive_threshold(sid, base_thr=best_thr, sensitivity=0.4)
            dec = int(row["p_final"] >= thr)
            y_pred_adapt.append(dec)
            tm.update(sid, bool(dec), 1.0)

        report["confusion_matrix_adaptive"] = confusion_matrix(y_test_bin.values, y_pred_adapt)
        report["cls_report_adaptive"] = classification_report(y_test_bin.values, y_pred_adapt, digits=4, output_dict=True)

        sample = list(groups_test.unique())[:5]
        report["sample_trust"] = {sid: round(tm.trust(sid),3) for sid in sample}
        report["used_families"] = fams_to_train
        return report


    def infer(self, df_new: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = self._prepare(df_new)
        X_cols = [c for c in df.columns if c not in [self.sender_col, self.time_col, self.label_col, self.attack_col]]
        Xs = self.scaler.transform(df[X_cols])


        p_bin = self.bin_calib.predict_proba(Xs)[:,1]


        fam_probs = {}

        if self.head_pos_calib is not None:
            fam_probs["pos_speed"] = self.head_pos_calib.predict_proba(Xs)[:,1]

        if self.lstm is not None:
            X_df_s = pd.DataFrame(Xs, index=df.index, columns=X_cols)
            y_dummy = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
            g = df[self.sender_col]
            X_seq, y_seq, gs, idx_last = make_sequences_per_sender(X_df_s, y_dummy, g, seq_len=self.seq_len)
            p_t = np.zeros(len(df))
            if len(X_seq)>0:
                p_seq = self.lstm.predict(X_seq, verbose=0).ravel()
                p_t[idx_last] = p_seq
            fam_probs["replay_stale"] = p_t

        if self.head_dos_calib is not None:
            dos_mask = np.array([c in self._dos_features_cols(X_cols) for c in X_cols])
            Xd = Xs[:, dos_mask] if dos_mask.sum()>0 else Xs
            fam_probs["dos"] = self.head_dos_calib.predict_proba(Xd)[:,1]
            sco = -self.iforest.score_samples(Xd)
            fam_probs["dos_iforest"] = np.clip((sco - self.if_min)/(self.if_max - self.if_min + 1e-9), 0,1)

        if self.head_sybil_calib is not None:
            sel = [i for i,c in enumerate(X_cols) if c.startswith("sybil_") or c=="rate_msgs_per_s"]
            Xsy = Xs[:, sel] if sel else Xs
            fam_probs["sybil"] = self.head_sybil_calib.predict_proba(Xsy)[:,1]

        if self.head_disr_calib is not None:
            sel = [i for i,c in enumerate(X_cols) if c.startswith("flag_") or c=="proto_anom_count"]
            Xdi = Xs[:, sel] if sel else Xs
            fam_probs["disruptive"] = self.head_disr_calib.predict_proba(Xdi)[:,1]


        def stack_build(pbin, fam_probs):
            feats = [pbin]
            for k in ["pos_speed","replay_stale","dos","sybil","disruptive","dos_iforest"]:
                if k in fam_probs: feats.append(fam_probs[k])
            return np.vstack(feats).T

        meta_X = stack_build(p_bin, fam_probs)
        p_final = self.meta.predict_proba(meta_X)[:,1]



        from sklearn.metrics import precision_recall_curve

        prec, rec, thrs = precision_recall_curve(y_test_bin, p_final)
        f1 = (2*prec*rec)/(prec+rec+1e-9)
        best_thr = thrs[np.nanargmax(f1)] if len(thrs)>0 else 0.5

        thr = float(best_thr)
        y_pred = (p_final >= thr).astype(int)

        report = {
            "best_threshold": float(thr),
            "confusion_matrix": confusion_matrix(y_test_bin, y_pred).tolist(),
            "cls_report": classification_report(y_test_bin, y_pred, digits=4, output_dict=True),
            "roc_auc": auc,
        }

        return p_final, preds



import sys, os, json, traceback, warnings
from typing import Dict, Any, Optional, List

warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QComboBox, QPlainTextEdit,
    QProgressBar, QFormLayout, QMessageBox, QTableWidget, QTableWidgetItem,
    QSizePolicy, QGroupBox, QCheckBox, QDoubleSpinBox
)

import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 3), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()

def _fit_numeric_transformer(df_tr: pd.DataFrame, cols: List[str], cat_col: Optional[str], logfn):
    Xtr = df_tr[cols].copy()
    ohe_values = None
    if cat_col and cat_col in Xtr.columns:
        ohe_values = Xtr[cat_col].astype(str).value_counts().index.tolist()[:20]
        for v in ohe_values:
            Xtr[f"{cat_col}__{v}"] = (Xtr[cat_col].astype(str) == str(v)).astype(int)
        Xtr = Xtr.drop(columns=[cat_col])
        logfn(f"[I] One-Hot لعمود {cat_col}: {len(ohe_values)} فئات (حتى 20).")

    obj_cols = Xtr.select_dtypes(include=['object','string','category']).columns.tolist()
    converted = []
    for c in obj_cols:
        s = pd.to_numeric(Xtr[c], errors='coerce')
        if s.notna().mean() >= 0.9:
            Xtr[c] = s; converted.append(c)
    if converted:
        logfn(f"[I] تحويل نص→رقم (TRAIN): {converted[:15]}{' ...' if len(converted)>15 else ''}")
    Xtr = Xtr.select_dtypes(include=[np.number]).replace([np.inf,-np.inf], np.nan)
    Xtr = Xtr.dropna(axis=1, how='all')
    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med)
    nun = Xtr.nunique(dropna=False)
    keep = nun[nun > 1].index.tolist()
    dropped_const = [c for c in Xtr.columns if c not in keep]
    if dropped_const:
        logfn(f"[W] أعمدة ثابتة (TRAIN): {dropped_const[:15]}{' ...' if len(dropped_const)>15 else ''}")
    return {"keep_cols": keep, "medians": med.to_dict(), "cat_col": cat_col, "ohe_values": ohe_values}

def _apply_numeric_transformer(df: pd.DataFrame, cols: List[str], tfm: dict) -> pd.DataFrame:
    X = df[cols].copy()
    cat_col = tfm.get("cat_col"); ohe_values = tfm.get("ohe_values")
    if cat_col and cat_col in X.columns and ohe_values is not None:
        for v in ohe_values:
            X[f"{cat_col}__{v}"] = (X[cat_col].astype(str) == str(v)).astype(int)
        X = X.drop(columns=[cat_col])
    for c in X.select_dtypes(include=['object','string','category']).columns.tolist():
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.select_dtypes(include=[np.number]).replace([np.inf,-np.inf], np.nan)
    med = tfm["medians"]
    X = X.fillna({k: v for k,v in med.items() if k in X.columns})
    keep = [c for c in tfm["keep_cols"] if c in X.columns]
    X = X.reindex(columns=keep, fill_value=0)
    return X


class TrainWorker(QObject):
    log = pyqtSignal(str)
    stage = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, csv_path: str, family: str, ohe_version: bool, save_model: bool, model_dir: str,
                 attack_ids: Optional[List[int]], split_mode: str, test_size: float):
        super().__init__()
        self.csv_path = csv_path
        self.family = family
        self.ohe_version = ohe_version
        self.save_model = save_model
        self.model_dir = model_dir
        self.attack_ids = attack_ids
        self.split_mode = split_mode
        self.test_size = float(test_size)

    def _emit(self, text: str):
        self.log.emit(text)

    def run(self):
        try:
            if 'RSUTrainer' not in globals():
                raise RuntimeError("تعذّر العثور على الصنف RSUTrainer داخل الملف.")

            self.stage.emit("قراءة البيانات"); self.progress.emit(5)
            self._emit(f"[I] تحميل CSV: {self.csv_path}")
            df_raw = pd.read_csv(self.csv_path, low_memory=False)
            self._emit(f"[I] عدد الصفوف: {len(df_raw):,}")

            self.stage.emit("تهيئة المدرب"); self.progress.emit(10)
            trainer = RSUTrainer(train_family=self.family, window_size=25)
            self._emit(f"[I] العائلة المختارة: {self.family}")

            sender_col = getattr(trainer, "sender_col", "sender_pseudo")
            time_col   = getattr(trainer, "time_col", "t_curr")
            label_col  = getattr(trainer, "label_col", "label")
            attack_col = getattr(trainer, "attack_col", "attack_id")

            if label_col not in df_raw or sender_col not in df_raw:
                raise RuntimeError("أعمدة label/sender غير موجودة في CSV.")


            if self.attack_ids:
                ids = set(int(x) for x in self.attack_ids)
                fam_mask = df_raw[attack_col].isin(ids)
                npos_before = int((df_raw[label_col]==1).sum())
                df_raw.loc[df_raw[label_col]==1, 'keep_pos'] = fam_mask & (df_raw[label_col]==1)
                keep = (df_raw[label_col]==0) | (df_raw['keep_pos']==True)
                df_raw = df_raw.loc[keep].drop(columns=['keep_pos'])
                self._emit(f"[I] تصفية الهجمات داخل العائلة: استخدمنا {len(ids)} attack_id — الموجبات {npos_before} → {int((df_raw[label_col]==1).sum())}.")

            y_all = df_raw[label_col].astype(int)

            group_col = "scenario_id" if "scenario_id" in df_raw.columns else sender_col
            groups_all = df_raw[group_col]
            self._emit(f"[I] Grouping by: {group_col}")


            if self.split_mode == "holdout_group":
                from sklearn.model_selection import GroupShuffleSplit
                self.stage.emit(f"تقسيم (Group Holdout: test_size={self.test_size:.2f})"); self.progress.emit(20)
                gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=42)
                tr_idx, te_idx = next(gss.split(df_raw, groups=groups_all))
            else:
                from sklearn.model_selection import StratifiedGroupKFold
                n_splits = max(2, int(round(1.0 / max(1e-6, self.test_size))))
                self.stage.emit(f"تقسيم (StratifiedGroupKFold: n_splits={n_splits})"); self.progress.emit(20)
                gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
                tr_idx, te_idx = next(gkf.split(df_raw, y_all, groups_all))

            raw_tr = df_raw.iloc[tr_idx].copy()
            raw_te = df_raw.iloc[te_idx].copy()


            self.stage.emit("هندسة الميزات (TRAIN)"); self.progress.emit(35)
            df_tr = trainer._prepare(raw_tr)
            self.stage.emit("هندسة الميزات (TEST)"); self.progress.emit(45)
            df_te = trainer._prepare(raw_te)

            drop_cols = [sender_col, time_col, label_col, attack_col]
            X_cols_tr = [c for c in df_tr.columns if c not in drop_cols]
            X_cols_te = [c for c in df_te.columns if c not in drop_cols]
            common = sorted(list(set(X_cols_tr) & set(X_cols_te)))
            if len(common) == 0:
                raise RuntimeError("لا توجد أعمدة مشتركة بين TRAIN/TEST بعد التحضير.")
            self._emit(f"[I] تقاطع الميزات: {len(common)} عمود.")

            y_train_bin = df_tr[label_col].astype(int)
            y_test_bin  = df_te[label_col].astype(int)
            groups_train, groups_test = df_tr[sender_col], df_te[sender_col]


            cat_col = 'mb_version' if self.ohe_version and 'mb_version' in common else None
            tfm = _fit_numeric_transformer(df_tr, common, cat_col, self._emit)
            X_train = _apply_numeric_transformer(df_tr, common, tfm)
            X_test  = _apply_numeric_transformer(df_te,  common, tfm)

            def _hash_df(X):

                Xr = X.copy()
                Xr = Xr.astype(float).round(6)
                return pd.util.hash_pandas_object(Xr, index=False).astype(np.uint64)

            h_tr = _hash_df(X_train)
            h_te = _hash_df(X_test)
            inter = np.intersect1d(h_tr.values, h_te.values)
            self._emit(f"[CHECK] train/test duplicate rows (by features) = {len(inter)}")


            from sklearn.metrics import roc_auc_score
            y_perm = y_test_bin.sample(frac=1.0, random_state=7).values
            _auc_dummy = None
            try:

                _auc_dummy = float(roc_auc_score(y_perm, np.random.RandomState(7).rand(len(y_perm))))
            except Exception:
                pass
            self._emit(f"[CHECK] sanity AUC with random scores ≈ {_auc_dummy} (يُفترض ≈ 0.5)")

            X_cols_final = X_train.columns.tolist()
            self._emit(f"[I] شكل الميزات بعد التحويل: {X_train.shape[1]} عمود رقمي (TRAIN-only).")


            from sklearn.preprocessing import StandardScaler
            self.stage.emit("Scaling"); self.progress.emit(55)
            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train); X_test_s = scaler.transform(X_test)


            from sklearn.calibration import CalibratedClassifierCV
            import lightgbm as lgb
            self.stage.emit("تدريب المصنف الثنائي"); self.progress.emit(70)
            base = lgb.LGBMClassifier(
                objective='binary', class_weight='balanced', n_estimators=400,
                learning_rate=0.05, num_leaves=128, min_data_in_leaf=40,
                feature_fraction=0.9, random_state=42, min_gain_to_split=1e-12,
                verbosity=-1
            )
            bin_calib = CalibratedClassifierCV(base, method='isotonic', cv=3)
            bin_calib.fit(X_train_s, y_train_bin)
            p_bin_train = bin_calib.predict_proba(X_train_s)[:, 1]
            p_bin_test  = bin_calib.predict_proba(X_test_s)[:, 1]
            self._emit("[I] تم تدريب/معايرة المصنف الثنائي.")


            fam_probs_train: Dict[str, np.ndarray] = {}
            fam_probs_test:  Dict[str, np.ndarray]  = {}
            fams_to_train = (['pos_speed','replay_stale','dos','sybil','disruptive']
                             if self.family == 'all' else ([] if self.family=='binary' else [self.family]))

            saved_heads = {}
            if 'pos_speed' in fams_to_train:
                self.stage.emit("رأس pos_speed"); self.progress.emit(78)
                head = lgb.LGBMClassifier(objective='binary', class_weight='balanced', n_estimators=400,
                                           learning_rate=0.05, num_leaves=128, min_data_in_leaf=40,
                                           feature_fraction=0.9, random_state=42, min_gain_to_split=1e-12, verbosity=-1)
                from sklearn.calibration import CalibratedClassifierCV as _Cal
                pos_cal = _Cal(head, method='isotonic', cv=3)
                y_fam_tr = trainer._build_family_labels(df_tr, 'pos_speed')
                pos_cal.fit(X_train_s, y_fam_tr)
                fam_probs_train['pos_speed'] = pos_cal.predict_proba(X_train_s)[:,1]
                fam_probs_test['pos_speed']  = pos_cal.predict_proba(X_test_s)[:,1]
                saved_heads['pos_speed'] = pos_cal
                self._emit("[I] pos_speed جاهز.")


            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report, roc_auc_score
            self.stage.emit("دمج الرؤوس (Stacking)"); self.progress.emit(86)
            def stack_build(pbin, fam_probs: Dict[str,np.ndarray]) -> np.ndarray:
                feats = [pbin]
                for k in ["pos_speed","replay_stale","dos","sybil","disruptive","dos_iforest"]:
                    if k in fam_probs: feats.append(fam_probs[k])
                return np.vstack(feats).T
            meta_X_tr = stack_build(p_bin_train, fam_probs_train)
            meta_X_te = stack_build(p_bin_test,  fam_probs_test)
            meta = LogisticRegression(class_weight='balanced', max_iter=300, random_state=42)
            meta.fit(meta_X_tr, y_train_bin)
            p_final = meta.predict_proba(meta_X_te)[:, 1]
            thr = 0.5
            y_pred = (p_final >= thr).astype(int)
            try:
                auc = float(roc_auc_score(y_test_bin, p_final))
            except Exception:
                auc = None

            report = {
                "best_threshold": float(thr),
                "confusion_matrix": confusion_matrix(y_test_bin, y_pred).tolist(),
                "cls_report": classification_report(y_test_bin, y_pred, digits=4, output_dict=True),
                "roc_auc": auc,
            }
            try:
                prec, rec, _ = precision_recall_curve(y_test_bin, p_final)
                report["pr_curve"] = {"precision": prec.tolist(), "recall": rec.tolist()}
            except Exception:
                report["pr_curve"] = {"precision": [], "recall": []}


            if getattr(self, "save_model", False):
                import joblib, datetime as _dt
                ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                base_dir = self.model_dir if self.model_dir else os.path.join(os.getcwd(), "models")
                out_dir = os.path.join(base_dir, self.family, ts)
                os.makedirs(out_dir, exist_ok=True)
                joblib.dump({"tfm": tfm, "scaler": scaler, "X_cols_final": X_cols_final}, os.path.join(out_dir, "preproc.joblib"))
                joblib.dump(bin_calib, os.path.join(out_dir, "bin_calib.joblib"))
                joblib.dump(meta, os.path.join(out_dir, "meta.joblib"))
                for name, model in saved_heads.items():
                    joblib.dump(model, os.path.join(out_dir, f"head_{name}.joblib"))
                with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "family": self.family,
                        "created": ts,
                        "split_mode": self.split_mode,
                        "test_size": self.test_size,
                        "ohe_version": bool(self.ohe_version),
                        "attack_ids_filter": self.attack_ids or [],
                        "features_count": len(X_cols_final),
                    }, f, ensure_ascii=False, indent=2)
                self._emit(f"[✓] تم حفظ النموذج في: {out_dir}")

            self.stage.emit("انتهى التدريب"); self.progress.emit(100)
            self.finished.emit(report)
        except Exception:
            err = traceback.format_exc()
            self.failed.emit(err)


class MetricsTab(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)

        self.lbl_auc = QLabel("ROC AUC: —")
        self.lbl_thr = QLabel("Threshold: 0.5")
        g = QHBoxLayout(); g.addWidget(self.lbl_auc); g.addWidget(self.lbl_thr); g.addStretch()
        v.addLayout(g)

        self.grp_conf = QGroupBox("مصفوفة الالتباس (TN FP / FN TP)")
        gl = QVBoxLayout(self.grp_conf)
        self.tbl_conf = QTableWidget(2, 2)
        self.tbl_conf.setHorizontalHeaderLabels(["Pred 0","Pred 1"])
        self.tbl_conf.setVerticalHeaderLabels(["True 0","True 1"])
        self.tbl_conf.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        gl.addWidget(self.tbl_conf)
        v.addWidget(self.grp_conf)

        self.grp_rep = QGroupBox("تقرير التصنيف (precision/recall/f1)")
        gl2 = QVBoxLayout(self.grp_rep)
        self.tbl_rep = QTableWidget(0, 4)
        self.tbl_rep.setHorizontalHeaderLabels(["label","precision","recall","f1-score"])
        gl2.addWidget(self.tbl_rep)
        v.addWidget(self.grp_rep)

        self.pr_canvas = MplCanvas()
        v.addWidget(QLabel("منحنى Precision-Recall"))
        v.addWidget(self.pr_canvas)

        self.setLayout(v)

    def update_metrics(self, report: Dict[str, Any]):
        auc = report.get("roc_auc", None)
        self.lbl_auc.setText(f"ROC AUC: {auc:.4f}" if auc is not None else "ROC AUC: —")
        thr = report.get("best_threshold", None)
        if thr is not None:
            self.lbl_thr.setText(f"Threshold: {thr:.4f}")

        cm = report.get("confusion_matrix", [[0,0],[0,0]])
        self.tbl_conf.setItem(0,0, QTableWidgetItem(str(cm[0][0])))
        self.tbl_conf.setItem(0,1, QTableWidgetItem(str(cm[0][1])))
        self.tbl_conf.setItem(1,0, QTableWidgetItem(str(cm[1][0])))
        self.tbl_conf.setItem(1,1, QTableWidgetItem(str(cm[1][1])))

        rep = report.get("cls_report", {})
        rows = []
        for lbl, d in rep.items():
            if isinstance(d, dict) and all(k in d for k in ("precision","recall","f1-score")):
                rows.append((lbl, d["precision"], d["recall"], d["f1-score"]))
        self.tbl_rep.setRowCount(len(rows))
        for i,(lbl,p,r,f1) in enumerate(rows):
            self.tbl_rep.setItem(i,0, QTableWidgetItem(str(lbl)))
            self.tbl_rep.setItem(i,1, QTableWidgetItem(f"{p:.4f}"))
            self.tbl_rep.setItem(i,2, QTableWidgetItem(f"{r:.4f}"))
            self.tbl_rep.setItem(i,3, QTableWidgetItem(f"{f1:.4f}"))

        self.pr_canvas.ax.clear()
        pr = report.get("pr_curve", {})
        prec = pr.get("precision", [])
        rec  = pr.get("recall", [])
        if prec and rec:
            self.pr_canvas.ax.plot(rec, prec)
            self.pr_canvas.ax.set_xlabel("Recall")
            self.pr_canvas.ax.set_ylabel("Precision")
            self.pr_canvas.ax.grid(True, alpha=0.3)
        self.pr_canvas.draw()


class TrainTab(QWidget):
    def __init__(self, metrics_tab: MetricsTab):
        super().__init__()
        self.metrics_tab = metrics_tab
        v = QVBoxLayout(self)

        form = QFormLayout()
        self.ed_path = QLineEdit(); self.ed_path.setPlaceholderText("اختر ملف CSV للتدريب…")
        btn_browse = QPushButton("استعراض…")
        btn_browse.clicked.connect(self.browse)
        h = QHBoxLayout(); h.addWidget(self.ed_path); h.addWidget(btn_browse)
        form.addRow("ملف البيانات:", QWidget())
        form.itemAt(form.rowCount()-1, QFormLayout.FieldRole).widget().setLayout(h)

        self.cmb_family = QComboBox()
        self.cmb_family.addItems(["binary","pos_speed","replay_stale","dos","sybil","disruptive","all"])
        form.addRow("عائلة التدريب:", self.cmb_family)

        self.chk_ohe = QCheckBox("ترميز فئوي لعمود mb_version (اختياري)")
        self.chk_ohe.setChecked(True)
        form.addRow("", self.chk_ohe)

        self.chk_save = QCheckBox("حفظ النموذج بعد التدريب")
        self.chk_save.setChecked(True)
        form.addRow("", self.chk_save)

        self.ed_model_dir = QLineEdit(); self.ed_model_dir.setPlaceholderText("مجلّد الحفظ (افتراضي: ./models)")
        btn_dir = QPushButton("اختيار مجلّد…")
        def pick_dir():
            d = QFileDialog.getExistingDirectory(self, "اختر مجلّد الحفظ")
            if d: self.ed_model_dir.setText(d)
        btn_dir.clicked.connect(pick_dir)
        hdir = QHBoxLayout(); hdir.addWidget(self.ed_model_dir); hdir.addWidget(btn_dir)
        form.addRow("مكان الحفظ:", QWidget()); form.itemAt(form.rowCount()-1, QFormLayout.FieldRole).widget().setLayout(hdir)

        self.ed_attacks = QLineEdit(); self.ed_attacks.setPlaceholderText("IDs للهجمات داخل العائلة (مثال: 1,2,3,4,6,7,9) — اختياري")
        form.addRow("تصفية attack_id:", self.ed_attacks)


        self.cmb_split = QComboBox()
        self.cmb_split.addItems(["Group Holdout 80/20", "StratifiedGroupKFold"])
        form.addRow("طريقة التقسيم:", self.cmb_split)

        self.spin_test = QDoubleSpinBox()
        self.spin_test.setDecimals(2)
        self.spin_test.setSingleStep(0.05)
        self.spin_test.setMinimum(0.05)
        self.spin_test.setMaximum(0.5)
        self.spin_test.setValue(0.2)
        form.addRow("نسبة الاختبار:", self.spin_test)

        v.addLayout(form)

        hb = QHBoxLayout()
        self.btn_start = QPushButton("بدء التدريب")
        self.btn_start.clicked.connect(self.start_train)
        hb.addWidget(self.btn_start)
        hb.addStretch()
        v.addLayout(hb)

        self.prog = QProgressBar(); self.prog.setValue(0)
        self.lbl_stage = QLabel("الجاهزية…")
        v.addWidget(self.lbl_stage)
        v.addWidget(self.prog)

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        v.addWidget(self.log)

        self.thread: Optional[QThread] = None
        self.worker: Optional[TrainWorker] = None


    def browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "اختر ملف CSV", "", "CSV Files (*.csv)")
        if path:
            self.ed_path.setText(path)

    def _log_metrics_summary(self, report: Dict[str, Any]):
        def _fmt(x):
            try:
                return f"{float(x):.4f}"
            except Exception:
                return str(x)

        lines = []
        lines.append("== ملخّص المؤشرات ==")

        thr = report.get("best_threshold")
        if thr is not None:
            lines.append(f"Threshold = {thr:.4f}")

        auc = report.get("roc_auc", None)
        if auc is not None:
            lines.append(f"ROC AUC = {auc:.4f}")


        cm = (report.get("confusion_matrix")
              or report.get("confusion_matrix_fixed")
              or report.get("confusion_matrix_adaptive"))
        if cm is not None:
            try:
                tn, fp = cm[0]
                fn, tp = cm[1]
                lines.append("Confusion Matrix [TN FP / FN TP]:")
                lines.append(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            except Exception:
                lines.append(f"Confusion Matrix = {cm}")

        rep = (report.get("cls_report")
               or report.get("cls_report_fixed")
               or report.get("cls_report_adaptive")
               or {})
        for lbl in ["0", "1", "macro avg", "weighted avg"]:
            d = rep.get(lbl)
            if isinstance(d, dict):
                p  = _fmt(d.get("precision")) if d.get("precision") is not None else "—"
                r  = _fmt(d.get("recall"))    if d.get("recall")    is not None else "—"
                f1 = _fmt(d.get("f1-score"))  if d.get("f1-score")  is not None else "—"
                s  = d.get("support")
                s  = int(s) if isinstance(s, (int, float)) else "—"
                lines.append(f"{lbl}: P={p}  R={r}  F1={f1}  (n={s})")

        used = report.get("used_families")
        if used:
            lines.append("الرؤوس المستخدمة: " + ", ".join(used))

        self.log.appendPlainText("\n".join(lines))

    def start_train(self):
        path = self.ed_path.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "تنبيه", "الرجاء اختيار ملف CSV صالح.")
            return
        family = self.cmb_family.currentText()
        ohe = self.chk_ohe.isChecked()
        save = self.chk_save.isChecked()
        model_dir = self.ed_model_dir.text().strip()
        att = self.ed_attacks.text().strip()
        attack_ids = None
        if att:
            try:
                attack_ids = [int(x) for x in att.replace(' ','').split(',') if x!='']
            except Exception:
                QMessageBox.warning(self, "تنبيه", "صيغة attack_id غير صحيحة. استخدم أرقام مفصولة بفواصل.")
                return

        split_mode = "holdout_group" if self.cmb_split.currentIndex()==0 else "sgkfold"
        test_size = float(self.spin_test.value())

        self.btn_start.setEnabled(False)
        self.prog.setValue(0)
        self.lbl_stage.setText("بدء...")
        self.log.clear()

        self.thread = QThread()
        self.worker = TrainWorker(path, family, ohe, save, model_dir, attack_ids, split_mode, test_size)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.on_log)
        self.worker.stage.connect(self.on_stage)
        self.worker.progress.connect(self.prog.setValue)
        self.worker.finished.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.worker.finished.connect(lambda _: self.thread.quit())
        self.worker.failed.connect(lambda _: self.thread.quit())
        self.thread.finished.connect(lambda: self.btn_start.setEnabled(True))
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_log(self, text: str):
        self.log.appendPlainText(text)

    def on_stage(self, s: str):
        self.lbl_stage.setText(s)
        self.log.appendPlainText(f"== {s} ==")

    def on_done(self, report: Dict[str, Any]):


        self._log_metrics_summary(report)

        self.log.appendPlainText("[✓] اكتمل التدريب.")

        self.metrics_tab.update_metrics(report)


    def on_fail(self, err: str):
        self.log.appendPlainText("[X] حدث خطأ أثناء التدريب:\\n" + err)
        QMessageBox.critical(self, "خطأ", "فشل التدريب. تحقق من السجل.")


class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RSU Trainer — واجهة التدريب (All-in-One v7)")
        self.resize(1200, 800)

        tabs = QTabWidget(); tabs.setDocumentMode(True)
        self.metrics_tab = MetricsTab()
        self.train_tab = TrainTab(self.metrics_tab)
        tabs.addTab(self.train_tab, "التدريب")
        tabs.addTab(self.metrics_tab, "المؤشرات")
        self.setCentralWidget(tabs)

        self.statusBar().showMessage("جاهز")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWin()
    w.show()
    sys.exit(app.exec_())
