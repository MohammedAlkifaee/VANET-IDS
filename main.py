#!/usr/bin/env python3
import os
import json
import math
import sys
import time
import numpy as np
import pandas as pd
import joblib
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Set
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QTextEdit, QSplitter, QHeaderView, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem
from PyQt5.QtGui import QBrush, QColor, QPainter, QPixmap, QPen
import pyqtgraph as pg
ROOT_DIR = Path('/home/instantf2md/F2MD/f2md-results/LuSTNanoScenario-ITSG5')
MAP_BACKGROUND_IMAGE = Path('/home/instantf2md/Desktop/VANET-IRAQ Live IDS Dashboard/BG.png')
MODEL_DIR = Path('/home/instantf2md/Desktop/model/')
MAP_ROT_DEG = 0.0
MAP_EXTRA_SCALE = 1.0
MAP_OFFSET_X = 0.0
MAP_OFFSET_Y = 0.0
SCAN_INTERVAL_MS = 500
MAX_MESSAGES = 4000
ANALYTICS_WINDOW_SECS = 60
WINDOW_SIZE = 25
EPS = 1e-06
pg.setConfigOptions(antialias=True)

def ang_norm(d: float) -> float:
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d

def mag(x: float, y: float) -> float:
    return math.hypot(x, y)

def heading_angle(hvec) -> float:
    if not isinstance(hvec, list) or len(hvec) < 2:
        return 0.0
    return math.atan2(hvec[1], hvec[0])

def safe_get(v, i, default=0.0) -> float:
    try:
        return float(v[i])
    except Exception:
        return float(default)

def parse_bsm_file(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    bp = obj.get('BsmPrint', {})
    meta = bp.get('Metadata', {})
    bsms = bp.get('BSMs', []) or []
    if not bsms:
        raise ValueError('no BSMs in file')
    b = bsms[0]
    recv = meta.get('receiverPseudo')
    genT = meta.get('generationTime')
    attack_meta = meta.get('attackType', meta.get('mbType', 'Genuine'))
    sender = b.get('Pseudonym') or b.get('RealId')
    if sender is None or recv is None:
        raise ValueError('missing sender/receiver')
    t = float(b.get('CreationTime', 0.0))
    pos = b.get('Pos', [0, 0, 0])
    x, y = (safe_get(pos, 0), safe_get(pos, 1))
    spd = b.get('Speed', [0, 0, 0])
    vx, vy = (safe_get(spd, 0), safe_get(spd, 1))
    acc = b.get('Accel', [0, 0, 0])
    ax, ay = (safe_get(acc, 0), safe_get(acc, 1))
    hd = b.get('Heading', [1, 0, 0])
    ang = heading_angle(hd)
    pc = b.get('PosConfidence', [0, 0, 0])
    pcx, pcy = (safe_get(pc, 0), safe_get(pc, 1))
    sc = b.get('SpeedConfidence', [0, 0, 0])
    scx, scy = (safe_get(sc, 0), safe_get(sc, 1))
    ac = b.get('AccelConfidence', [0, 0, 0])
    acx, acy = (safe_get(ac, 0), safe_get(ac, 1))
    hc = b.get('HeadingConfidence', [0, 0, 0])
    hcx, hcy = (safe_get(hc, 0), safe_get(hc, 1))
    label = 0 if b.get('AttackType', 'Genuine') == 'Genuine' and (attack_meta or 'Genuine') == 'Genuine' else 1
    rec = dict(file_path=str(path), receiver_pseudo=int(recv), sender_pseudo=int(sender), creation_time=t, x=x, y=y, vx=vx, vy=vy, ax=ax, ay=ay, heading=ang, pos_conf_x=pcx, pos_conf_y=pcy, spd_conf_x=scx, spd_conf_y=scy, acc_conf_x=acx, acc_conf_y=acy, head_conf_x=hcx, head_conf_y=hcy, label=int(label), attack_type=b.get('AttackType', 'Genuine'), meta_attack_type=attack_meta, meta_generation_time=genT if genT is not None else t)
    return rec

class IDSModel:

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.preproc = joblib.load(self.model_dir / 'preproc.joblib')
        self.bin_calib = joblib.load(self.model_dir / 'bin_calib.joblib')
        self.family_head = joblib.load(self.model_dir / 'head_pos_speed.joblib')
        self.meta_model = joblib.load(self.model_dir / 'meta.joblib')
        with open(self.model_dir / 'model_meta.json', 'r', encoding='utf-8') as f:
            self.meta_info = json.load(f)
        self.tfm = self.preproc['tfm']
        self.scaler = self.preproc['scaler']
        self.X_cols = self.preproc['X_cols_final']

    def _apply_numeric_transformer(self, df: pd.DataFrame) -> pd.DataFrame:
        tfm = self.tfm
        df2 = df.copy()
        medians = tfm.get('medians', {})
        for col, med in medians.items():
            if col in df2.columns:
                df2[col] = df2[col].fillna(med)
            else:
                df2[col] = med
        cat_col = tfm.get('cat_col')
        ohe_vals = tfm.get('ohe_values', [])
        if cat_col is not None and ohe_vals:
            if cat_col in df2.columns:
                df2[cat_col] = df2[cat_col].fillna(ohe_vals[0])
            else:
                df2[cat_col] = ohe_vals[0]
        for c in self.X_cols:
            if c not in df2.columns:
                df2[c] = 0.0
        return df2[self.X_cols].astype(float)

    def predict_one(self, feats: Dict[str, float]) -> Dict[str, float]:
        res = self.predict_many([feats])
        return res[0] if res else {}

    def predict_many(self, feats_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if not feats_list:
            return []
        df = pd.DataFrame(feats_list)
        X = self._apply_numeric_transformer(df)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.X_cols)
        p_bin = self.bin_calib.predict_proba(X_scaled)[:, 1]
        p_pos = self.family_head.predict_proba(X_scaled)[:, 1]
        meta_in = np.column_stack([p_bin, p_pos])
        p_final = self.meta_model.predict_proba(meta_in)[:, 1]
        preds = []
        for pf, pb, pp in zip(p_final, p_bin, p_pos):
            preds.append({'p_attack': float(pf), 'p_bin': float(pb), 'p_pos_speed': float(pp), 'pred_label': int(pf >= 0.5)})
        return preds

class CarItem(QGraphicsEllipseItem):

    def __init__(self, veh_id: int, heading: float, color: QColor, dashboard: 'LiveDashboard', selected: bool=False):
        self.base_radius = 14.0 if not selected else 18.0
        super().__init__(-self.base_radius, -self.base_radius, 2 * self.base_radius, 2 * self.base_radius)
        self.veh_id = veh_id
        self.heading = heading
        self.dashboard = dashboard
        self.selected = selected
        self.setBrush(QBrush(color))
        pen = QPen(Qt.black if not selected else QColor('#ffeb3b'))
        pen.setWidth(2 if selected else 1)
        self.setPen(pen)
        self.setZValue(10)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setToolTip(f'Vehicle {veh_id}')

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing, True)
        super().paint(painter, option, widget)
        if self.dashboard and self.dashboard.is_attack_highlight(self.veh_id):
            painter.save()
            halo_pen = QPen(QColor(198, 40, 40, 180))
            halo_pen.setWidth(6)
            painter.setPen(halo_pen)
            painter.setBrush(Qt.NoBrush)
            r = self.base_radius + 6
            painter.drawEllipse(QRectF(-r, -r, 2 * r, 2 * r))
            painter.restore()
        painter.save()
        painter.setPen(QPen(Qt.black, 2))
        painter.rotate(-math.degrees(self.heading))
        r = self.base_radius
        painter.drawLine(0, 0, 0, -r - 4)
        painter.restore()
        rect = self.boundingRect().adjusted(0, -16, 0, 0)
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(Qt.black)
        painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, str(self.veh_id))

    def mousePressEvent(self, event):
        if self.dashboard is not None:
            self.dashboard.on_car_clicked(self.veh_id)
        super().mousePressEvent(event)

class LiveDashboard(QMainWindow):

    def __init__(self, root_dir: Path):
        super().__init__()
        self.root_dir = root_dir
        self.setWindowTitle('VANET LuST – Live BSM Dashboard')
        self.resize(1600, 900)
        self.seen_files: Set[Path] = set()
        self.vehicles: Dict[int, Dict[str, Any]] = {}
        self.messages: List[Dict[str, Any]] = []
        self.current_vehicle_filter: int = None
        self.map_items: Dict[int, CarItem] = {}
        self.selected_vehicle_id: int = None
        self.window_size = WINDOW_SIZE
        self.attack_highlights: Dict[int, float] = {}
        self.pending_center_vehicle_id: int = None
        self.sender_history: Dict[int, deque] = {}
        self.sender_rate_state: Dict[int, Dict[str, float]] = {}
        self.sender_state_code_map: Dict[int, Dict[Any, int]] = {}
        self.time_base: float = None
        self.window_sender_counts: Dict[int, Dict[int, int]] = {}
        self.window_sender_sets: Dict[int, Set[int]] = {}
        self.window_total_msgs: Dict[int, int] = {}
        self.window_new_msgs: Dict[int, int] = {}
        self.sender_first_window: Dict[int, int] = {}
        self.sybil_rate_ewma: float = None
        self.ids_model: IDSModel = None
        self.global_min_x = None
        self.global_max_x = None
        self.global_min_y = None
        self.global_max_y = None
        self.per_sec_total: Dict[int, int] = {}
        self.per_sec_attack: Dict[int, int] = {}
        self.per_sec_vehicle_ids: Dict[int, Set[int]] = {}
        self.per_sec_sum_speed: Dict[int, float] = {}
        self.per_sec_count_speed: Dict[int, int] = {}
        self.per_sec_new_veh: Dict[int, int] = {}
        self.attack_type_counts: Dict[str, int] = {}
        self.seen_vehicle_ids: Set[int] = set()
        self.decision_threshold: float = 0.5
        self.perf_history: List[Any] = []
        self.perf_history_maxlen: int = 50000
        if MAP_BACKGROUND_IMAGE.is_file():
            self.map_background = QPixmap(str(MAP_BACKGROUND_IMAGE))
        else:
            self.map_background = None
        self._setup_ui()
        self._load_ids_model()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.scan_for_new_bsms)
        self.timer.start(SCAN_INTERVAL_MS)

    def _setup_ui(self):
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        tab_live = QWidget()
        v_layout = QVBoxLayout(tab_live)
        splitter = QSplitter(Qt.Horizontal)
        self.tbl_vehicles = QTableWidget()
        self.tbl_vehicles.setColumnCount(9)
        self.tbl_vehicles.setHorizontalHeaderLabels(['SenderPseudo', 'Last time', 'Speed', 'X', 'Y', 'Label (GT)', 'Last AttackType', 'Pred label', 'Pred p_attack'])
        self.tbl_vehicles.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_vehicles.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_vehicles.setSelectionMode(QTableWidget.SingleSelection)
        self.tbl_vehicles.setAlternatingRowColors(True)
        self.tbl_vehicles.itemSelectionChanged.connect(self.on_vehicle_selected)
        self.tbl_messages = QTableWidget()
        self.tbl_messages.setColumnCount(11)
        self.tbl_messages.setHorizontalHeaderLabels(['Time', 'Receiver', 'Sender', 'X', 'Y', 'Speed', 'Label (GT)', 'AttackType', 'File', 'Pred label', 'Pred p_attack'])
        self.tbl_messages.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_messages.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_messages.setAlternatingRowColors(True)
        splitter.addWidget(self.tbl_vehicles)
        splitter.addWidget(self.tbl_messages)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        v_layout.addWidget(splitter)
        tabs.addTab(tab_live, 'Live – Tables')
        tab_map = QWidget()
        map_layout = QVBoxLayout(tab_map)
        stats_layout = QHBoxLayout()
        self.lbl_total_veh = QLabel('Vehicles: 0')
        self.lbl_attack_veh = QLabel('Attack vehicles: 0')
        self.lbl_benign_veh = QLabel('Benign vehicles: 0')
        self.lbl_total_msgs = QLabel('Messages: 0')
        self.lbl_attack_msgs = QLabel('Attack msgs: 0')
        for lbl in (self.lbl_total_veh, self.lbl_attack_veh, self.lbl_benign_veh, self.lbl_total_msgs, self.lbl_attack_msgs):
            stats_layout.addWidget(lbl)
        stats_layout.addStretch()
        map_layout.addLayout(stats_layout)
        self.map_scene = QGraphicsScene(self)
        self.map_view = QGraphicsView(self.map_scene)
        self.map_view.setRenderHint(QPainter.Antialiasing, True)
        self.map_view.setDragMode(QGraphicsView.ScrollHandDrag)
        map_layout.addWidget(self.map_view)
        self.sel_title = QLabel('Selected vehicle: none')
        self.sel_title.setStyleSheet('font-weight: bold;')
        self.sel_basic = QLabel('Time: -   Speed: -   Label: -   Attack: -')
        self.sel_pos = QLabel('Pos: -   Heading: -   Receiver: -')
        self.sel_conf = QLabel('Conf: pos=( -, - )  spd=( -, - )  acc=( -, - )  head=( -, - )')
        for lbl in (self.sel_title, self.sel_basic, self.sel_pos, self.sel_conf):
            lbl.setStyleSheet('font-size: 11px;')
        map_layout.addWidget(self.sel_title)
        map_layout.addWidget(self.sel_basic)
        map_layout.addWidget(self.sel_pos)
        map_layout.addWidget(self.sel_conf)
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(self._make_color_legend('#2e7d32', 'Benign'))
        legend_layout.addWidget(self._make_color_legend('#c62828', 'Attack'))
        legend_layout.addStretch()
        map_layout.addLayout(legend_layout)
        tabs.addTab(tab_map, 'Map')
        tab_analytics = QWidget()
        anal_layout = QVBoxLayout(tab_analytics)
        self.analytics_graph = pg.GraphicsLayoutWidget()
        anal_layout.addWidget(self.analytics_graph)
        self.plot_msg_rate = self.analytics_graph.addPlot(title='Messages per second')
        self.plot_msg_rate.setLabel('bottom', 'Sim time (s)')
        self.plot_msg_rate.setLabel('left', '#Msg/s')
        self.plot_msg_rate.showGrid(x=True, y=True, alpha=0.3)
        self.cur_msg_rate = self.plot_msg_rate.plot(pen=pg.mkPen('#90caf9', width=2))
        self.analytics_graph.nextColumn()
        self.plot_veh_rate = self.analytics_graph.addPlot(title='Vehicles per second')
        self.plot_veh_rate.setLabel('bottom', 'Sim time (s)')
        self.plot_veh_rate.setLabel('left', '#Veh/s')
        self.plot_veh_rate.showGrid(x=True, y=True, alpha=0.3)
        self.cur_veh_rate = self.plot_veh_rate.plot(pen=pg.mkPen('#81c784', width=2))
        self.analytics_graph.nextColumn()
        self.plot_attack_rate = self.analytics_graph.addPlot(title='Attack messages per second')
        self.plot_attack_rate.setLabel('bottom', 'Sim time (s)')
        self.plot_attack_rate.setLabel('left', '#Atk/s')
        self.plot_attack_rate.showGrid(x=True, y=True, alpha=0.3)
        self.cur_attack_rate = self.plot_attack_rate.plot(pen=pg.mkPen('#ef5350', width=2))
        self.analytics_graph.nextRow()
        self.plot_cum_msg = self.analytics_graph.addPlot(title='Cumulative messages')
        self.plot_cum_msg.setLabel('bottom', 'Sim time (s)')
        self.plot_cum_msg.setLabel('left', 'Total msgs')
        self.plot_cum_msg.showGrid(x=True, y=True, alpha=0.3)
        self.cur_cum_msg = self.plot_cum_msg.plot(pen=pg.mkPen('#42a5f5', width=2))
        self.analytics_graph.nextColumn()
        self.plot_cum_attack = self.analytics_graph.addPlot(title='Cumulative attacks')
        self.plot_cum_attack.setLabel('bottom', 'Sim time (s)')
        self.plot_cum_attack.setLabel('left', 'Total attacks')
        self.plot_cum_attack.showGrid(x=True, y=True, alpha=0.3)
        self.cur_cum_attack = self.plot_cum_attack.plot(pen=pg.mkPen('#e53935', width=2))
        self.analytics_graph.nextColumn()
        self.plot_cum_veh = self.analytics_graph.addPlot(title='Cumulative vehicles')
        self.plot_cum_veh.setLabel('bottom', 'Sim time (s)')
        self.plot_cum_veh.setLabel('left', 'Vehicles')
        self.plot_cum_veh.showGrid(x=True, y=True, alpha=0.3)
        self.cur_cum_veh = self.plot_cum_veh.plot(pen=pg.mkPen('#66bb6a', width=2))
        self.analytics_graph.nextRow()
        self.plot_attack_ratio = self.analytics_graph.addPlot(title='Attack ratio')
        self.plot_attack_ratio.setLabel('bottom', 'Sim time (s)')
        self.plot_attack_ratio.setLabel('left', 'Ratio')
        self.plot_attack_ratio.showGrid(x=True, y=True, alpha=0.3)
        self.cur_attack_ratio = self.plot_attack_ratio.plot(pen=pg.mkPen('#ffca28', width=2))
        self.analytics_graph.nextColumn()
        self.plot_avg_speed = self.analytics_graph.addPlot(title='Average speed')
        self.plot_avg_speed.setLabel('bottom', 'Sim time (s)')
        self.plot_avg_speed.setLabel('left', 'Speed')
        self.plot_avg_speed.showGrid(x=True, y=True, alpha=0.3)
        self.cur_avg_speed = self.plot_avg_speed.plot(pen=pg.mkPen('#ab47bc', width=2))
        self.analytics_graph.nextColumn()
        self.plot_attack_types = self.analytics_graph.addPlot(title='Attack types (cumulative)')
        self.plot_attack_types.setLabel('bottom', 'Attack type')
        self.plot_attack_types.setLabel('left', 'Count')
        self.plot_attack_types.showGrid(x=True, y=True, alpha=0.3)
        self.attack_bar_item = None
        tabs.addTab(tab_analytics, 'Analytics')
        tab_perf = QWidget()
        perf_layout = QVBoxLayout(tab_perf)
        self.perf_lbl_title = QLabel('Detection performance (BSM-level)')
        self.perf_lbl_title.setStyleSheet('font-weight: bold; font-size: 12px;')
        self.perf_lbl_sub = QLabel('Cumulative since first message (all processed, not limited by table) + last 500 messages')
        self.perf_lbl_sub.setStyleSheet('font-size: 11px; color: #555;')
        self.perf_lbl_thresh = QLabel('Decision threshold: 0.50 (auto)')
        self.perf_lbl_thresh.setStyleSheet('font-size: 11px; color: #333;')
        self.perf_lbl_counts = QLabel('Evaluated msgs: 0 | TP: 0 | FP: 0 | FN: 0 | TN: 0')
        self.perf_lbl_metrics = QLabel('Accuracy: - | Precision: - | Recall: - | F1: -')
        self.perf_lbl_recent = QLabel('Last 500 msgs -> Accuracy: - | Precision: - | Recall: - | F1: -')
        self.perf_lbl_rates = QLabel('Rates: FPR: - | FNR: - | Pred attack ratio: -')
        self.perf_lbl_probs = QLabel('Scores mean (overall/last500) -> attack: -/- | benign: -/- | overall: -/-')
        for lbl in (self.perf_lbl_thresh, self.perf_lbl_counts, self.perf_lbl_metrics, self.perf_lbl_recent, self.perf_lbl_rates, self.perf_lbl_probs):
            lbl.setStyleSheet('font-size: 11px;')
        perf_layout.addWidget(self.perf_lbl_title)
        perf_layout.addWidget(self.perf_lbl_sub)
        perf_layout.addWidget(self.perf_lbl_thresh)
        perf_layout.addWidget(self.perf_lbl_counts)
        perf_layout.addWidget(self.perf_lbl_metrics)
        perf_layout.addWidget(self.perf_lbl_recent)
        perf_layout.addWidget(self.perf_lbl_rates)
        perf_layout.addWidget(self.perf_lbl_probs)
        perf_layout.addStretch()
        tabs.addTab(tab_perf, 'Performance')
        tab_log = QWidget()
        log_layout = QVBoxLayout(tab_log)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(QLabel('Event log'))
        log_layout.addWidget(self.log_view)
        tabs.addTab(tab_log, 'Log')

    def _load_ids_model(self):
        try:
            self.ids_model = IDSModel(MODEL_DIR)
            self.append_log('<b>[IDS]</b> model loaded successfully.')
        except Exception as e:
            self.ids_model = None
            self.append_log(f"<span style='color:orange'>[IDS] failed to load model: {e}</span>")

    def _make_color_legend(self, color_hex: str, text: str) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        swatch = QLabel()
        swatch.setFixedSize(16, 16)
        swatch.setStyleSheet(f'background-color: {color_hex}; border-radius: 3px;')
        label = QLabel(text)
        layout.addWidget(swatch)
        layout.addWidget(label)
        layout.setContentsMargins(0, 0, 0, 0)
        return w

    def on_vehicle_selected(self):
        sel = self.tbl_vehicles.selectedItems()
        if not sel:
            self.current_vehicle_filter = None
            self.update_messages_table()
            return
        sender_item = sel[0]
        try:
            sender = int(sender_item.text())
        except ValueError:
            sender = None
        self.current_vehicle_filter = sender
        self.update_messages_table()

    def on_car_clicked(self, veh_id: int):
        self.selected_vehicle_id = veh_id
        self.update_map_view()
        for row in range(self.tbl_vehicles.rowCount()):
            item = self.tbl_vehicles.item(row, 0)
            if item is not None:
                try:
                    if int(item.text()) == veh_id:
                        self.tbl_vehicles.selectRow(row)
                        break
                except ValueError:
                    continue
        veh = self.vehicles.get(veh_id)
        if veh:
            r = veh.get('last_rec', {})
            self.sel_title.setText(f'Selected vehicle: {veh_id}')
            self.sel_basic.setText(f"Time={veh['last_time']:.3f} s   Speed={veh['speed']:.2f} m/s   Label={veh['label']}   Attack={veh['attack_type']}")
            self.sel_pos.setText(f"Pos=({veh['x']:.1f}, {veh['y']:.1f})   Heading={veh['heading']:.2f} rad   Receiver={r.get('receiver_pseudo', '-')}")
            self.sel_conf.setText('Conf: pos=({:.3f}, {:.3f})  spd=({:.3f}, {:.3f})  acc=({:.3f}, {:.3f})  head=({:.3f}, {:.3f})'.format(r.get('pos_conf_x', 0.0), r.get('pos_conf_y', 0.0), r.get('spd_conf_x', 0.0), r.get('spd_conf_y', 0.0), r.get('acc_conf_x', 0.0), r.get('acc_conf_y', 0.0), r.get('head_conf_x', 0.0), r.get('head_conf_y', 0.0)))
            self.append_log(f"<b>Vehicle {veh_id} selected</b>: time={veh['last_time']:.3f}, speed={veh['speed']:.1f}, label={veh['label']}, attack={veh['attack_type']}")
            item = self.map_items.get(veh_id)
            if item:
                self.map_view.centerOn(item)
        if self.pending_center_vehicle_id == veh_id:
            self.pending_center_vehicle_id = None

    def is_attack_highlight(self, veh_id: int) -> bool:
        expiry = self.attack_highlights.get(veh_id)
        if expiry is None:
            return False
        if time.time() > expiry:
            self.attack_highlights.pop(veh_id, None)
            if self.pending_center_vehicle_id == veh_id:
                self.pending_center_vehicle_id = None
            return False
        return True

    def scan_for_new_bsms(self):
        bsm_dirs = sorted(self.root_dir.glob('MDBsms_V2_*'))
        if not bsm_dirs:
            return
        new_data: List[Any] = []
        for d in bsm_dirs:
            for f in sorted(d.glob('*.bsm')):
                if f in self.seen_files:
                    continue
                try:
                    rec = parse_bsm_file(f)
                except Exception as e:
                    self.append_log(f"<span style='color:orange'>[!] Error parsing {f.name}: {e}</span>")
                    self.seen_files.add(f)
                    continue
                self.seen_files.add(f)
                feats = None
                if self.ids_model is not None:
                    try:
                        feats = self._build_feature_row(rec)
                    except Exception as e:
                        self.append_log(f"<span style='color:orange'>[IDS] feature build error for sender {rec.get('sender_pseudo', '-')}: {e}</span>")
                new_data.append((rec, feats))
        if not new_data:
            return
        det_results: List[Dict[str, Any]] = [{} for _ in new_data]
        if self.ids_model is not None:
            feats_only = [ft for _, ft in new_data if ft is not None]
            if feats_only:
                try:
                    preds = self.ids_model.predict_many(feats_only)
                    preds_iter = iter(preds)
                    for i, (_rec, ft) in enumerate(new_data):
                        if ft is not None:
                            det_results[i] = next(preds_iter, {})
                except Exception as e:
                    self.append_log(f"<span style='color:orange'>[IDS] batch prediction error: {e}</span>")
        for (rec, _ft), det in zip(new_data, det_results):
            self.handle_new_record(rec, det)
        self.update_vehicle_table()
        self.update_messages_table()
        self.update_stats_labels()
        self.update_map_view()
        self.update_analytics_charts()
        self.update_performance_tab()

    def _normalize_time_value(self, t_raw: Any) -> float:
        try:
            t = float(t_raw)
        except Exception:
            return 0.0
        if t > 1000000000000000.0:
            return t / 1000000000.0
        if t > 1000000000000.0:
            return t / 1000000.0
        if t > 1000000000.0:
            return t / 1000.0
        return t

    def _build_feature_row(self, rec: Dict[str, Any]) -> Dict[str, float]:
        sender = rec['sender_pseudo']
        t_norm = self._normalize_time_value(rec.get('creation_time', 0.0))
        gen_t_norm = self._normalize_time_value(rec.get('meta_generation_time', t_norm))
        t = t_norm
        gen_t = gen_t_norm
        x = float(rec.get('x', 0.0))
        y = float(rec.get('y', 0.0))
        vx = float(rec.get('vx', 0.0))
        vy = float(rec.get('vy', 0.0))
        heading = float(rec.get('heading', 0.0))
        speed_curr = mag(vx, vy)
        hist = self.sender_history.get(sender)
        if hist is None:
            hist = deque(maxlen=self.window_size)
            self.sender_history[sender] = hist
        prev = hist[-1] if hist else None
        t_prev = prev['time'] if prev else t
        dt = max(t - t_prev, 0.0)
        dt_safe = dt if dt > EPS else EPS
        x_prev = prev['x'] if prev else x
        y_prev = prev['y'] if prev else y
        speed_prev = prev['speed_curr'] if prev else speed_curr
        heading_prev = prev['heading'] if prev else heading
        acc_prev = prev['acc_curr'] if prev else 0.0
        dx = x - x_prev
        dy = y - y_prev
        dist = mag(dx, dy)
        dv = speed_curr - speed_prev
        acc_curr = dv / dt_safe
        dacc = acc_curr - acc_prev
        dacc_jerk = dacc / dt_safe
        jerk = dacc / dt_safe
        dtheta = heading - heading_prev
        heading_rate = dtheta / dt_safe
        x_pred = x_prev + speed_prev * math.cos(heading_prev) * dt
        y_pred = y_prev + speed_prev * math.sin(heading_prev) * dt
        dr_dx = x - x_pred
        dr_dy = y - y_pred
        dr_angle = math.atan2(dr_dy, dr_dx)
        sin_a = math.sin(dr_angle)
        cos_a = math.cos(dr_angle)
        consistency_err = abs(dist - speed_curr * dt)
        flag_consistency = 1.0 if consistency_err > max(1.0, 0.25 * speed_curr * dt_safe) else 0.0
        state_hash = (round(x, 3), round(y, 3), round(speed_curr, 2), round(heading, 2))
        code_map = self.sender_state_code_map.setdefault(sender, {})
        if state_hash in code_map:
            state_code = code_map[state_hash]
        else:
            state_code = len(code_map)
            code_map[state_hash] = state_code
        hist_with_curr = list(hist) + [{'dt': dt, 'dv': dv, 'heading_rate': heading_rate, 'dist': dist, 'consistency_err': consistency_err, 'dacc_jerk': dacc_jerk, 'dr_angle': dr_angle, 'acc_curr': acc_curr, 'speed_curr': speed_curr, 'state_code': state_code}]

        def stats(key: str):
            vals = [float(h.get(key, 0.0)) for h in hist_with_curr]
            if not vals:
                return (0.0, 0.0, 0.0)
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                var = sum(((v - mean) ** 2 for v in vals)) / (len(vals) - 1)
                std = math.sqrt(max(var, 0.0))
            else:
                std = 0.0
            return (mean, std, max(vals))
        dist_mean, dist_std, dist_max = stats('dist')
        ce_mean, _, _ = stats('consistency_err')
        jerk_mean, jerk_std, jerk_max = stats('dacc_jerk')
        dt_mean, dt_std, dt_max = stats('dt')
        dv_mean, dv_std, dv_max = stats('dv')
        hr_mean, hr_std, hr_max = stats('heading_rate')
        dt_cv = dt_std / dt_mean if dt_mean > 0 else 0.0
        dt_vals = [float(h.get('dt', 0.0)) for h in hist_with_curr]
        if dt_vals:
            dt_med = float(np.median(dt_vals))
            dt_mad = float(np.median(np.abs(np.array(dt_vals) - dt_med)) + EPS)
            dt_z = (dt - dt_med) / dt_mad
        else:
            dt_med = 0.0
            dt_mad = EPS
            dt_z = 0.0
        sin_list = [math.sin(float(h.get('dr_angle', 0.0))) for h in hist_with_curr]
        cos_list = [math.cos(float(h.get('dr_angle', 0.0))) for h in hist_with_curr]
        if sin_list:
            sin_mean = sum(sin_list) / len(sin_list)
            cos_mean = sum(cos_list) / len(cos_list)
            dr_angle_var = 1.0 - math.sqrt(cos_mean ** 2 + sin_mean ** 2)
        else:
            dr_angle_var = 0.0

        def ratio(vals, predicate):
            if not vals:
                return 0.0
            return sum((1 for v in vals if predicate(v))) / len(vals)
        freeze_ratio_dist = ratio([h.get('dist', 0.0) for h in hist_with_curr], lambda v: abs(v) < 0.001)
        freeze_ratio_dv = ratio([h.get('dv', 0.0) for h in hist_with_curr], lambda v: abs(v) < 0.0001)
        freeze_ratio_hr = ratio([h.get('heading_rate', 0.0) for h in hist_with_curr], lambda v: abs(v) < 0.0001)
        low_speed_flag = 1.0 if speed_curr < 0.5 else 0.0
        low_speed_ratio = ratio([h.get('speed_curr', 0.0) for h in hist_with_curr], lambda v: v < 0.5)
        neg_acc_flag = 1.0 if acc_curr < -0.3 else 0.0
        neg_acc_ratio = ratio([h.get('acc_curr', 0.0) for h in hist_with_curr], lambda v: v < -0.3)
        eff = self.window_size - 1
        times = [float(h.get('time', 0.0)) for h in hist] + [t]
        if len(times) > 1:
            idx_prev = max(0, len(times) - 1 - eff)
            span = t - times[idx_prev]
            span = span if span > EPS else EPS
            rate = min(eff, len(times) - 1) / span
        else:
            rate = 0.0
        rate_state = self.sender_rate_state.get(sender, {'ewma': rate, 'cumsum_pos': 0.0, 'cumsum_neg': 0.0, 'cusum_pos': 0.0, 'cusum_neg': 0.0})
        alpha_rate = 2.0 / (self.window_size + 1)
        ewma = alpha_rate * rate + (1 - alpha_rate) * rate_state.get('ewma', rate)
        r = rate - ewma
        cumsum_pos = max(0.0, rate_state.get('cumsum_pos', 0.0) + r)
        cumsum_neg = max(0.0, rate_state.get('cumsum_neg', 0.0) - r)
        cusum_pos = max(rate_state.get('cusum_pos', 0.0), cumsum_pos)
        cusum_neg = max(rate_state.get('cusum_neg', 0.0), cumsum_neg)
        self.sender_rate_state[sender] = {'ewma': ewma, 'cumsum_pos': cumsum_pos, 'cumsum_neg': cumsum_neg, 'cusum_pos': cusum_pos, 'cusum_neg': cusum_neg}
        state_codes = [int(h.get('state_code', -1)) for h in hist] + [int(state_code)]
        if state_codes:
            state_dup_ratio = 1.0 - len(set(state_codes)) / len(state_codes)
        else:
            state_dup_ratio = 0.0
        window_id = int(gen_t // 5)
        if sender not in self.sender_first_window:
            self.sender_first_window[sender] = window_id
        is_new_id = 1 if window_id == self.sender_first_window.get(sender, window_id) else 0
        self.window_total_msgs[window_id] = self.window_total_msgs.get(window_id, 0) + 1
        if is_new_id:
            self.window_new_msgs[window_id] = self.window_new_msgs.get(window_id, 0) + 1
        counts = self.window_sender_counts.setdefault(window_id, {})
        counts[sender] = counts.get(sender, 0) + 1
        self.window_sender_counts[window_id] = counts
        cur_set = set(counts.keys())
        self.window_sender_sets[window_id] = cur_set
        prev_set = self.window_sender_sets.get(window_id - 1, set())
        union = cur_set | prev_set
        inter = cur_set & prev_set
        jaccard = len(inter) / len(union) if union else 0.0
        total_w = self.window_total_msgs.get(window_id, 1)
        new_w = self.window_new_msgs.get(window_id, 0)
        sybil_new_rate = new_w / total_w
        prev_ewma = self.sybil_rate_ewma if self.sybil_rate_ewma is not None else sybil_new_rate
        alpha_sybil = 2.0 / (8 + 1)
        self.sybil_rate_ewma = alpha_sybil * sybil_new_rate + (1 - alpha_sybil) * prev_ewma
        sybil_burst = sybil_new_rate - prev_ewma
        probs = []
        total_counts = sum(counts.values())
        if total_counts > 0:
            for c in counts.values():
                p = c / total_counts
                probs.append(p)
        entropy = -sum((p * math.log(p + 1e-09) for p in probs)) if probs else 0.0
        hist.append({'time': t, 'x': x, 'y': y, 'speed_curr': speed_curr, 'acc_curr': acc_curr, 'heading': heading, 'dt': dt, 'dv': dv, 'dist': dist, 'dacc_jerk': dacc_jerk, 'heading_rate': heading_rate, 'consistency_err': consistency_err, 'dr_angle': dr_angle, 'state_code': state_code})
        feats: Dict[str, float] = {}
        feats['acc_conf_x_curr'] = float(rec.get('acc_conf_x', 0.0))
        feats['acc_conf_y_curr'] = float(rec.get('acc_conf_y', 0.0))
        feats['acc_curr'] = acc_curr
        feats['acc_prev'] = acc_prev
        feats['consistency_err'] = consistency_err
        feats['consistency_err_mean_w25'] = ce_mean
        feats['cos_a'] = cos_a
        feats['dacc'] = dacc
        feats['dacc_jerk'] = dacc_jerk
        feats['dacc_jerk_max_w25'] = jerk_max
        feats['dacc_jerk_mean_w25'] = jerk_mean
        feats['dacc_jerk_std_w25'] = jerk_std
        feats['dist'] = dist
        feats['dist_max_w25'] = dist_max
        feats['dist_mean_w25'] = dist_mean
        feats['dist_std_w25'] = dist_std
        feats['dr_angle'] = dr_angle
        feats['dr_angle_var_w25'] = dr_angle_var
        feats['dr_dx'] = dr_dx
        feats['dr_dy'] = dr_dy
        feats['dt'] = dt
        feats['dt_cv_w'] = dt_cv
        feats['dt_jitter_w25'] = dt_std
        feats['dt_max_w25'] = dt_max
        feats['dt_mean_w25'] = dt_mean
        feats['dt_std_w25'] = dt_std
        feats['dt_z'] = dt_z
        feats['dtheta'] = dtheta
        feats['dv'] = dv
        feats['dv_max_w25'] = dv_max
        feats['dv_mean_w25'] = dv_mean
        feats['dv_std_w25'] = dv_std
        feats['dx'] = dx
        feats['dy'] = dy
        feats['flag_consistency'] = flag_consistency
        feats['freeze_ratio_dist_w25'] = freeze_ratio_dist
        feats['freeze_ratio_dv_w25'] = freeze_ratio_dv
        feats['freeze_ratio_hr_w25'] = freeze_ratio_hr
        feats['head_conf_x_curr'] = float(rec.get('head_conf_x', 0.0))
        feats['head_conf_y_curr'] = float(rec.get('head_conf_y', 0.0))
        feats['heading_curr'] = heading
        feats['heading_prev'] = heading_prev
        feats['heading_rate'] = heading_rate
        feats['heading_rate_max_w25'] = hr_max
        feats['heading_rate_mean_w25'] = hr_mean
        feats['heading_rate_std_w25'] = hr_std
        feats['jerk'] = jerk
        feats['low_speed_flag'] = low_speed_flag
        feats['low_speed_ratio_w25'] = low_speed_ratio
        feats['neg_acc_flag'] = neg_acc_flag
        feats['neg_acc_ratio_w25'] = neg_acc_ratio
        feats['pos_conf_x_curr'] = float(rec.get('pos_conf_x', 0.0))
        feats['pos_conf_y_curr'] = float(rec.get('pos_conf_y', 0.0))
        feats['rate_cusum_neg'] = cusum_neg
        feats['rate_cusum_pos'] = cusum_pos
        feats['rate_ewma'] = ewma
        feats['rate_msgs_per_s'] = rate
        feats['receiver_pseudo'] = float(rec.get('receiver_pseudo', 0.0))
        feats['sin_a'] = sin_a
        feats['spd_conf_x_curr'] = float(rec.get('spd_conf_x', 0.0))
        feats['spd_conf_y_curr'] = float(rec.get('spd_conf_y', 0.0))
        feats['speed_curr'] = speed_curr
        feats['speed_prev'] = speed_prev
        feats['state_code'] = float(state_code)
        feats['state_dup_ratio_w'] = state_dup_ratio
        feats['sybil_jaccard_ids_5s'] = jaccard
        feats['sybil_new_ids_burst'] = sybil_burst
        feats['sybil_new_ids_rate'] = sybil_new_rate
        feats['sybil_sender_entropy_5s'] = entropy
        feats['sybil_unique_ids_5s'] = float(len(cur_set))
        feats['t_prev'] = t_prev
        feats['window_id'] = window_id
        feats['x_curr'] = x
        feats['x_prev'] = x_prev
        feats['y_curr'] = y
        feats['y_prev'] = y_prev
        return feats

    def _detect_with_model(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        if self.ids_model is None:
            return {}
        try:
            feats = self._build_feature_row(rec)
            det = self.ids_model.predict_one(feats)
            return det
        except Exception as e:
            self.append_log(f"<span style='color:orange'>[IDS] prediction error for sender {rec.get('sender_pseudo', '-')}: {e}</span>")
            return {}

    def _update_cumulative_performance(self, msg: Dict[str, Any]):
        score = msg.get('det_score')
        if score is None:
            return
        try:
            s_val = float(score)
            gt = int(msg.get('label'))
        except Exception:
            return
        self.perf_history.append((s_val, gt))
        if len(self.perf_history) > self.perf_history_maxlen:
            self.perf_history = self.perf_history[-self.perf_history_maxlen:]
        self._maybe_update_threshold()

    def _maybe_update_threshold(self):
        if len(self.perf_history) < 300:
            return
        scores = np.array([s for s, _ in self.perf_history], dtype=float)
        labels = np.array([l for _, l in self.perf_history], dtype=int)
        qs = np.linspace(0.05, 0.95, 19)
        ths = np.unique(np.concatenate([np.array([0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], dtype=float), np.quantile(scores, qs)]))
        best_th = self.decision_threshold
        best_f1 = -1.0
        min_precision = 0.7
        for th in ths:
            pred = scores >= th
            tp = int(((pred == 1) & (labels == 1)).sum())
            fp = int(((pred == 1) & (labels == 0)).sum())
            fn = int(((pred == 0) & (labels == 1)).sum())
            if tp + fp == 0 or tp + fn == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            if prec < min_precision:
                continue
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            if f1 > best_f1 + 1e-06:
                best_f1 = f1
                best_th = float(th)
        if best_f1 > -0.5 and abs(best_th - self.decision_threshold) > 0.0001:
            self.decision_threshold = best_th
            self._apply_threshold_to_state()

    def _apply_threshold_to_state(self):
        for m in self.messages:
            score = m.get('det_score')
            if score is None:
                continue
            m['det_label'] = 1 if score >= self.decision_threshold else 0
        for vid, v in self.vehicles.items():
            score = v.get('det_score')
            if score is None:
                continue
            v['det_label'] = 1 if score >= self.decision_threshold else 0
        self.update_vehicle_table()
        self.update_messages_table()

    def trigger_attack_alert(self, sender: int, x: float, y: float, score: float=None):
        now = time.time()
        self.attack_highlights[sender] = now + 5.0
        self.pending_center_vehicle_id = sender
        self.selected_vehicle_id = sender
        msg = f'[ALERT] Attack detected: sender={sender}, pos=({x:.1f},{y:.1f})'
        if score is not None:
            msg += f', score={score:.3f}'
        self.append_log(f"<span style='color:red'><b>{msg}</b></span>")
        self.statusBar().showMessage(msg, 5000)

    def handle_new_record(self, rec: Dict[str, Any], det_result: Dict[str, Any]=None):
        sender = rec['sender_pseudo']
        t = rec['creation_time']
        sec = int(t)
        sp = mag(rec['vx'], rec['vy'])
        x, y = (rec['x'], rec['y'])
        if self.global_min_x is None:
            self.global_min_x = self.global_max_x = x
            self.global_min_y = self.global_max_y = y
        else:
            self.global_min_x = min(self.global_min_x, x)
            self.global_max_x = max(self.global_max_x, x)
            self.global_min_y = min(self.global_min_y, y)
            self.global_max_y = max(self.global_max_y, y)
        veh = self.vehicles.get(sender, {'sender_pseudo': sender, 'last_time': t, 'x': rec['x'], 'y': rec['y'], 'speed': sp, 'label': rec['label'], 'attack_type': rec['attack_type'], 'heading': rec['heading'], 'last_rec': rec, 'det_label': 0, 'det_score': None})
        veh['last_time'] = t
        veh['x'] = rec['x']
        veh['y'] = rec['y']
        veh['speed'] = sp
        veh['label'] = rec['label']
        veh['attack_type'] = rec['attack_type']
        veh['heading'] = rec['heading']
        veh['last_rec'] = rec
        det = det_result if det_result is not None else self._detect_with_model(rec)
        if det:
            score = det.get('p_attack', None)
            veh['det_score'] = score
            veh['det_label'] = 1 if score is not None and score >= self.decision_threshold else 0
            if veh['det_label'] == 1:
                self.trigger_attack_alert(sender, rec['x'], rec['y'], veh.get('det_score'))
        self.vehicles[sender] = veh
        msg = {'time': t, 'receiver': rec['receiver_pseudo'], 'sender': sender, 'x': rec['x'], 'y': rec['y'], 'speed': sp, 'label': rec['label'], 'attack_type': rec['attack_type'], 'file': rec['file_path'], 'det_label': veh.get('det_label', rec['label']), 'det_score': veh.get('det_score', None)}
        self._update_cumulative_performance(msg)
        self.messages.append(msg)
        if len(self.messages) > MAX_MESSAGES:
            self.messages = self.messages[-MAX_MESSAGES:]
        self.per_sec_total[sec] = self.per_sec_total.get(sec, 0) + 1
        if rec['label'] == 1:
            self.per_sec_attack[sec] = self.per_sec_attack.get(sec, 0) + 1
            atype = rec['attack_type'] or rec['meta_attack_type'] or 'Unknown'
            self.attack_type_counts[atype] = self.attack_type_counts.get(atype, 0) + 1
        self.per_sec_vehicle_ids.setdefault(sec, set()).add(sender)
        self.per_sec_sum_speed[sec] = self.per_sec_sum_speed.get(sec, 0.0) + sp
        self.per_sec_count_speed[sec] = self.per_sec_count_speed.get(sec, 0) + 1
        if sender not in self.seen_vehicle_ids:
            self.seen_vehicle_ids.add(sender)
            self.per_sec_new_veh[sec] = self.per_sec_new_veh.get(sec, 0) + 1
        if rec['label'] == 1:
            self.append_log(f"<span style='color:red'><b>ALERT</b> [t={t:.3f}] sender={sender} recv={rec['receiver_pseudo']} speed={sp:.1f} attack={rec['attack_type']}</span>")
        else:
            self.append_log(f"[t={t:.3f}] sender={sender} recv={rec['receiver_pseudo']} speed={sp:.1f} label={rec['label']} attack={rec['attack_type']}")

    def update_vehicle_table(self):
        rows = list(self.vehicles.values())
        rows.sort(key=lambda r: r['last_time'])
        self.tbl_vehicles.setRowCount(len(rows))
        for i, v in enumerate(rows):
            det_label = v.get('det_label', 0)
            det_score = v.get('det_score', None)
            items = [str(v['sender_pseudo']), f"{v['last_time']:.3f}", f"{v['speed']:.2f}", f"{v['x']:.1f}", f"{v['y']:.1f}", str(v['label']), v['attack_type'], str(det_label), '-' if det_score is None else f'{det_score:.3f}']
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                if det_label == 1:
                    item.setBackground(QColor('#ffcdd2'))
                self.tbl_vehicles.setItem(i, j, item)

    def update_messages_table(self):
        if self.current_vehicle_filter is None:
            rows = self.messages
        else:
            rows = [m for m in self.messages if m['sender'] == self.current_vehicle_filter]
        rows = list(rows)[-MAX_MESSAGES:]
        rows.sort(key=lambda r: r['time'], reverse=True)
        self.tbl_messages.setRowCount(len(rows))
        for i, m in enumerate(rows):
            det_label = m.get('det_label', 0)
            det_score = m.get('det_score', None)
            items = [f"{m['time']:.3f}", str(m['receiver']), str(m['sender']), f"{m['x']:.1f}", f"{m['y']:.1f}", f"{m['speed']:.2f}", str(m['label']), m['attack_type'], os.path.basename(m['file']), str(det_label), '-' if det_score is None else f'{det_score:.3f}']
            for j, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                if det_label == 1:
                    if j == 0:
                        item.setBackground(QColor('#c62828'))
                        item.setForeground(Qt.white)
                    else:
                        item.setBackground(QColor('#ffcdd2'))
                self.tbl_messages.setItem(i, j, item)

    def update_stats_labels(self):
        total_veh = len(self.vehicles)
        attack_veh = sum((1 for v in self.vehicles.values() if v['label'] == 1))
        benign_veh = total_veh - attack_veh
        total_msgs = len(self.messages)
        attack_msgs = sum((1 for m in self.messages if m['label'] == 1))
        self.lbl_total_veh.setText(f'Vehicles: {total_veh}')
        self.lbl_attack_veh.setText(f'Attack vehicles: {attack_veh}')
        self.lbl_benign_veh.setText(f'Benign vehicles: {benign_veh}')
        self.lbl_total_msgs.setText(f'Messages: {total_msgs}')
        self.lbl_attack_msgs.setText(f'Attack msgs: {attack_msgs}')

    def update_map_view(self):
        self.map_scene.clear()
        self.map_items.clear()
        if self.map_background is not None and (not self.map_background.isNull()):
            img_w = self.map_background.width()
            img_h = self.map_background.height()
            bg_item = self.map_scene.addPixmap(self.map_background)
            bg_item.setPos(0, 0)
            bg_item.setZValue(-100)
        else:
            img_w = img_h = 1000
        if not self.vehicles:
            self.map_scene.setSceneRect(0, 0, img_w, img_h)
            self.map_view.fitInView(self.map_scene.sceneRect(), Qt.KeepAspectRatio)
            return
        if self.global_min_x is None or self.global_max_x is None or self.global_min_y is None or (self.global_max_y is None):
            xs = [v['x'] for v in self.vehicles.values()]
            ys = [v['y'] for v in self.vehicles.values()]
            gmin_x, gmax_x = (min(xs), max(xs))
            gmin_y, gmax_y = (min(ys), max(ys))
        else:
            gmin_x, gmax_x = (self.global_min_x, self.global_max_x)
            gmin_y, gmax_y = (self.global_min_y, self.global_max_y)
        sim_w = max(gmax_x - gmin_x, 1.0)
        sim_h = max(gmax_y - gmin_y, 1.0)
        for vid, v in self.vehicles.items():
            nx = (v['x'] - gmin_x) / sim_w
            ny = (v['y'] - gmin_y) / sim_h
            px = nx * img_w * MAP_EXTRA_SCALE + MAP_OFFSET_X
            py = (1.0 - ny) * img_h * MAP_EXTRA_SCALE + MAP_OFFSET_Y
            det_label = v.get('det_label', 0)
            color = QColor('#c62828') if det_label == 1 else QColor('#2e7d32')
            heading = float(v.get('heading', 0.0))
            car = CarItem(veh_id=vid, heading=heading, color=color, dashboard=self, selected=vid == self.selected_vehicle_id or self.is_attack_highlight(vid))
            car.setPos(px, py)
            self.map_scene.addItem(car)
            self.map_items[vid] = car
            if self.pending_center_vehicle_id == vid and self.is_attack_highlight(vid):
                self.map_view.centerOn(car)
        self.map_scene.setSceneRect(0, 0, img_w, img_h)
        self.map_view.fitInView(self.map_scene.sceneRect(), Qt.KeepAspectRatio)
        self.append_log(f'MAP DEBUG: {len(self.vehicles)} vehicles, x∈[{gmin_x:.1f},{gmax_x:.1f}], y∈[{gmin_y:.1f},{gmax_y:.1f}]')

    def update_analytics_charts(self):
        if not self.messages:
            self.cur_msg_rate.setData([], [])
            self.cur_veh_rate.setData([], [])
            self.cur_attack_rate.setData([], [])
            self.cur_cum_msg.setData([], [])
            self.cur_cum_attack.setData([], [])
            self.cur_cum_veh.setData([], [])
            self.cur_attack_ratio.setData([], [])
            self.cur_avg_speed.setData([], [])
            self.plot_attack_types.clear()
            self.attack_bar_item = None
            return
        max_time = max((m['time'] for m in self.messages))
        max_sec = int(max_time)
        min_sec = max(0, max_sec - ANALYTICS_WINDOW_SECS + 1)
        secs = list(range(min_sec, max_sec + 1))
        totals = []
        attacks = []
        veh_rates = []
        avg_speeds = []
        ratios = []
        cum_msg = []
        cum_atk = []
        cum_veh = []
        running_msg = 0
        running_atk = 0
        running_veh = 0
        for s in secs:
            tot = self.per_sec_total.get(s, 0)
            atk = self.per_sec_attack.get(s, 0)
            veh_set = self.per_sec_vehicle_ids.get(s, set())
            veh_rate = len(veh_set)
            sum_sp = self.per_sec_sum_speed.get(s, 0.0)
            cnt_sp = self.per_sec_count_speed.get(s, 0)
            avg_sp = sum_sp / cnt_sp if cnt_sp > 0 else 0.0
            new_veh = self.per_sec_new_veh.get(s, 0)
            running_msg += tot
            running_atk += atk
            running_veh += new_veh
            totals.append(tot)
            attacks.append(atk)
            veh_rates.append(veh_rate)
            avg_speeds.append(avg_sp)
            ratios.append(atk / tot if tot > 0 else 0.0)
            cum_msg.append(running_msg)
            cum_atk.append(running_atk)
            cum_veh.append(running_veh)
        self.cur_msg_rate.setData(secs, totals)
        self.cur_veh_rate.setData(secs, veh_rates)
        self.cur_attack_rate.setData(secs, attacks)
        self.cur_cum_msg.setData(secs, cum_msg)
        self.cur_cum_attack.setData(secs, cum_atk)
        self.cur_cum_veh.setData(secs, cum_veh)
        self.cur_attack_ratio.setData(secs, ratios)
        self.cur_avg_speed.setData(secs, avg_speeds)
        self.plot_attack_types.clear()
        types = sorted(self.attack_type_counts.keys())
        if types:
            xs = list(range(len(types)))
            heights = [self.attack_type_counts[t] for t in types]
            self.attack_bar_item = pg.BarGraphItem(x=xs, height=heights, width=0.6, brush=pg.mkBrush('#ef5350'))
            self.plot_attack_types.addItem(self.attack_bar_item)
            ax = self.plot_attack_types.getAxis('bottom')
            ax.setTicks([[(i, t) for i, t in enumerate(types)]])

    def update_performance_tab(self):

        def fmt(v):
            return '-' if v is None else f'{v:.3f}'

        def compute_from_scores(hist: List[Any], th: float):
            tp = fp = fn = tn = 0
            pred_attack = 0
            for score, gt in hist:
                pred = 1 if score >= th else 0
                pred_attack += pred
                if pred == 1 and gt == 1:
                    tp += 1
                elif pred == 1 and gt == 0:
                    fp += 1
                elif pred == 0 and gt == 1:
                    fn += 1
                else:
                    tn += 1
            total = tp + fp + fn + tn
            if total == 0:
                return dict(tp=0, fp=0, fn=0, tn=0, total=0)
            acc = (tp + tn) / total
            prec = tp / (tp + fp) if tp + fp > 0 else None
            rec = tp / (tp + fn) if tp + fn > 0 else None
            fpr = fp / (fp + tn) if fp + tn > 0 else None
            fnr = fn / (tp + fn) if tp + fn > 0 else None
            f1 = 2 * prec * rec / (prec + rec) if prec is not None and rec is not None and (prec + rec > 0) else None
            return dict(tp=tp, fp=fp, fn=fn, tn=tn, total=total, acc=acc, prec=prec, rec=rec, f1=f1, fpr=fpr, fnr=fnr, pred_attack_ratio=pred_attack / total if total > 0 else None)

        def prob_stats(hist: List[Any]):
            atk_scores = [s for s, l in hist if l == 1]
            ben_scores = [s for s, l in hist if l == 0]
            all_scores = [s for s, _ in hist]
            return (np.mean(atk_scores) if atk_scores else None, np.mean(ben_scores) if ben_scores else None, np.mean(all_scores) if all_scores else None)
        all_hist = list(self.perf_history)
        recent_hist = all_hist[-500:]
        stats = compute_from_scores(all_hist, self.decision_threshold)
        r_stats = compute_from_scores(recent_hist, self.decision_threshold)
        atk_mean, ben_mean, all_mean = prob_stats(all_hist)
        r_atk_mean, r_ben_mean, r_all_mean = prob_stats(recent_hist)
        self.perf_lbl_thresh.setText(f'Decision threshold: {self.decision_threshold:.3f} (auto, N={len(all_hist)})')
        self.perf_lbl_counts.setText(f"Evaluated msgs: {stats.get('total', 0)} | TP: {stats.get('tp', 0)} | FP: {stats.get('fp', 0)} | FN: {stats.get('fn', 0)} | TN: {stats.get('tn', 0)}")
        self.perf_lbl_metrics.setText('Accuracy: {} | Precision: {} | Recall: {} | F1: {}'.format(fmt(stats.get('acc')), fmt(stats.get('prec')), fmt(stats.get('rec')), fmt(stats.get('f1'))))
        self.perf_lbl_recent.setText(f"Last 500 msgs -> N={r_stats.get('total', 0)} | Accuracy: {fmt(r_stats.get('acc'))} | Precision: {fmt(r_stats.get('prec'))} | Recall: {fmt(r_stats.get('rec'))} | F1: {fmt(r_stats.get('f1'))}")
        self.perf_lbl_rates.setText('Rates: FPR: {} | FNR: {} | Pred attack ratio: {}'.format(fmt(stats.get('fpr')), fmt(stats.get('fnr')), fmt(stats.get('pred_attack_ratio'))))
        self.perf_lbl_probs.setText('Scores mean (overall/last500) -> attack: {}/{} | benign: {}/{} | overall: {}/{}'.format(fmt(atk_mean), fmt(r_atk_mean), fmt(ben_mean), fmt(r_ben_mean), fmt(all_mean), fmt(r_all_mean)))

    def append_log(self, text: str):
        self.log_view.append(text)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = LiveDashboard(ROOT_DIR)
    win.show()
    sys.exit(app.exec_())
if __name__ == '__main__':
    main()
