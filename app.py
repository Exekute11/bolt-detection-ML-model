"""
Godrej Fastener Vision - multi-station Flask backend (up to 4 independent inspection stations)

Architecture
  * ONE YOLO model (models/hex_bolt_best.pt) shared by all stations.
  * ONE inference worker thread walks the active stations round-robin and runs the model
    one frame at a time under MODEL_LOCK  -> the model object is never called concurrently.
  * Every station owns its own: config, camera process (camera_worker.py) or video thread, newest frame,
    detections, count history, stable count, PASS/FAIL status, error text and MJPEG stream.
  * Disabled / stopped stations open no camera and consume no inference time.
  * A supervisor thread restarts failed cameras. A failed camera NEVER falls back to the demo video
    and NEVER shows PASS: the station shows CAMERA ERROR.
"""
import atexit
import csv
import io
import json
import logging
import os
import signal
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory
from rich.console import Console
from rich.markup import escape
from ultralytics import YOLO

app = Flask(__name__)
console = Console()
logging.getLogger("werkzeug").setLevel(logging.ERROR)     # hide Flask request spam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(BASE_DIR, "snapshots")
DB_PATH = os.path.join(BASE_DIR, "inspections.db")
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "demo.mp4")
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
CAM_WORKER = os.path.join(BASE_DIR, "camera_worker.py")
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "hex_bolt_best_openvino_model")
EXPECTED_CLASSES = {0: "hex_bolt"}
MAX_STATIONS = 4
FRAME_TIMEOUT = 5.0        # seconds without a frame -> CAMERA ERROR
OPEN_TIMEOUT = 40.0        # seconds a camera may take to open
STALL_SECONDS = 6.0        # camera opened but stopped delivering
INFER_MIN_W = 640         # frames wider than this are downscaled before YOLO (CPU saver)
SHUTDOWN = threading.Event()


def log(msg, style=""):
    console.print(f"[{style}]{msg}[/{style}]" if style else msg, markup=True, highlight=False)


# ======================================================================= model
def resolve_model_path():
    """Load the configured bolt detection model."""
    env = os.environ.get("BOLT_MODEL", "").strip()

    if env:
        p = env if os.path.isabs(env) else os.path.join(BASE_DIR, env)

        if not os.path.exists(p):
            log(
                f"ERROR: BOLT_MODEL points to a missing model: {p}",
                "bold red"
            )
            raise SystemExit(1)

        log(
            f"WARNING: BOLT_MODEL override in use -> {p}",
            "bold yellow"
        )
        return p

    if not os.path.exists(DEFAULT_MODEL):
        log("ERROR: model not found.", "bold red")
        log(f"Expected: {DEFAULT_MODEL}")
        raise SystemExit(1)

    return DEFAULT_MODEL


MODEL_PATH = None
model = None
CLASS_NAMES = {}
CLASSES_OK = False
MODEL_LOCK = threading.Lock()          # the ONLY way the model is ever called
USE_GPU = False


def init_model():
    global MODEL_PATH, model, CLASS_NAMES, CLASSES_OK, USE_GPU
    MODEL_PATH = resolve_model_path()
    log(f"\nUSING MODEL: {MODEL_PATH}\n", "bold green")
    model = YOLO(MODEL_PATH)
    CLASS_NAMES = {int(k): str(v) for k, v in (getattr(model, "names", {}) or {}).items()}
    CLASSES_OK = CLASS_NAMES == EXPECTED_CLASSES
    log(f"MODEL CLASSES: {CLASS_NAMES}", "cyan")
    if CLASSES_OK:
        log("Class mapping verified: {0: 'hex_bolt'}", "bold green")
    else:
        log(f"WARNING: expected {EXPECTED_CLASSES} but the model reports {CLASS_NAMES}. "
            f"Check that the correct model file is installed.", "bold red")
    try:
        import torch
        USE_GPU = bool(torch.cuda.is_available())
        if not USE_GPU:                               # leave CPU cores free for video reading/encoding
            torch.set_num_threads(max(2, (os.cpu_count() or 4) - 2))
    except Exception:
        USE_GPU = False
    log(f"Inference device: {'GPU (CUDA)' if USE_GPU else 'CPU'}", "cyan")


# ================================================================= settings.json
CAM_LIMITS = {"display_width": (320, 3840), "jpeg_quality": (50, 100), "width": (320, 3840),
              "height": (240, 2160), "fps": (5, 120), "exposure": (-13, 0), "focus": (0, 255), "zoom": (0, 500),
              "sw_brightness": (-100, 100), "sw_contrast": (0.5, 2.0), "sw_gamma": (0.4, 3.0)}
CAM_INT_KEYS = ("display_width", "jpeg_quality", "width", "height", "fps", "exposure", "focus", "zoom", "sw_brightness")
CAM_REOPEN_KEYS = ("width", "height", "fps", "backend")
CAM_DEFAULTS = {"auto_exposure": True, "exposure": -6, "autofocus": True, "focus": 0,
                "sw_brightness": 0, "sw_contrast": 1.0, "sw_gamma": 1.0, "sw_enhance": False}


def default_stations():
    names = ["Front Shield", "End Shield", "Inspection 3", "Inspection 4"]
    refs, cams, en = [5, 6, 0, 0], [0, 1, 2, 3], [True, True, False, False]
    return [{"id": i + 1, "enabled": en[i], "name": names[i], "camera": cams[i], "reference": refs[i],
             "conf": 0.35, "smooth": 5, "roi": None} for i in range(MAX_STATIONS)]


settings = {
    "detection": {"iou": 0.5, "imgsz": None, "device": None, "min_box_area_pct": 0.0, "max_box_area_pct": 100.0,
                  "class_labels": {"hex_bolt": "Hex Bolt"}, "dedupe_iou": 0.6},
    "camera": {"zoom": None, "backend": "auto", "display_width": 960, "jpeg_quality": 80,
               "width": 1280, "height": 720, "fps": 30, "auto_exposure": True, "exposure": -6,
               "autofocus": True, "focus": 0, "sw_brightness": 0, "sw_contrast": 1.0, "sw_gamma": 1.0,
               "sw_enhance": False},
    "camera_modes": {},
    "autostart": True,
    "stations": default_stations(),
}
settings_lock = threading.RLock()


def clean_station(patch):
    """Validate a station config patch. Returns (clean_dict, errors)."""
    out, err = {}, []
    if "enabled" in patch:
        out["enabled"] = bool(patch["enabled"])
    if "name" in patch:
        n = str(patch["name"]).strip()[:40]
        if n:
            out["name"] = n
        else:
            err.append("Inspection name cannot be empty")
    if "camera" in patch:
        c = patch["camera"]
        if isinstance(c, str) and c.strip().lower() == "video":
            out["camera"] = "video"
        else:
            try:
                c = int(str(c).lower().replace("camera:", "").replace("camera", "").strip())
                if not 0 <= c <= 9:
                    raise ValueError
                out["camera"] = c
            except ValueError:
                err.append("Camera must be 0-9 or 'video'")
    if "reference" in patch:
        try:
            r = int(patch["reference"])
            if not 0 <= r <= 999:
                raise ValueError
            out["reference"] = r
        except (TypeError, ValueError):
            err.append("Reference count must be a whole number 0-999")
    if "conf" in patch:
        try:
            out["conf"] = round(max(0.05, min(float(patch["conf"]), 0.95)), 2)
        except (TypeError, ValueError):
            err.append("Confidence must be a number 0.05-0.95")
    if "smooth" in patch:
        try:
            out["smooth"] = max(1, min(int(patch["smooth"]), 30))
        except (TypeError, ValueError):
            err.append("Smoothing must be a whole number 1-30")
    if "roi" in patch:
        r = patch["roi"]
        if r is None or r is False:
            out["roi"] = None
        else:
            try:
                if isinstance(r, dict):
                    r = [r["x1"], r["y1"], r["x2"], r["y2"]]
                x1, y1, x2, y2 = [max(0.0, min(float(v), 1.0)) for v in r]
                x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
                if x2 - x1 < 0.03 or y2 - y1 < 0.03:
                    raise ValueError
                out["roi"] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
            except (KeyError, TypeError, ValueError):
                err.append("Detection zone too small - draw a larger zone")
    return out, err


def load_settings():
    """Merge settings.json into defaults. Unknown/old keys (parts, cycle, ...) are preserved when saving."""
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            saved = json.load(f)
    except Exception as e:
        saved = {}
        if os.path.exists(SETTINGS_PATH):
            log(f"settings.json unreadable ({e}) - using defaults", "yellow")
    with settings_lock:
        for k, v in saved.items():
            if k in ("detection", "camera") and isinstance(v, dict):
                settings[k].update(v)
            elif k != "stations":
                settings[k] = v
        lbl = settings["detection"].get("class_labels") or {}
        lbl.setdefault("hex_bolt", "Hex Bolt")
        settings["detection"]["class_labels"] = lbl
        old_conf = saved.get("detection", {}).get("conf")          # migrate the old single global confidence
        stations = default_stations()
        if old_conf and not saved.get("stations"):
            for s in stations:
                s["conf"] = round(float(old_conf), 2)
        for i, s in enumerate(saved.get("stations") or []):
            if i < MAX_STATIONS and isinstance(s, dict):
                clean, _ = clean_station(s)
                stations[i].update(clean)
        settings["stations"] = stations


def save_settings():
    """Atomic write (temp file + replace) so a crash can never leave a half-written settings.json."""
    with settings_lock:
        try:
            tmp = SETTINGS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            os.replace(tmp, SETTINGS_PATH)
        except Exception as e:
            log(f"Settings save failed: {e}", "red")


# ============================================================ events + history DB
events = deque(maxlen=60)
_event_id = [0]
events_lock = threading.Lock()


def log_event(level, msg):
    with events_lock:
        _event_id[0] += 1
        events.appendleft({"id": _event_id[0], "t": time.strftime("%H:%M:%S"), "level": level, "msg": msg})
    color = {"alert": "red", "ok": "green"}.get(level, "cyan")
    console.print(f"[{color}]{level.upper()}[/{color}] {escape(msg)}", highlight=False)


db_lock = threading.Lock()


def db_init():
    os.makedirs(SNAP_DIR, exist_ok=True)
    with db_lock, sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS inspections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL, type TEXT NOT NULL, source TEXT,
            current_count INTEGER, reference_count INTEGER, missing_count INTEGER,
            message TEXT, snapshot TEXT, part TEXT)""")
        cols = [r[1] for r in c.execute("PRAGMA table_info(inspections)")]
        if "part" not in cols:                       # migrate databases created by older versions
            c.execute("ALTER TABLE inspections ADD COLUMN part TEXT")


def record(kind, message, current, reference, missing, jpg, source, part):
    """Store an inspection record (+ optional snapshot). `part` = station name. Never raises."""
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        name = None
        if jpg is not None:
            safe = "".join(ch for ch in part if ch.isalnum())[:12] or "station"
            name = time.strftime("%Y%m%d_%H%M%S") + f"_{safe}_{kind}.jpg"
            with open(os.path.join(SNAP_DIR, name), "wb") as f:
                f.write(jpg)
        with db_lock, sqlite3.connect(DB_PATH) as c:
            c.execute("INSERT INTO inspections(ts,type,source,current_count,reference_count,missing_count,"
                      "message,snapshot,part) VALUES(?,?,?,?,?,?,?,?,?)",
                      (ts, kind, source, current, reference, missing, message, name, part))
    except Exception as e:
        log(f"History write failed: {e}", "red")


def query_history(kind=None, limit=200):
    q, args = "SELECT * FROM inspections", []
    if kind in ("alert", "reference", "ok", "pass", "fail"):
        q += " WHERE type=?"
        args.append(kind)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with db_lock, sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, args).fetchall()]


# ============================================================ image adjustment
_lut_lock = threading.Lock()
_lut_cache = {}


def sw_adjust(frame, clahe):
    """Software brightness / contrast / highlight recovery / local contrast (before display AND detection)."""
    c = settings["camera"]
    b, ct, g, enh = c["sw_brightness"], c["sw_contrast"], c["sw_gamma"], c["sw_enhance"]
    if b == 0 and ct == 1.0 and g == 1.0 and not enh:
        return frame
    key = (b, ct, g)
    with _lut_lock:
        lut = _lut_cache.get(key)
        if lut is None:
            x = np.power(np.arange(256) / 255.0, g) * 255.0
            lut = np.clip((x - 128.0) * ct + 128.0 + b, 0, 255).astype(np.uint8)
            if len(_lut_cache) > 8:
                _lut_cache.clear()
            _lut_cache[key] = lut
    out = cv2.LUT(frame, lut)
    if enh:
        l, a, bb = cv2.split(cv2.cvtColor(out, cv2.COLOR_BGR2LAB))
        out = cv2.cvtColor(cv2.merge((clahe.apply(l), a, bb)), cv2.COLOR_LAB2BGR)
    return out


# ================================================================ drawing
COL = {"PASS": (60, 170, 40), "FAIL": (50, 50, 225), "WAITING": (0, 165, 255),
       "CAMERA_ERROR": (170, 40, 140), "STOPPED": (130, 130, 130), "DISABLED": (130, 130, 130)}


def draw_overlay(img, d):
    """Boxes are normalised (0..1) so they fit any resolution. Numbers come from the station snapshot."""
    h, w = img.shape[:2]
    k = max(0.7, h / 720.0)
    F = cv2.FONT_HERSHEY_SIMPLEX
    BOX, TXT = (56, 200, 98), (0, 45, 0)
    for (x1, y1, x2, y2, label) in d["boxes"]:
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        bw, bh = max(p2[0] - p1[0], 1), max(p2[1] - p1[1], 1)
        ref = max(8, min(bw, bh))
        lw = max(1, int(round((1 if ref < 40 else 2 if ref < 130 else 3) * max(1.0, h / 800.0))))
        cv2.rectangle(img, p1, p2, BOX, lw)
        txt_h = min(max(0.12 * ref, 10 * k), 20 * k)
        fs, ft = txt_h / 22.0, (1 if txt_h < 15 else 2)
        pad = max(2, int(txt_h * 0.28))
        (tw, tht), base = cv2.getTextSize(label, F, fs, ft)
        if tw + 2 * pad > 1.3 * bw and " " in label:        # small box: keep only the confidence
            label = label.rsplit(" ", 1)[1]
            (tw, tht), base = cv2.getTextSize(label, F, fs, ft)
        bar_h, bar_w = tht + base + 2 * pad, tw + 2 * pad
        top = p1[1] - bar_h + lw // 2
        if top < 0:
            top = p1[1]
        left = p1[0] - lw // 2
        cv2.rectangle(img, (left, top), (left + bar_w, top + bar_h), BOX, -1)
        cv2.putText(img, label, (left + pad, top + bar_h - pad - base // 2), F, fs, TXT, ft, cv2.LINE_AA)
    if d.get("roi"):
        rx1, ry1, rx2, ry2 = d["roi"]
        cv2.rectangle(img, (int(rx1 * w), int(ry1 * h)), (int(rx2 * w), int(ry2 * h)), (255, 255, 255), 1)
        cv2.putText(img, "ZONE", (int(rx1 * w) + 6, int(ry1 * h) + int(18 * k)), F, 0.5 * k, (255, 255, 255), 1)
    # header band: station name, counts, status
    band = int(34 * k)
    img[:band] = (img[:band] * 0.35 + np.array((30, 30, 30), np.float32) * 0.65).astype(np.uint8)
    det = "--" if d["detected"] is None else str(d["detected"])
    cv2.putText(img, f"{d['name']}   DETECTED {det} / REF {d['reference']}", (int(10 * k), int(24 * k)),
                F, 0.62 * k, (255, 255, 255), 1, cv2.LINE_AA)
    stt = d["status"].replace("_", " ")
    (tw, _), _ = cv2.getTextSize(stt, F, 0.62 * k, 2)
    cv2.rectangle(img, (w - tw - int(24 * k), 0), (w, band), COL.get(d["status"], (90, 90, 90)), -1)
    cv2.putText(img, stt, (w - tw - int(12 * k), int(24 * k)), F, 0.62 * k, (255, 255, 255), 2, cv2.LINE_AA)


def iou_xyxy(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe(dets, thr):
    """Greedy NMS on top of YOLO's own NMS: overlapping duplicates are never counted as two bolts."""
    keep = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        if all(iou_xyxy(d["xyxy"], k["xyxy"]) <= thr for k in keep):
            keep.append(d)
    return keep


# ========================================================= camera sub-process
def _readn(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class CamProc:
    """One webcam session of ONE station = one separate process (camera_worker.py).
    A driver hang/crash can only kill that process; the app and the other stations keep running."""

    def __init__(self, station):
        self.st = station
        c = dict(settings["camera"])
        c["index"] = int(station.cfg["camera"])
        self.cfg = c
        self.key = f"{c['index']}|{c['width']}x{c['height']}@{c['fps']}"
        self.proc = None
        self.ready = self.failed = self.stopped = False
        self.fail_msg = ""
        self.t0 = self.last_msg = time.time()

    def start(self):
        cfg = dict(self.cfg)
        cfg["mode_hint"] = settings["camera_modes"].get(self.key)
        flags = 0x08000000 if os.name == "nt" else 0          # no console window on Windows
        self.proc = subprocess.Popen([sys.executable, CAM_WORKER, json.dumps(cfg)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags)
        threading.Thread(target=self._pump, daemon=True, name=f"cam{self.st.id}-pump").start()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def send(self, obj):
        try:
            self.proc.stdin.write((json.dumps(obj) + "\n").encode())
            self.proc.stdin.flush()
        except Exception:
            pass

    def stop(self):
        """Ask politely, then kill. Killing the process is what frees a hung camera driver."""
        self.stopped = True
        if self.proc is None:
            return
        try:
            self.send({"cmd": "stop"})
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(1.0)
            except Exception:
                self.proc.kill()
                self.proc.wait(2.0)
        except Exception:
            pass

    def _pump(self):
        out = self.proc.stdout
        try:
            while True:
                hdr = _readn(out, 5)
                if not hdr:
                    break
                payload = _readn(out, struct.unpack("<I", hdr[1:])[0])
                if payload is None:
                    break
                self.last_msg = time.time()
                if hdr[:1] == b"F":
                    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None and not self.stopped:
                        self.st.on_frame(frame, self)
                else:
                    self._on_status(json.loads(payload))
        except Exception as e:
            if not self.stopped:
                self.failed, self.fail_msg = True, f"Camera link error: {e}"
        finally:
            if not self.stopped and not self.ready:
                self.failed = True

    def _on_status(self, m):
        t = m.get("t")
        if t in ("open", "slow"):
            self.st.opening_msg = m.get("msg", "")
        elif t == "ready":
            self.ready = True
            self.st.cam_info = dict(m["info"])
            self.st.opening_msg = ""
            log_event("info", f"{self.st.name}: camera #{m['info']['index']} ({m['info']['backend']}) "
                              f"{m['info']['width']}x{m['info']['height']} {m['info']['fourcc']}")
        elif t == "info":
            self.st.cam_info.update(m.get("info", {}))
            if m.get("mode"):
                settings["camera_modes"][self.key] = m["mode"]
                save_settings()
            if m.get("note"):
                log_event("info", f"{self.st.name}: {m['note']}")
        elif t == "error":
            self.failed, self.fail_msg = True, m.get("msg", "Camera error")


class VideoThread(threading.Thread):
    """Plays videos/demo.mp4 in a loop for testing without cameras (own VideoCapture per station)."""

    def __init__(self, station, gen):
        super().__init__(daemon=True, name=f"video{station.id}")
        self.st, self.gen = station, gen
        self.failed_msg = ""

    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            self.failed_msg = "Video file not found or unreadable: videos/demo.mp4"
            return
        native = cap.get(cv2.CAP_PROP_FPS)
        period = 1.0 / native if native and 1 < native < 120 else 0.04
        fails = 0
        try:
            while self.st.gen == self.gen and not SHUTDOWN.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)           # loop
                    if fails > 30:
                        self.failed_msg = "Video file cannot be read"
                        return
                    continue
                fails = 0
                self.st.on_frame(frame, self)
                time.sleep(max(period - (time.time() - t0), 0))
        finally:
            cap.release()


# ============================================================== Station
def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


STATIONS = []
START_LOCK = threading.RLock()


class Station:
    """Everything that belongs to ONE inspection station. Nothing here is shared with other stations."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.id = cfg["id"]
        self.lock = threading.RLock()
        self.running = False
        self.gen = 0                                  # incremented on every start/stop (kills old threads/streams)
        self.cp = None
        self.vt = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.frame_cond = threading.Condition()
        self.jpg_cond = threading.Condition()
        self.jpg, self.jpg_seq = None, 0
        self.cam_info = {}
        self.next_retry, self.fail_count = 0.0, 0
        self._reset_runtime()

    @property
    def name(self):
        return self.cfg["name"]

    def _reset_runtime(self):
        with self.lock:
            self.frame, self.frame_seq, self.infer_seq = None, 0, 0
            self.last_frame_t = 0.0
            self.frame_w = self.frame_h = 0
            self.boxes, self.raw_count, self.avg_conf = [], None, 0.0
            self.history = deque(maxlen=30)
            self.stable_count = None
            self.error = ""
            self.opening_msg = ""
            self.cam_fps = self.det_fps = 0.0
            self._t_cam = self._t_det = time.time()
            self.last_reported = None

    # ---------------------------------------------------------------- status
    def _status_locked(self, now):
        cfg = self.cfg
        if not cfg["enabled"]:
            return "DISABLED", "Station disabled", None, None
        if not self.running:
            return "STOPPED", "Stopped - press Start", None, None
        if self.error:
            return "CAMERA_ERROR", self.error, None, None
        if self.last_frame_t == 0:
            return "WAITING", self.opening_msg or "Opening source...", None, None
        age = now - self.last_frame_t
        if age > FRAME_TIMEOUT:
            return "CAMERA_ERROR", f"No camera frames for {age:.0f} s", None, None
        if self.stable_count is None:
            return "WAITING", "Waiting for stable detection...", None, None
        d, r = self.stable_count, cfg["reference"]
        missing, extra = max(r - d, 0), max(d - r, 0)
        if d == r:
            return "PASS", f"All {plural(r, 'hex bolt')} present", missing, extra
        if missing:
            return "FAIL", f"{plural(missing, 'missing hex bolt')} ({d}/{r})", missing, extra
        return "FAIL", f"{plural(extra, 'extra detection')} ({d}/{r})", missing, extra

    def view(self):
        with self.lock:
            now = time.time()
            status, msg, missing, extra = self._status_locked(now)
            cfg = dict(self.cfg)
            return {"id": self.id, "station_id": self.id, "name": cfg["name"], "enabled": cfg["enabled"],
                    "camera": cfg["camera"], "reference": cfg["reference"], "conf": cfg["conf"],
                    "smooth": cfg["smooth"], "roi": cfg["roi"], "running": self.running,
                    "status": status, "message": msg, "error": self.error,
                    "detected": self.stable_count, "live_count": self.raw_count,
                    "missing": missing, "extra": extra,
                    "pass": (status == "PASS") if status in ("PASS", "FAIL") else None,
                    "avg_conf": self.avg_conf, "fps": round(self.cam_fps, 1), "det_fps": round(self.det_fps, 1),
                    "frame_w": self.frame_w, "frame_h": self.frame_h, "gen": self.gen,
                    "opening": self.opening_msg, "camera_info": dict(self.cam_info),
                    "feed_url": f"/api/stations/{self.id}/video_feed"}

    def _overlay_state(self):
        with self.lock:
            status, _, _, _ = self._status_locked(time.time())
            return {"boxes": list(self.boxes), "roi": self.cfg["roi"], "name": self.cfg["name"],
                    "reference": self.cfg["reference"], "detected": self.stable_count, "status": status}

    # ---------------------------------------------------------------- frames
    def on_frame(self, frame, source):
        """Called by this station's own camera pump / video thread only."""
        if source is not self.cp and source is not self.vt:
            return                                       # frame from an already-stopped source
        frame = sw_adjust(frame, self.clahe)
        now = time.time()
        with self.lock:
            if not self.running:
                return
            self.frame, self.frame_seq = frame, self.frame_seq + 1
            self.frame_h, self.frame_w = frame.shape[:2]
            inst = 1 / max(now - self._t_cam, 1e-3)
            self.cam_fps = 0.9 * self.cam_fps + 0.1 * inst if self.cam_fps else inst
            self._t_cam = now
            self.last_frame_t = now
            if self.error:                               # source recovered
                self.error, self.fail_count = "", 0
                self.history.clear()
                self.stable_count = None
                self.last_reported = None
        with self.frame_cond:
            self.frame_cond.notify_all()

    def take_frame(self):
        with self.lock:
            if not self.running or self.error or self.frame is None or self.frame_seq == self.infer_seq:
                return None
            self.infer_seq = self.frame_seq
            return self.frame, self.gen

    def commit(self, gen, boxes, confs):
        """Publish one detection result. Smoothing history is per station."""
        now = time.time()
        with self.lock:
            if gen != self.gen or not self.running:
                return
            self.boxes = boxes
            self.raw_count = len(boxes)
            self.avg_conf = round(100 * sum(confs) / len(confs), 1) if confs else 0.0
            self.history.append(self.raw_count)
            k = max(1, int(self.cfg["smooth"]))
            last = list(self.history)[-k:]
            if len(last) >= k and len(set(last)) == 1:   # same count for k frames in a row = stable
                self.stable_count = last[0]
            inst = 1 / max(now - self._t_det, 1e-3)
            self.det_fps = 0.85 * self.det_fps + 0.15 * inst if self.det_fps else inst
            self._t_det = now
            status, msg, missing, extra = self._status_locked(now)
            key = (status, self.stable_count, self.cfg["reference"])
            changed = status in ("PASS", "FAIL") and key != self.last_reported
            if changed:
                self.last_reported = key
            det, ref, name, cam = self.stable_count, self.cfg["reference"], self.cfg["name"], self.cfg["camera"]
        if changed:
            self._report(status, msg, det, ref, missing, name, cam)

    def _report(self, status, msg, det, ref, missing, name, cam):
        src = "video" if cam == "video" else f"camera {cam}"
        jpg = None
        if status == "FAIL":
            jpg = self.render_jpg(quality=90)
            log_event("alert", f"{name}: FAIL - {msg}")
        else:
            log_event("ok", f"{name}: PASS - {det}/{ref} hex bolts")
        record(status.lower(), f"{status}: {msg}", det, ref, missing or 0, jpg, f"S{self.id} {src}", name)

    # ---------------------------------------------------------------- render
    def render_jpg(self, frame=None, quality=None):
        with self.lock:
            frame = frame if frame is not None else self.frame
        if frame is None:
            return None
        c = settings["camera"]
        H, W = frame.shape[:2]
        maxw = int(c.get("display_width", 960))
        img = cv2.resize(frame, (maxw, int(H * maxw / W)), interpolation=cv2.INTER_AREA) if W > maxw else frame.copy()
        draw_overlay(img, self._overlay_state())
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality or c.get("jpeg_quality", 80))])
        return buf.tobytes() if ok else None

    def _render_loop(self, gen):
        last = 0
        while self.gen == gen and not SHUTDOWN.is_set():
            with self.frame_cond:
                self.frame_cond.wait_for(lambda: self.frame_seq != last or self.gen != gen, timeout=0.5)
                if self.gen != gen:
                    break
                if self.frame_seq == last:
                    continue
                frame, last = self.frame, self.frame_seq
            try:
                jpg = self.render_jpg(frame)
            except Exception as e:
                log(f"S{self.id} render error: {e}", "red")
                continue
            if jpg:
                with self.jpg_cond:
                    self.jpg, self.jpg_seq = jpg, self.jpg_seq + 1
                    self.jpg_cond.notify_all()

    # ---------------------------------------------------------------- control
    def start(self):
        """Returns (ok, message)."""
        with START_LOCK:
            with self.lock:
                if not self.cfg["enabled"]:
                    return False, f"{self.name} is disabled - enable it first"
                if self.running:
                    return True, "Already running"
                cam = self.cfg["camera"]
            if cam != "video":
                for o in STATIONS:
                    if o is not self and o.running and o.cfg["camera"] == cam:
                        msg = f"Camera {cam} is already used by {o.name}. Assign a different camera."
                        log_event("alert", f"{self.name}: {msg}")
                        return False, msg
            self._reset_runtime()
            with self.lock:
                self.gen += 1
                self.running = True
                gen = self.gen
                self.fail_count, self.next_retry = 0, 0.0
                self.opening_msg = "Starting camera..." if cam != "video" else "Opening video file..."
            threading.Thread(target=self._render_loop, args=(gen,), daemon=True, name=f"render{self.id}").start()
            self._open_source()
            log_event("info", f"{self.name}: started ({'video file' if cam == 'video' else f'camera {cam}'})")
            return True, f"{self.name} started"

    def _open_source(self):
        cam = self.cfg["camera"]
        try:
            if cam == "video":
                vt = VideoThread(self, self.gen)
                self.vt = vt
                vt.start()
            else:
                cp = CamProc(self)
                self.cp = cp
                cp.start()
        except Exception as e:
            self._source_failed(f"Cannot start camera process: {e}")

    def _close_source(self):
        cp, self.cp, self.vt = self.cp, None, None
        if cp is not None:
            cp.stop()                                    # releases the camera device

    def stop(self, reason="stopped"):
        with self.lock:
            was = self.running
            self.running = False
            self.gen += 1
        self._close_source()
        with self.frame_cond:
            self.frame_cond.notify_all()
        with self.jpg_cond:
            self.jpg = None
            self.jpg_cond.notify_all()
        self._reset_runtime()
        with self.lock:
            self.cam_info = {}
        if was:
            log_event("info", f"{self.name}: {reason} (camera released)")

    def restart_source(self):
        """Re-open the camera (new resolution/driver) without touching other stations."""
        if self.running:
            self._close_source()
            with self.lock:
                self.history.clear()
                self.stable_count = None
                self.last_frame_t = 0.0
                self.frame = None
                self.opening_msg = "Restarting camera..."
            self._open_source()

    def _source_failed(self, msg):
        self._close_source()
        with self.lock:
            first = not self.error
            self.error = msg
            self.fail_count += 1
            self.next_retry = time.time() + min(5 * self.fail_count, 20)
            self.history.clear()
            self.stable_count, self.boxes, self.raw_count = None, [], None
            self.last_reported = None
        if first:
            log_event("alert", f"{self.name}: CAMERA ERROR - {msg}")

    def supervise(self):
        """Called by the supervisor thread ~2x per second."""
        if not self.running:
            return
        now = time.time()
        cam = self.cfg["camera"]
        if cam == "video":
            vt = self.vt
            if vt is not None and not vt.is_alive() and vt.failed_msg:
                self._source_failed(vt.failed_msg)
            elif vt is None and now >= self.next_retry:
                self._open_source()
            return
        cp = self.cp
        if cp is None:
            if now >= self.next_retry:
                self.opening_msg = "Reconnecting camera..."
                self._open_source()
            return
        died = (not cp.alive()) and not cp.stopped
        if cp.failed or died:
            self._source_failed(cp.fail_msg or f"Camera {cam} unavailable or disconnected. Close other apps "
                                f"using it (Teams, Zoom, Camera app) and check the USB cable.")
        elif not cp.ready and now - cp.t0 > OPEN_TIMEOUT:
            self._source_failed(f"Camera {cam} is not responding")
        elif cp.ready and now - cp.last_msg > STALL_SECONDS:
            self._source_failed(f"Camera {cam} stalled (no frames for {STALL_SECONDS:.0f} s)")

    # ---------------------------------------------------------------- config
    def update_config(self, patch):
        clean, errors = clean_station(patch)
        if errors:
            return None, errors
        with self.lock:
            old = dict(self.cfg)
            self.cfg.update(clean)
            new = dict(self.cfg)
            if any(new[k] != old[k] for k in ("roi", "smooth", "conf", "reference")):
                self.history.clear()                    # counts must be re-stabilised under the new settings
                if any(new[k] != old[k] for k in ("roi", "smooth", "conf")):
                    self.stable_count = None
                self.last_reported = None
        save_settings()
        if not new["enabled"] and self.running:
            self.stop("disabled")
        elif new["camera"] != old["camera"] and self.running:
            self.stop("camera changed")
            self.start()
        return self.view(), []


# ============================================================ inference worker
def infer_once(st):
    got = st.take_frame()
    if got is None:
        return False
    frame, gen = got
    det = settings["detection"]
    with st.lock:
        conf, roi = st.cfg["conf"], st.cfg["roi"]
    H0, W0 = frame.shape[:2]
    lim = max(INFER_MIN_W, int(det.get("imgsz") or 0))
    small = cv2.resize(frame, (lim, int(H0 * lim / W0)), interpolation=cv2.INTER_AREA) if W0 > lim else frame
    kw = dict(verbose=False, conf=conf, iou=det["iou"])
    if det.get("imgsz"):
        kw["imgsz"] = int(det["imgsz"])
    if det.get("device") is not None:
        kw["device"] = det["device"]
    elif USE_GPU:
        kw["device"], kw["half"] = 0, True
    with MODEL_LOCK:                                      # never two threads inside the model
        results = model.predict(small, **kw)
    H, W = small.shape[:2]
    amin, amax = det.get("min_box_area_pct", 0) / 100.0, det.get("max_box_area_pct", 100) / 100.0
    labels = det.get("class_labels") or {}
    dets = []
    for r in results:
        for b in r.boxes:
            cid = int(b.cls[0])
            if CLASSES_OK and cid != 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            area = max(x2 - x1, 0) * max(y2 - y1, 0) / float(W * H)
            if area < amin or area > amax:
                continue
            if roi:                                       # ignore detections inside THIS station's exclusion zone
                cx = ((x1 + x2) / 2.0) / W
                cy = ((y1 + y2) / 2.0) / H

                inside_exclusion_zone = (
                    roi[0] <= cx <= roi[2]
                    and roi[1] <= cy <= roi[3]
                )

                if inside_exclusion_zone:
                    continue
            raw_name = str(CLASS_NAMES.get(cid, "hex_bolt"))
            dets.append({"xyxy": (x1, y1, x2, y2), "conf": float(b.conf[0]),
                         "name": labels.get(raw_name) or raw_name.replace("_", " ").title()})
    dets = dedupe(dets, det.get("dedupe_iou", 0.6))
    boxes = [(d["xyxy"][0] / W, d["xyxy"][1] / H, d["xyxy"][2] / W, d["xyxy"][3] / H,
              f"{d['name']} {d['conf']:.2f}") for d in dets]
    st.commit(gen, boxes, [d["conf"] for d in dets])
    return True


def inference_worker():
    """ONE thread serves every active station in turn -> efficient on CPU and thread-safe."""
    last_err = 0.0
    while not SHUTDOWN.is_set():
        did = False
        for st in STATIONS:
            if not (st.running and st.cfg["enabled"]):    # disabled/stopped stations cost nothing
                continue
            try:
                did |= infer_once(st)
            except Exception as e:
                with st.lock:                              # never keep an old count that could show PASS
                    st.stable_count = None
                    st.history.clear()
                if time.time() - last_err > 10:
                    last_err = time.time()
                    log_event("alert", f"{st.name}: inference error: {e}")
                    console.print(traceback.format_exc(), markup=False, highlight=False)
        if not did:
            SHUTDOWN.wait(0.01)


def supervisor():
    while not SHUTDOWN.is_set():
        for st in STATIONS:
            try:
                st.supervise()
            except Exception as e:
                log(f"supervisor error S{st.id}: {e}", "red")
        SHUTDOWN.wait(0.5)


def _forever(fn, name):
    """A worker thread must never die: log the error and restart it."""
    while not SHUTDOWN.is_set():
        try:
            fn()
        except Exception:
            log_event("alert", f"{name} crashed - restarting")
            console.print(traceback.format_exc(), markup=False, highlight=False)
            SHUTDOWN.wait(0.5)


# ============================================================ overall result
def compute_overall(views):
    en = [v for v in views if v["enabled"]]
    if not en:
        return {"status": "NO_STATIONS", "label": "NO STATIONS ENABLED",
                "reasons": ["Enable at least one station"], "enabled": 0, "counts": {}}
    counts = {}
    for v in en:
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    err = [v for v in en if v["status"] == "CAMERA_ERROR"]
    fail = [v for v in en if v["status"] == "FAIL"]
    wait = [v for v in en if v["status"] in ("WAITING", "STOPPED")]
    if err:
        st, label = "CAMERA_ERROR", "OVERALL CAMERA ERROR"
    elif fail:
        st, label = "FAIL", "OVERALL FAIL"
    elif wait:
        st, label = "WAITING", "OVERALL WAITING"
    else:
        st, label = "PASS", "OVERALL PASS"
    if st == "PASS":
        reasons = [f"{v['name']}: {v['detected']}/{v['reference']}" for v in en]
    else:
        reasons = [f"{v['name']}: {v['message']}" for v in err + fail + wait]
    return {"status": st, "label": label, "reasons": reasons, "enabled": len(en), "counts": counts}


def duplicate_warnings():
    """Same physical camera assigned to two enabled stations."""
    seen, out = {}, []
    for s in STATIONS:
        if s.cfg["enabled"] and s.cfg["camera"] != "video":
            seen.setdefault(s.cfg["camera"], []).append(s.name)
    for cam, names in seen.items():
        if len(names) > 1:
            out.append(f"Camera {cam} is assigned to {' and '.join(names)}. "
                       f"One camera cannot run two stations - choose different cameras.")
    return out


# ============================================================ camera detection
PROBE_CODE = ("import cv2,sys\n"
              "i=int(sys.argv[1]);nt=sys.argv[2]=='nt'\n"
              "cap=cv2.VideoCapture(i,cv2.CAP_DSHOW) if nt else cv2.VideoCapture(i)\n"
              "ok=False\n"
              "if cap.isOpened():\n"
              "    for _ in range(3):\n"
              "        r,f=cap.read()\n"
              "        if r and f is not None: ok=True;break\n"
              "cap.release();print('1' if ok else '0')\n")
_cam_cache = {"t": 0.0, "data": None}
_cam_lock = threading.Lock()


def scan_cameras(force=False):
    """Probe camera indexes 0-3 in a throw-away process (a hung driver cannot freeze the app).
    Cameras that a running station is using are reported 'in use' and are NOT opened again."""
    with _cam_lock:
        if not force and _cam_cache["data"] and time.time() - _cam_cache["t"] < 15:
            return _cam_cache["data"]
        out = []
        for i in range(MAX_STATIONS):
            owner = next((s for s in STATIONS if s.running and s.cfg["camera"] == i and not s.error), None)
            if owner:
                out.append({"index": i, "label": f"Camera {i}", "available": True, "in_use_by": owner.name})
                continue
            ok = False
            try:
                p = subprocess.run([sys.executable, "-c", PROBE_CODE, str(i), "nt" if os.name == "nt" else "posix"],
                                   capture_output=True, text=True, timeout=10,
                                   creationflags=0x08000000 if os.name == "nt" else 0)
                ok = p.stdout.strip().endswith("1")
            except Exception:
                ok = False
            out.append({"index": i, "label": f"Camera {i}", "available": ok, "in_use_by": None})
        _cam_cache.update(t=time.time(), data=out)
        return out


# ============================================================ routes
def get_station(sid):
    for s in STATIONS:
        if s.id == sid:
            return s
    abort(404, description="Unknown station")


def full_state():
    views = [s.view() for s in STATIONS]
    with events_lock:
        ev = list(events)
    return {"stations": views, "overall": compute_overall(views), "warnings": duplicate_warnings(),
            "events": ev, "camera": dict(settings["camera"]), "server_time": time.time()}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/status")
@app.route("/api/stations")
def api_stations():
    resp = jsonify(full_state())
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/stations/<int:sid>")
def api_station(sid):
    return jsonify(get_station(sid).view())


@app.route("/api/stations/<int:sid>/config", methods=["POST", "PUT", "PATCH"])
def api_station_config(sid):
    st = get_station(sid)
    v, errors = st.update_config(request.get_json(silent=True) or {})
    if errors:
        return jsonify(ok=False, station_id=sid, errors=errors, message="; ".join(errors)), 400
    return jsonify(ok=True, station_id=sid, station=v, warnings=duplicate_warnings(), message="Saved")


@app.route("/api/stations/<int:sid>/start", methods=["POST"])
def api_station_start(sid):
    st = get_station(sid)
    ok, msg = st.start()
    return jsonify(ok=ok, station_id=sid, message=msg, station=st.view()), (200 if ok else 409)


@app.route("/api/stations/<int:sid>/stop", methods=["POST"])
def api_station_stop(sid):
    st = get_station(sid)
    st.stop("stopped by operator")
    return jsonify(ok=True, station_id=sid, message=f"{st.name} stopped", station=st.view())


@app.route("/api/stations/start_all", methods=["POST"])
def api_start_all():
    res = []
    for s in STATIONS:
        if s.cfg["enabled"]:
            ok, msg = s.start()
            res.append({"station_id": s.id, "ok": ok, "message": msg})
    return jsonify(ok=all(r["ok"] for r in res), results=res)


@app.route("/api/stations/stop_all", methods=["POST"])
def api_stop_all():
    for s in STATIONS:
        s.stop("stopped by operator")
    return jsonify(ok=True)


@app.route("/api/stations/<int:sid>/video_feed")
def api_station_feed(sid):
    st = get_station(sid)
    if not st.running:
        abort(409, description="Station is not running")
    gen_id = st.gen

    def generate():
        seq = 0
        while st.running and st.gen == gen_id and not SHUTDOWN.is_set():
            with st.jpg_cond:
                st.jpg_cond.wait_for(lambda: st.jpg_seq != seq or st.gen != gen_id, timeout=1.0)
                if st.jpg_seq == seq or st.jpg is None:
                    continue
                seq, jpg = st.jpg_seq, st.jpg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"

    resp = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store"
    return resp


@app.route("/video_feed")                       # old URL kept: shows station 1
def video_feed():
    return api_station_feed(1)


@app.route("/api/overall")
def api_overall():
    return jsonify(compute_overall([s.view() for s in STATIONS]))

@app.route("/api/plc-output")  # New PLC output endpoint
def api_plc_output():
    views = [s.view() for s in STATIONS]
    enabled = [v for v in views if v["enabled"]]

    # Ready = 1 only when every enabled station is running
    # and has produced a valid PASS or FAIL result.
    ready = int(
        bool(enabled)
        and all(
            v["running"] and v["status"] in ("PASS", "FAIL")
            for v in enabled
        )
    )

    # Convert station results into binary outputs:
    # PASS = 1
    # FAIL or any other state = 0
    station_outputs = {
        f"station_{v['id']}": int(v["status"] == "PASS")
        for v in enabled
    }

    # Overall = 1 only when the system is ready
    # and every enabled station has passed.
    overall = int(
        ready == 1
        and bool(station_outputs)
        and all(value == 1 for value in station_outputs.values())
    )

    # Prepare the PLC response.
    response = {
        "ready": ready,
        "overall": overall
    }

    response.update(station_outputs)

    plc_response = jsonify(response)
    plc_response.headers["Cache-Control"] = "no-store"

    return plc_response

def plc_output_monitor():
    """Print PLC binary outputs only when a value changes."""
    previous_output = None

    while not SHUTDOWN.is_set():
        try:
            views = [s.view() for s in STATIONS]
            enabled = [v for v in views if v["enabled"]]

            ready = int(
                bool(enabled)
                and all(
                    v["running"] and v["status"] in ("PASS", "FAIL")
                    for v in enabled
                )
            )

            station_outputs = {
                f"station_{v['id']}": int(v["status"] == "PASS")
                for v in enabled
            }

            overall = int(
                ready == 1
                and bool(station_outputs)
                and all(value == 1 for value in station_outputs.values())
            )

            current_output = {
                "ready": ready,
                "overall": overall
            }

            current_output.update(station_outputs)

            # Print only when one of the PLC values changes.
            if current_output != previous_output:
                station_text = " | ".join(
                    "Station {} ({}): {}".format(
                        v["id"],
                        v["name"],
                        station_outputs.get("station_{}".format(v["id"]), 0)
                    )
                    for v in enabled
                )

                console.print(
                    f"[bold cyan]BINARY OUTPUT[/bold cyan] | "
                    f"Ready: [bold]{ready}[/bold] | "
                    f"{station_text} | "
                    f"Overall: [bold {'green' if overall == 1 else 'red'}]"
                    f"{overall}[/bold {'green' if overall == 1 else 'red'}]",
                    highlight=False
                )

                previous_output = current_output

        except Exception as e:
            log(f"PLC output monitor error: {e}", "red")

        SHUTDOWN.wait(0.2)

@app.route("/api/cameras")
def api_cameras():
    force = request.args.get("scan") == "1"
    if force or _cam_cache["data"]:
        cams = scan_cameras(force)
    else:
        cams = [{"index": i, "label": f"Camera {i}", "available": None, "in_use_by": None}
                for i in range(MAX_STATIONS)]
    return jsonify(cameras=cams, video_available=os.path.isfile(VIDEO_PATH), warnings=duplicate_warnings())


@app.route("/api/camera-settings", methods=["POST"])
def api_camera_settings():
    j = request.get_json(silent=True) or {}
    cs, reopen = settings["camera"], False
    try:
        for k, v in j.items():
            if k in CAM_LIMITS:
                lo, hi = CAM_LIMITS[k]
                v = max(lo, min(float(v), hi))
                v = int(v) if k in CAM_INT_KEYS else round(v, 2)
            elif k == "backend":
                if v not in ("auto", "dshow", "msmf"):
                    continue
            elif k in ("auto_exposure", "autofocus", "sw_enhance"):
                v = bool(v)
            else:
                continue
            if cs.get(k) != v:
                cs[k] = v
                reopen |= k in CAM_REOPEN_KEYS
        save_settings()
    except (TypeError, ValueError):
        return jsonify(ok=False, message="Invalid camera value"), 400
    if reopen:
        for s in STATIONS:
            if s.running and s.cfg["camera"] != "video":
                threading.Thread(target=s.restart_source, daemon=True).start()
    return jsonify(ok=True, camera=dict(cs), message="Camera settings updated")


@app.route("/api/camera-settings/reset", methods=["POST"])
def api_camera_reset():
    settings["camera"].update(CAM_DEFAULTS)
    save_settings()
    return jsonify(ok=True, camera=dict(settings["camera"]), message="Image settings reset")


@app.route("/api/health")
def health():
    views = [s.view() for s in STATIONS]
    return jsonify(
        ok=model is not None,
        model=os.path.relpath(MODEL_PATH, BASE_DIR) if MODEL_PATH else None,
        model_path=MODEL_PATH, classes={str(k): v for k, v in CLASS_NAMES.items()},
        classes_ok=CLASSES_OK, expected_classes={"0": "hex_bolt"},
        device="cuda" if USE_GPU else "cpu",
        active_stations=[v["id"] for v in views if v["enabled"] and v["running"]],
        stations=[{"station_id": v["id"], "name": v["name"], "enabled": v["enabled"], "running": v["running"],
                   "status": v["status"], "camera": v["camera"], "camera_fps": v["fps"],
                   "detection_fps": v["det_fps"], "error": v["error"] or None} for v in views],
        camera_errors=[{"station_id": v["id"], "name": v["name"], "error": v["error"] or v["message"]}
                       for v in views if v["enabled"] and v["status"] == "CAMERA_ERROR"],
        overall=compute_overall(views)["status"], detection=settings["detection"])


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/api/history")
def api_history():
    return jsonify(query_history(request.args.get("type"), min(int(request.args.get("limit", 200)), 1000)))


@app.route("/history/export.csv")
def history_csv():
    rows = query_history(request.args.get("type"), 100000)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ID", "Time", "Type", "Source", "Current", "Reference", "Missing", "Message", "Snapshot", "Part"])
    for r in rows:
        w.writerow([r["id"], r["ts"], r["type"], r["source"], r["current_count"], r["reference_count"],
                    r["missing_count"], r["message"], r["snapshot"] or "", r.get("part") or ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=inspection_history.csv"})


@app.route("/snapshots/<path:name>")
def snapshot_file(name):
    return send_from_directory(SNAP_DIR, name)


# ============================================================ startup / shutdown
_started = [False]


def shutdown_all(*_):
    """Release every camera. Safe to call more than once."""
    SHUTDOWN.set()
    for s in STATIONS:
        try:
            s.stop("shutdown")
        except Exception:
            pass


def init_app(autostart=True):
    if _started[0]:
        return
    _started[0] = True
    load_settings()
    db_init()
    STATIONS[:] = [Station(cfg) for cfg in settings["stations"]]
    log("\nInspection stations:", "bold")
    for s in STATIONS:
        c = s.cfg
        log(f"  Station {s.id}: {c['name']:<16} enabled={c['enabled']!s:<5} camera={c['camera']} "
            f"reference={c['reference']} conf={c['conf']}")
    for fn in (inference_worker, supervisor, plc_output_monitor):
        threading.Thread(target=_forever, args=(fn, fn.__name__), daemon=True, name=fn.__name__).start()
    atexit.register(shutdown_all)
    if autostart and settings.get("autostart", True):
        def _auto():
            for s in STATIONS:
                if s.cfg["enabled"] and not SHUTDOWN.is_set():
                    s.start()
                    SHUTDOWN.wait(1.0)                    # stagger camera opening (USB bandwidth)
        threading.Thread(target=_auto, daemon=True, name="autostart").start()


if __name__ == "__main__":
    init_model()
    init_app()
    for sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            try:
                signal.signal(getattr(signal, sig), lambda *a: (shutdown_all(), os._exit(0)))
            except Exception:
                pass
    log("[bold green]AI SYSTEM READY[/bold green]")
    log("[bold yellow]Dashboard:[/bold yellow] http://127.0.0.1:5000")
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
    finally:
        shutdown_all()
