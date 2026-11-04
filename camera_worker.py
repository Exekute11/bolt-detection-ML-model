"""
Isolated CAMERA PROCESS (started and supervised by app.py).

Why a separate process? Camera drivers (DirectShow / Media Foundation) can hang or even crash the whole program.
Here a driver problem can only kill THIS process - app.py notices, kills it if needed, and carries on
(falls back to the video file or reconnects). The dashboard never stops.

Protocol on stdout (binary):  [1 byte type][4 byte length][payload]
   type F = JPEG frame      type S = JSON status message
Commands on stdin (JSON lines):  {"cmd":"hw", ...}  or  {"cmd":"stop"}.  When stdin closes the process exits.
Test switch (no camera needed): environment variable BOLT_FAKE_CAMERA = ok | slow | fail | crash | hang
"""
import json, os, struct, sys, threading, time

import cv2
import numpy as np

cfg = json.loads(sys.argv[1])
OUT = sys.stdout.buffer
_wl = threading.Lock()
STOP = threading.Event()
HW = {"pending": None}
NT = os.name == "nt"
FAKE = os.environ.get("BOLT_FAKE_CAMERA", "")


def send(kind, payload):
    with _wl:
        OUT.write(kind + struct.pack("<I", len(payload)) + payload)
        OUT.flush()


def status(**kw):
    send(b"S", json.dumps(kw).encode())


def _stdin_loop():
    try:
        for line in sys.stdin:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("cmd") == "stop":
                break
            if m.get("cmd") == "hw":
                HW["pending"] = m
    finally:
        STOP.set()                                   # parent closed the pipe / asked us to stop


threading.Thread(target=_stdin_loop, daemon=True).start()


class FakeCap:
    """Synthetic camera for testing the pipeline without hardware."""
    def __init__(s):
        s.w, s.h, s.t = 640, 480, 0
    def isOpened(s): return True
    def release(s): pass
    def set(s, p, v):
        if p == cv2.CAP_PROP_FRAME_WIDTH: s.w = int(v)
        elif p == cv2.CAP_PROP_FRAME_HEIGHT: s.h = int(v)
        return True
    def _fps(s): return 5 if (FAKE == "slow" and s.w > 1280) else 30
    def get(s, p):
        if p == cv2.CAP_PROP_FPS: return s._fps()
        if p == cv2.CAP_PROP_FOURCC:
            return cv2.VideoWriter_fourcc(*("YUY2" if (FAKE == "slow" and s.w > 1280) else "MJPG"))
        return 0
    def read(s):
        time.sleep(1 / s._fps()); s.t += 1
        img = np.full((s.h, s.w, 3), 90, np.uint8)
        x = int((s.t * 9) % max(s.w - 120, 1))
        cv2.rectangle(img, (x, 60), (x + 120, 180), (60, 205, 98), -1)
        return True, img


def flag(b):
    return {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}.get(b, cv2.CAP_DSHOW)


def open_cam(idx, b):
    if FAKE:
        return FakeCap()
    try:
        cap = cv2.VideoCapture(idx, flag(b)) if NT else cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap
        cap.release()
    except cv2.error:
        pass
    return None


def rd(cap):
    try:
        ok, f = cap.read()
        return bool(ok and f is not None), f
    except cv2.error:
        return False, None


def fcc(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") or "?"


def set_mode(cap, w, h, f, order):
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    try:
        if order == 0:
            cap.set(cv2.CAP_PROP_FOURCC, mjpg)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        cap.set(cv2.CAP_PROP_FPS, f)
    except cv2.error:
        pass


def apply_hw(cap, c, backend, initial=False):
    if initial and c.get("auto_exposure", True) and c.get("autofocus", True) and c.get("zoom") is None:
        return                                        # camera defaults are already automatic
    ae_auto, ae_man = ((0.75, 0.25) if backend == "dshow" else (1, 0)) if NT else (3, 1)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae_auto if c.get("auto_exposure", True) else ae_man)
        if not c.get("auto_exposure", True):
            cap.set(cv2.CAP_PROP_EXPOSURE, c.get("exposure", -6))
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if c.get("autofocus", True) else 0)
        if not c.get("autofocus", True):
            cap.set(cv2.CAP_PROP_FOCUS, c.get("focus", 0))
        if c.get("zoom") is not None:
            cap.set(cv2.CAP_PROP_ZOOM, c["zoom"])
    except cv2.error:
        pass


def measure(cap, seconds=0.8, n=10):
    for _ in range(2):
        rd(cap)
    t0, got, last = time.time(), 0, None
    while got < n and time.time() - t0 < seconds:
        ok, f = rd(cap)
        if ok:
            got += 1; last = f
    dt = time.time() - t0
    return (got / dt if got and dt > 0 else 0.0), last


def send_frame(fr):
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        send(b"F", buf.tobytes())


def deep_search(cap, backend, W, H, F):
    """Camera is laggy in the asked mode: find the best smooth one (30 FPS first, then 720p, then VGA)."""
    cands = []
    if F > 30: cands.append((W, H, 30))
    if W * H > 1280 * 720: cands.append((1280, 720, 30))
    cands.append((640, 480, 30))
    cands = [m for m in cands if m[0] * m[1] <= W * H] or [(640, 480, 30)]
    best, t0 = None, time.time()
    for i, (w, h, f) in enumerate(cands):
        if best and time.time() - t0 > 12:
            break
        for order in ((0, 1) if i == 0 else (0,)):
            status(t="open", msg=f"Testing {backend} {w}x{h} @{f} FPS…")
            set_mode(cap, w, h, f, order)
            fps, fr = measure(cap)
            if fr is None:
                continue
            gh, gw = fr.shape[:2]
            smooth = fps >= min(0.8 * f, 24) and gw >= w * 0.95
            cur = (smooth, gw * gh if fps >= 15 else 0, fps)
            if best is None or cur > best[0]:
                best = (cur, (w, h, f), gw, gh, fps, fcc(cap))
            if smooth or fcc(cap) == "MJPG":
                break
        if best and best[0][0]:
            break
    return best


def main():
    idx, W, H, F = int(cfg["index"]), int(cfg["width"]), int(cfg["height"]), int(cfg["fps"])
    pref, hint = cfg.get("backend", "auto"), cfg.get("mode_hint")
    order = ["dshow", "msmf"] if (pref == "auto" and NT) else [pref if pref != "auto" else "dshow"]
    mode = (W, H, F)
    if hint:                                          # a mode that was found smooth before: use it directly
        mode = tuple(hint[:3])
        if hint[3] in order:
            order = [hint[3]] + [b for b in order if b != hint[3]]
    if FAKE == "fail":
        status(t="error", msg="Camera could not be opened (test mode: fail)")
        return
    cap, backend, frame = None, None, None
    for b in order:
        status(t="open", msg=f"Opening camera #{idx} ({b})…")
        cap = open_cam(idx, b)
        if cap is None:
            continue
        set_mode(cap, mode[0], mode[1], mode[2], 0)
        ok, frame = rd(cap)
        if ok:
            backend = b
            break
        cap.release(); cap = None
    if cap is None:
        status(t="error", msg=f"Camera #{idx} could not be opened. Close other apps using it (Teams, Zoom, Camera app) "
                              f"or try another Camera # / Driver.")
        return
    cur = fcc(cap)
    if cur != "MJPG" and mode[0] >= 1280:            # some drivers need the other property order
        set_mode(cap, mode[0], mode[1], mode[2], 1)
        ok2, f2 = rd(cap)
        if ok2:
            frame, cur = f2, fcc(cap)
    apply_hw(cap, cfg, backend, initial=True)
    h, w = frame.shape[:2]
    try:
        zoom = int(cap.get(cv2.CAP_PROP_ZOOM))
    except Exception:
        zoom = -1
    info = dict(width=w, height=h, fps=round(cap.get(cv2.CAP_PROP_FPS), 1), fourcc=cur,
                backend=backend if NT else "default", index=idx, note="", zoom=zoom)
    status(t="ready", info=info, from_cache=bool(hint))
    send_frame(frame)
    if FAKE == "crash":
        threading.Timer(1.5, lambda: os._exit(3)).start()
    if FAKE == "hang":
        time.sleep(1e9)

    t_ready, n, fails, checked = time.time(), 0, 0, bool(hint)
    while not STOP.is_set():
        if HW["pending"]:
            apply_hw(cap, HW["pending"], backend, initial=False)
            HW["pending"] = None
        ok, fr = rd(cap)
        if not ok:
            fails += 1
            time.sleep(0.03)
            if fails > 100:
                status(t="error", msg="Camera stream lost")
                return
            continue
        fails, n = 0, n + 1
        send_frame(fr)
        if not checked and (n >= 25 or time.time() - t_ready > 2):
            checked = True
            fps_now = n / max(time.time() - t_ready, 1e-3)
            if fps_now < 12 and W * H > 640 * 480:
                status(t="slow", msg="Camera is slow in this mode - finding a smooth mode…")
                best = deep_search(cap, backend, W, H, F)
                if best:
                    _, (bw, bh, bf), gw, gh, bfps, bfcc = best
                    set_mode(cap, bw, bh, bf, 0)
                    note = ""
                    if (gw, gh) != (W, H) or bfps < 0.8 * F:
                        note = (f"Asked {W}x{H} @{F}, but this camera cannot do that smoothly - "
                                f"using {gw}x{gh} @{round(bfps)} FPS ({bfcc}).")
                    if not best[0][0]:
                        note = (f"Camera only gives {round(bfps)} FPS ({bfcc}). Try another Camera # / Driver, "
                                f"a USB 3 port, or more light.")
                    status(t="info", mode=[bw, bh, bf, backend], note=note,
                           info=dict(width=gw, height=gh, fps=round(bfps, 1), fourcc=bfcc,
                                     backend=backend if NT else "default", index=idx, note=note))
                else:
                    status(t="info", note="", info=info)


try:
    main()
except (BrokenPipeError, OSError):
    pass
except Exception as e:                                # never leave a silent hang: tell the parent
    try:
        status(t="error", msg=f"Camera process error: {e}")
    except Exception:
        pass
finally:
    os._exit(0)
