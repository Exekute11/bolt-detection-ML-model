"""
Camera doctor: finds WHY the picture is low resolution / low FPS / blurry and what settings fix it.

    python tools/camera_test.py            # tests camera #0..#4
    python tools/camera_test.py --cam 1    # tests ONLY camera #1 (faster)

Close the dashboard (python app.py) first - only one program can use the camera at a time.
A driver that crashes is skipped and reported - the test always continues.
"""
import argparse, os, sys, time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=None, help="test only this camera number")
a = ap.parse_args()

BACKENDS = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF)] if os.name == "nt" else [("default", 0)]
SIZES = [(640, 480), (1280, 720), (1920, 1080)]
OUT = "camera_test"; os.makedirs(OUT, exist_ok=True)
print("OpenCV version:", cv2.__version__)


def fourcc(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") or "?"


def measure(cap, n=25):
    """real FPS + last frame. A driver error returns (0, None) instead of crashing the test."""
    try:
        for _ in range(3):
            cap.read()                                   # warm-up
        t0, got, last = time.time(), 0, None
        for _ in range(n):
            ok, f = cap.read()
            if ok:
                got += 1; last = f
        dt = time.time() - t0
        return (got / dt if dt > 0 else 0), last
    except cv2.error as e:
        return 0.0, None


def open_cam(idx, flag):
    try:
        cap = cv2.VideoCapture(idx, flag) if os.name == "nt" else cv2.VideoCapture(idx)
        return cap if cap.isOpened() else None
    except cv2.error:
        return None


print("\n=== STEP 1: which cameras exist, and what they really deliver (MJPG, 30 FPS requested) ===")
print(f"{'cam#':>4} {'driver':>7} {'asked':>10} {'got':>10} {'mode':>5} {'real FPS':>9}")
results = []
cams = [a.cam] if a.cam is not None else range(5)
for idx in cams:
    for bname, flag in BACKENDS:
        cap = open_cam(idx, flag)
        if cap is None:
            continue
        for (w, h) in SIZES:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); cap.set(cv2.CAP_PROP_FPS, 30)
                fps, f = measure(cap)
                if f is None:
                    print(f"{idx:>4} {bname:>7} {w}x{h:<5} {'-- driver error / no picture (skipped) --':>34}")
                    continue
                gh, gw = f.shape[:2]
                print(f"{idx:>4} {bname:>7} {w}x{h:<5} {gw}x{gh:<5} {fourcc(cap):>5} {fps:>9.1f}")
                results.append((idx, bname, flag, w, h, gw, gh, fps, fourcc(cap)))
                cv2.imwrite(os.path.join(OUT, f"cam{idx}_{bname}_{gw}x{gh}.jpg"), f)
            except cv2.error:
                print(f"{idx:>4} {bname:>7} {w}x{h:<5} {'-- driver crashed (skipped) --':>34}")
        cap.release()
if not results:
    sys.exit("\nNo camera could be opened. Close Teams/Zoom/Camera app, re-plug the USB camera and run again.")

print("\n=== SUMMARY PER CAMERA (best smooth mode) ===")
best_by_cam = {}
for r in results:
    idx, bname, flag, w, h, gw, gh, fps, fcc = r
    ok = fps >= 20
    key = (ok, gw * gh, fps)
    if idx not in best_by_cam or key > best_by_cam[idx][0]:
        best_by_cam[idx] = (key, r)
for idx, (key, r) in sorted(best_by_cam.items()):
    _, bname, _, _, _, gw, gh, fps, fcc = r
    verdict = "OK" if fps >= 20 else "SLOW"
    print(f"  Camera #{idx}: {gw}x{gh} @ {fps:.0f} FPS, {fcc}, driver {bname}  -> {verdict}")
slow = [i for i, (k, r) in best_by_cam.items() if r[7] < 20]
if slow:
    print("  (SLOW = under 20 FPS in every mode: use a USB 3 port, more light, close Teams/Zoom, or another camera)")

good = [r for r in results if r[5] >= 1280 and r[7] >= 20]
best = max(good or results, key=lambda r: (r[5] * r[6] if r[7] >= 20 else 0, r[7]))
idx, bname, flag, w, h, gw, gh, fps, fcc = best
print(f"\nBest overall: camera #{idx}, driver {bname}, {gw}x{gh} at {fps:.0f} FPS   (sample pictures saved in '{OUT}')")

print(f"\n=== STEP 2: exposure test on camera #{idx} (auto exposure often makes FPS drop and the picture blurry) ===")
rows = []
cap = open_cam(idx, flag)
try:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); cap.set(cv2.CAP_PROP_FPS, 30)
    ae_auto, ae_man = ((0.75, 0.25) if bname == "dshow" else (1, 0)) if os.name == "nt" else (3, 1)
    print(f"{'exposure':>10} {'real FPS':>9} {'brightness %':>13}")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae_auto)
    fps2, f = measure(cap, 30)
    if f is not None:
        b = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() / 2.55
        print(f"{'AUTO':>10} {fps2:>9.1f} {b:>13.0f}"); rows.append(("AUTO", fps2, b))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae_man)
    for e in (-4, -5, -6, -7, -8, -9):
        cap.set(cv2.CAP_PROP_EXPOSURE, e)
        time.sleep(0.4)
        fps2, f = measure(cap, 30)
        if f is None:
            continue
        b = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean() / 2.55
        print(f"{e:>10} {fps2:>9.1f} {b:>13.0f}"); rows.append((e, fps2, b))
except cv2.error:
    print("  (this driver does not allow exposure control - use the software sliders in the dashboard)")
finally:
    if cap is not None:
        cap.release()

print("\n=== WHAT TO DO ===")
aa = next((r for r in rows if r[0] == "AUTO"), None)
man = [r for r in rows if r[0] != "AUTO" and r[1] >= 24 and 25 <= r[2] <= 75]
if gw < 1280:
    print(f"* This camera/driver only gives {gw}x{gh}. Use an external USB 3 webcam, or another Camera # / Driver.")
if aa and aa[1] < 22 and man:
    e = max(man, key=lambda r: r[1])[0]
    print(f"* AUTO exposure makes the camera slow ({aa[1]:.0f} FPS) and blurry. In the dashboard press 'Sharp mode' "
          f"(manual exposure) and set exposure to about {e}. Add more light.")
elif aa and aa[1] >= 22:
    print("* Auto exposure is fine for FPS. If shaking looks 'liquid', turn OFF Windows Studio Effects (see below).")
if rows and not man:
    print("* The room is too dark for a fast shutter - add a bright lamp/LED light on the part. This fixes blur AND FPS.")
if any(r[1] == "msmf" for r in results) is False and os.name == "nt":
    print("* The 'msmf' driver gave no usable picture here: in the dashboard set Driver = DirectShow.")
print("* Windows: Settings > Bluetooth & devices > Cameras > your camera > turn OFF Automatic framing, Eye contact,")
print("  Background effects. These crop/zoom the picture (low resolution) and warp it when you move ('liquid').")
print("* Connect the camera to a blue USB 3 port directly (no hub). Close Teams, Zoom and the Windows Camera app.")
