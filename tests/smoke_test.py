"""
Smoke test for the multi-station backend. Needs NO cameras, NO GPU and NO real model:
a stub 'ultralytics' always detects 5 hex bolts, and BOLT_FAKE_CAMERA=ok makes camera_worker.py produce a synthetic picture.
Run:  python tests/smoke_test.py
The test works on a temporary COPY of the project - your settings.json / inspections.db are never touched.
"""
import os, shutil, sys, tempfile, time, types

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="boltmulti_")
for item in ("app.py", "camera_worker.py", "settings.json", "templates", "static", "videos"):
    s, d = os.path.join(SRC, item), os.path.join(TMP, item)
    shutil.copytree(s, d) if os.path.isdir(s) else shutil.copy(s, d)
os.makedirs(os.path.join(TMP, "models")); open(os.path.join(TMP, "models", "hex_bolt_best.pt"), "wb").write(b"stub")

class _T:  # tiny tensor-like
    def __init__(s, v): s.v = v
    def __getitem__(s, i): return s.v[i]
    def __float__(s): return float(s.v[0])
    def __int__(s): return int(s.v[0])
class _Box:
    def __init__(s, x, c): s.xyxy, s.conf, s.cls = _T([[x, 100, x + 60, 160]]), _T([c]), _T([0])
class _Res:
    def __init__(s, n): s.boxes = [_Box(40 + i * 90, 0.9) for i in range(n)]
class _YOLO:
    names = {0: "hex_bolt"}
    calls, inside = 0, 0
    overlap = False
    def __init__(s, path): pass
    def predict(s, img, **kw):
        _YOLO.inside += 1
        if _YOLO.inside > 1: _YOLO.overlap = True        # would mean two threads inside the model
        time.sleep(0.02); _YOLO.calls += 1; _YOLO.inside -= 1
        return [_Res(5)]
stub = types.ModuleType("ultralytics"); stub.YOLO = _YOLO; sys.modules["ultralytics"] = stub

os.environ["BOLT_FAKE_CAMERA"] = "ok"
sys.path.insert(0, TMP)
import app as A
A.init_model(); A.init_app(autostart=False)
c = A.app.test_client()
ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(("PASS  " if cond else "FAIL  ") + name + (f"  [{extra}]" if extra and not cond else ""))
def state(): return c.get("/api/stations").get_json()
def wait(fn, t=15):
    t0 = time.time()
    while time.time() - t0 < t:
        if fn(): return True
        time.sleep(0.2)
    return False

h = c.get("/api/health").get_json()
check("health reports models/hex_bolt_best.pt", h["model"].replace("\\", "/") == "models/hex_bolt_best.pt", h["model"])
check("health classes == {0: hex_bolt}", h["classes"] == {"0": "hex_bolt"} and h["classes_ok"])
check("station defaults (Front Shield 5 / End Shield 6, 3+4 disabled)",
      [(s["name"], s["reference"], s["enabled"]) for s in state()["stations"]] ==
      [("Front Shield", 5, True), ("End Shield", 6, True), ("Inspection 3", 0, False), ("Inspection 4", 0, False)])
check("nothing runs before start (no cameras opened)", not any(s["running"] for s in state()["stations"]))

# --- 2 cameras
print(c.post("/api/stations/1/start").get_json()["message"], "|", c.post("/api/stations/2/start").get_json()["message"])
check("both stations reach a verdict", wait(lambda: all(s["status"] in ("PASS", "FAIL") for s in state()["stations"][:2])))
st = state()
s1, s2 = st["stations"][0], st["stations"][1]
check("Front Shield 5/5 -> PASS", s1["status"] == "PASS" and s1["detected"] == 5 and s1["missing"] == 0, s1)
check("End Shield 5/6 -> FAIL, 1 missing", s2["status"] == "FAIL" and s2["missing"] == 1 and s2["extra"] == 0, s2)
check("overall FAIL with reason", st["overall"]["status"] == "FAIL" and "End Shield" in st["overall"]["reasons"][0] and "1 missing" in st["overall"]["reasons"][0], st["overall"])
check("counts are per station (ids differ)", s1["station_id"] == 1 and s2["station_id"] == 2)

# --- extra bolts
c.post("/api/stations/2/config", json={"reference": 4})
check("reference 4 vs 5 detected -> FAIL, 1 extra", wait(lambda: state()["stations"][1]["extra"] == 1))
c.post("/api/stations/2/config", json={"reference": 5})
check("reference 5 -> PASS, overall PASS", wait(lambda: state()["overall"]["status"] == "PASS"))
c.post("/api/stations/2/config", json={"reference": 6})

# --- feed + zone
r = c.get("/api/stations/1/video_feed", buffered=False)
chunk = next(iter(r.response))
check("MJPEG feed delivers a JPEG", b"image/jpeg" in chunk and b"\xff\xd8" in chunk)
r.close()
check("ROI outside all boxes -> count 0", c.post("/api/stations/1/config", json={"roi": [0.0, 0.8, 0.2, 0.95]}).get_json()["ok"]
      and wait(lambda: state()["stations"][0]["detected"] == 0))
c.post("/api/stations/1/config", json={"roi": None})
check("ROI cleared -> 5 again", wait(lambda: state()["stations"][0]["detected"] == 5))
check("invalid config rejected", c.post("/api/stations/1/config", json={"reference": "abc"}).status_code == 400)

# --- duplicate camera
d = c.post("/api/stations/2/config", json={"camera": 0}).get_json()
check("duplicate camera warning", any("Camera 0" in w for w in d["warnings"]), d)
check("duplicate camera cannot be started twice", not c.post("/api/stations/2/start").get_json()["ok"] or state()["stations"][1]["status"] in ("CAMERA_ERROR",))
c.post("/api/stations/2/config", json={"camera": 1})
c.post("/api/stations/2/start")

# --- disable releases camera, overall ignores disabled
proc2 = A.STATIONS[1].cp
c.post("/api/stations/2/config", json={"enabled": False})
check("disable stops station + releases camera process", not A.STATIONS[1].running and (proc2 is None or not proc2.alive()))
s2 = state()["stations"][1]
check("disabled station shows DISABLED", s2["status"] == "DISABLED")
check("overall ignores disabled station -> PASS", wait(lambda: state()["overall"]["status"] == "PASS"), state()["overall"])

# --- camera error never passes
c.post("/api/stations/3/config", json={"enabled": True, "camera": 2, "reference": 0})
os.environ["BOLT_FAKE_CAMERA"] = "fail"
c.post("/api/stations/3/start")
check("unavailable camera -> CAMERA_ERROR (not PASS, even with reference 0)", wait(lambda: state()["stations"][2]["status"] == "CAMERA_ERROR"), state()["stations"][2])
check("overall CAMERA ERROR", state()["overall"]["status"] == "CAMERA_ERROR")
check("health lists the camera error", len(c.get("/api/health").get_json()["camera_errors"]) == 1)
os.environ["BOLT_FAKE_CAMERA"] = "ok"
check("camera recovers automatically (auto-retry)", wait(lambda: state()["stations"][2]["status"] in ("PASS", "FAIL", "WAITING"), 30))
c.post("/api/stations/3/stop")
c.post("/api/stations/3/config", json={"enabled": False})

# --- video source + 4th station
c.post("/api/stations/4/config", json={"enabled": True, "camera": "video", "reference": 5, "name": "Custom Part B"})
c.post("/api/stations/4/start")
check("video-file station works + custom name", wait(lambda: state()["stations"][3]["status"] == "PASS") and state()["stations"][3]["name"] == "Custom Part B", state()["stations"][3])

check("model was never entered by two threads at once", not _YOLO.overlap)
check("history DB records written", len(c.get("/api/history").get_json()) >= 2)
check("index + history pages render", c.get("/").status_code == 200 and c.get("/history").status_code == 200)
cfg = open(os.path.join(TMP, "settings.json"), encoding="utf-8").read()
check("settings persisted", '"Custom Part B"' in cfg)

procs = [s.cp for s in A.STATIONS if s.cp]
A.shutdown_all()
time.sleep(0.5)
check("shutdown released every camera", all(not s.running for s in A.STATIONS) and all(not p.alive() for p in procs))
shutil.rmtree(TMP, ignore_errors=True)
print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED"); sys.exit(0 if ok else 1)
