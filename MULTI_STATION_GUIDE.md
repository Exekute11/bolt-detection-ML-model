# Multi-Station Inspection (up to 4 stations)

## Install & run (Windows, existing virtual environment)
```
cd BoltDetectionAI_Final
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 . Quick self-test without cameras/model: `python tests/smoke_test.py`

Startup prints `USING MODEL: ...\models\hex_bolt_best.pt` and `MODEL CLASSES: {0: 'hex_bolt'}`.
Check `http://127.0.0.1:5000/api/health` -> `model`, `classes`, `classes_ok`, `active_stations`, `camera_errors`.

## Assign Front Shield = 5, End Shield = 6
These are the defaults. To change anything: edit the station card on the dashboard
(name box, **Reference** box, **Camera / source** drop-down). Changes save instantly to `settings.json`.
Station 1: name "Front Shield", Camera 0, Reference 5 · Station 2: "End Shield", Camera 1, Reference 6.
Stations 3 and 4 are disabled by default - flip the switch on the card to use them.

## Architecture
* One YOLO model, one inference worker thread (round-robin over active stations) + `MODEL_LOCK` -> never two threads inside the model.
* Each station = own config, camera process (`camera_worker.py`) or video thread, newest frame, boxes, count history,
  stable count, status, error and MJPEG stream (`/api/stations/<id>/video_feed`). Nothing is shared.
* Stable count: PASS/FAIL appears only after the count was identical for `smooth` consecutive detections (default 5);
  a one-frame miss never changes the verdict. Extra YOLO NMS + IoU de-duplication prevents double counting.
* Status: DISABLED / STOPPED / WAITING / PASS / FAIL / CAMERA ERROR. No frame, stale frame or failed camera can never show PASS.
* Overall: CAMERA ERROR > FAIL > WAITING > PASS; disabled stations ignored.
* Stop/disable kills the station's camera process -> device released. Shutdown (Ctrl+C / close) releases all cameras.
* Failed cameras auto-retry every 5-20 s. They do NOT fall back to the demo video any more (it would hide a missing camera).

## API
| Method | Route | Purpose |
|---|---|---|
| GET | `/api/stations` (`/status`) | all stations + overall + events + warnings |
| GET | `/api/stations/<id>` | one station |
| POST | `/api/stations/<id>/config` | `{name, enabled, camera (0-9 or "video"), reference, conf, smooth, roi}` |
| POST | `/api/stations/<id>/start` · `/stop` | start / stop one station |
| POST | `/api/stations/start_all` · `/stop_all` | all enabled / all |
| GET | `/api/stations/<id>/video_feed` | MJPEG for that station |
| GET | `/api/overall` | overall result |
| GET | `/api/cameras?scan=1` | detect cameras 0-3 (throw-away probe process) |
| GET | `/api/health` | model path, classes, active stations, FPS, camera errors |
| POST | `/api/camera-settings` | resolution / fps / driver / image correction (all stations) |

## Test checklist
**1 camera** - enable only Station 1 (others off), Start. Card turns PASS at 5 bolts; cover one bolt -> FAIL "1 missing hex bolt"; overall mirrors it. Stop -> camera LED off.
**2 cameras** - Station 1 = Camera 0 / ref 5, Station 2 = Camera 1 / ref 6. Each card counts independently; 5/6 on End Shield -> FAIL, overall FAIL with reason "End Shield: 1 missing hex bolt".
**3 cameras** - enable Station 3, Camera 2, set its own name/reference. Unplug that USB cable -> only that card shows CAMERA ERROR (purple); overall CAMERA ERROR; replug -> recovers automatically.
**4 cameras** - enable Station 4, Camera 3. Watch Task Manager CPU; if too high lower resolution to 640x480 or raise smoothing. Disable a station -> its camera is released and CPU drops.
Also check: select the same camera in two stations -> amber warning + second one refuses to start; Set Zone on each card works independently; restart the app -> names/references/zones/cameras are kept.

## What changed / removed (and why)
* Replaced: `app.py`, `templates/index.html`, `static/js/app.js`, new `static/js/stations.js`, new `static/css/stations.css`, `settings.json`, new `tests/smoke_test.py`.
* Unchanged: `camera_worker.py`, `templates/history.html`, `style.css`, `anim.js`, `inspections.db` (history keeps working; the "Part" column now shows the station name), snapshots, runs/, tools/.
* Removed from the live dashboard (single global pipeline cannot coexist with per-station cameras): global Capture/Clear Reference, Timed Inspection Cycle, Part library, Video/Webcam toggle (video is now a per-station source), tracker IDs (`model.track` keeps ONE tracker state, which would mix stations; `model.predict` is used). Old code is kept in `legacy/` (not run by default).
* Model: only `models/hex_bolt_best.pt` is loaded. Old `runs/**/best.pt` fallbacks were removed. `BOLT_MODEL=<path>` env var still overrides, with a loud warning.
