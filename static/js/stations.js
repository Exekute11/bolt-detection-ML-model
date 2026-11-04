/* Station cards: build once, then update in place. Every card talks only to /api/stations/<id>/... */
(function () {
  const grid = $('#stationGrid');
  const cards = {};
  const LABEL = { PASS: 'PASS', FAIL: 'FAIL', WAITING: 'WAITING', CAMERA_ERROR: 'CAMERA ERROR', STOPPED: 'STOPPED', DISABLED: 'DISABLED' };
  let camInfo = [null, null, null, null], videoOk = true;

  function camOptions() {
    let h = '';
    for (let i = 0; i < 4; i++) {
      const c = camInfo[i];
      let t = 'Camera ' + i;
      if (c) t += c.in_use_by ? ' (in use: ' + c.in_use_by + ')' : c.available === false ? ' (not found)' : c.available ? ' ✓' : '';
      h += `<option value="${i}">${esc(t)}</option>`;
    }
    return h + `<option value="video">Video file (demo)${videoOk ? '' : ' - missing'}</option>`;
  }

  function build(s) {
    const el = document.createElement('article');
    el.className = 'scard DISABLED';
    el.dataset.id = s.id;
    el.innerHTML = `
      <header class="sc-head">
        <span class="sc-num">Station ${s.id}</span>
        <input class="sc-name" maxlength="40" title="Inspection name (editable)">
        <label class="switch" title="Enable / disable station"><input type="checkbox" class="sc-en"><i></i></label>
        <span class="sbadge">--</span>
      </header>
      <div class="sc-stage">
        <img class="sc-img" alt="Station ${s.id} live feed">
        <div class="sc-ov"><b></b><small></small></div>
        <div class="zone-layer"><div class="zone-hint">Drag around the area that detections should ignore · Esc cancels</div><div class="zone-rect"></div></div>
      </div>
      <div class="sc-metrics">
        <div class="sm"><span>Detected</span><b class="m-det">--</b></div>
        <div class="sm"><span>Reference</span><input type="number" class="sc-ref" min="0" max="999"></div>
        <div class="sm m-miss"><span>Missing</span><b>--</b></div>
        <div class="sm m-extra"><span>Extra</span><b>--</b></div>
      </div>
      <div class="sc-msg"></div>
      <div class="sc-cfg">
        <label>Camera / source<select class="sc-cam"></select></label>
        <label>Confidence <b class="sc-confv"></b><input type="range" class="sc-conf" min="0.05" max="0.95" step="0.05"></label>
        <label>Smoothing<input type="number" class="sc-smooth" min="1" max="30" title="Frames the count must stay equal before PASS/FAIL"></label>
      </div>
      <div class="sc-btns">
        <button class="btn primary b-start"><svg viewBox="0 0 24 24"><path d="M7 4v16l13-8z"/></svg>Start</button>
        <button class="btn b-stop"><svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>Stop</button>
        <button class="btn ghost b-zone">Draw Exclusion Zone</button>
        <button class="btn ghost b-zoneclr">Clear Exclusion Zone</button>
        <button class="btn ghost b-fs">Fullscreen</button>
        <span class="sc-fps"></span>
      </div>`;
    grid.appendChild(el);
    const q = sel => el.querySelector(sel);
    const c = { el, id: s.id, img: q('.sc-img'), ov: q('.sc-ov'), stage: q('.sc-stage'), layer: q('.zone-layer'), rect: q('.zone-rect'),
      name: q('.sc-name'), en: q('.sc-en'), badge: q('.sbadge'), det: q('.m-det'), ref: q('.sc-ref'), miss: q('.m-miss'), extra: q('.m-extra'),
      msg: q('.sc-msg'), cam: q('.sc-cam'), conf: q('.sc-conf'), confv: q('.sc-confv'), smooth: q('.sc-smooth'),
      start: q('.b-start'), stop: q('.b-stop'), zone: q('.b-zone'), zoneclr: q('.b-zoneclr'), fps: q('.sc-fps'), gen: -1 };
    cards[s.id] = c;
    c.cam.innerHTML = camOptions();
    wire(c);
    return c;
  }

  async function save(c, patch, okMsg) {
    const d = await api(`/api/stations/${c.id}/config`, patch);
    if (!d.ok) { toast(d.message || 'Could not save'); return d; }
    if (okMsg) toast(okMsg);
    return d;
  }

  function wire(c) {
    c.name.onchange = () => save(c, { name: c.name.value }, 'Name saved').then(d => { if (!d.ok) c.name.value = ''; });
    c.ref.onchange = () => save(c, { reference: c.ref.value }, `Reference set to ${c.ref.value}`);
    c.smooth.onchange = () => save(c, { smooth: c.smooth.value }, 'Smoothing saved');
    c.conf.oninput = () => { c.confv.textContent = (+c.conf.value).toFixed(2); c.conf.dataset.drag = '1'; };
    c.conf.onchange = () => save(c, { conf: c.conf.value }, 'Confidence saved').then(() => { c.conf.dataset.drag = ''; });
    c.cam.onchange = () => save(c, { camera: c.cam.value }, 'Camera assigned');
    c.en.onchange = async () => {
      const d = await save(c, { enabled: c.en.checked });
      if (d.ok && c.en.checked) { const r = await api(`/api/stations/${c.id}/start`, {}); toast(r.message); }
      else if (d.ok) toast('Station disabled - camera released');
    };
    c.start.onclick = async () => { const r = await api(`/api/stations/${c.id}/start`, {}); toast(r.message || 'Started'); };
    c.stop.onclick = async () => { const r = await api(`/api/stations/${c.id}/stop`, {}); toast(r.message || 'Stopped'); };
    c.img.onerror = () => { if (c.gen >= 0) setTimeout(() => { c.gen = -1; }, 1500); };   /* re-attach the stream on next poll */
    c.el.querySelector('.b-fs').onclick = () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (c.stage.requestFullscreen) c.stage.requestFullscreen();
    };
    /* ---- per-station detection zone ---- */
    let drawing = false, st = null;
    const norm = e => { const r = c.stage.getBoundingClientRect();
      return { x: Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1), y: Math.min(Math.max((e.clientY - r.top) / r.height, 0), 1) }; };
    const end = () => { c.layer.classList.remove('on'); c.rect.style.display = 'none'; drawing = false; st = null; };
    c.zone.onclick = () => {
  c.layer.classList.add('on');
  toast(`Draw the exclusion zone for ${c.name.value}`);
};

c.zoneclr.onclick = () => {
  save(c, { roi: null }, 'Exclusion zone cleared');
};
    c.layer.onpointerdown = e => { st = norm(e); drawing = true; c.rect.style.display = 'block'; c.layer.setPointerCapture(e.pointerId); e.preventDefault(); };
    c.layer.onpointermove = e => {
      if (!drawing) return; const p = norm(e);
      Object.assign(c.rect.style, { left: Math.min(st.x, p.x) * 100 + '%', top: Math.min(st.y, p.y) * 100 + '%',
        width: Math.abs(p.x - st.x) * 100 + '%', height: Math.abs(p.y - st.y) * 100 + '%' });
    };
    c.layer.onpointerup = e => {
      if (!drawing) return; const p = norm(e), s0 = st; end();
      save(
  c,
  {
    roi: {
      x1: s0.x,
      y1: s0.y,
      x2: p.x,
      y2: p.y
    }
  },
  'Exclusion zone saved for ' + c.name.value
);
    };
    document.addEventListener('keydown', e => { if (e.key === 'Escape') end(); });
  }

  const idle = el => document.activeElement !== el;

  function update(c, s, dupCams) {
    c.el.className = 'scard ' + s.status;
    c.badge.className = 'sbadge ' + s.status;
    c.badge.textContent = LABEL[s.status] || s.status;
    if (idle(c.name)) c.name.value = s.name;
    c.en.checked = s.enabled;
    if (idle(c.ref)) c.ref.value = s.reference;
    if (idle(c.smooth)) c.smooth.value = s.smooth;
    if (!c.conf.dataset.drag) { c.conf.value = s.conf; c.confv.textContent = (+s.conf).toFixed(2); }
    if (idle(c.cam)) c.cam.value = String(s.camera);
    c.cam.classList.toggle('dup', dupCams.has(String(s.camera)));
    c.det.textContent = s.detected === null || s.detected === undefined ? '--' : s.detected;
    c.miss.querySelector('b').textContent = s.missing === null ? '--' : s.missing;
    c.extra.querySelector('b').textContent = s.extra === null ? '--' : s.extra;
    c.miss.classList.toggle('bad', s.missing > 0);
    c.extra.classList.toggle('bad', s.extra > 0);
    c.msg.className = 'sc-msg ' + s.status;
    c.msg.textContent = s.message;
    c.start.disabled = !s.enabled || s.running;
    c.stop.disabled = !s.running;
    c.zone.classList.toggle('on', !!s.roi);
    c.zoneclr.disabled = !s.roi;
    c.fps.textContent = s.running ? `cam ${s.fps} fps · AI ${s.det_fps} fps` : '';
    if (s.frame_w && s.frame_h) {
      const ar = s.frame_w + ' / ' + s.frame_h;
      if (c.stage.dataset.ar !== ar) { c.stage.dataset.ar = ar; c.stage.style.aspectRatio = ar; }
    }
    /* stream: attach on (re)start, detach when stopped / disabled so the camera is truly idle */
    if (s.running && c.gen !== s.gen) { c.gen = s.gen; c.img.src = s.feed_url + '?g=' + s.gen + '&t=' + Date.now(); }
    if (!s.running && c.gen !== -2) { c.gen = -2; c.img.removeAttribute('src'); }
    /* overlay text when there is no live picture */
    const showOv = !s.running || s.status === 'CAMERA_ERROR' || s.status === 'DISABLED' || (s.status === 'WAITING' && !s.live_count && s.live_count !== 0);
    c.ov.classList.toggle('hide', !showOv);
    c.ov.className = 'sc-ov ' + s.status + (showOv ? '' : ' hide');
    if (showOv) {
      c.ov.querySelector('b').textContent = s.status === 'CAMERA_ERROR' ? 'CAMERA ERROR' : s.status === 'DISABLED' ? 'Station disabled'
        : s.running ? (s.opening || 'Opening source…') : 'Stopped';
      c.ov.querySelector('small').textContent = s.status === 'CAMERA_ERROR' ? s.message
        : s.status === 'DISABLED' ? 'Turn the switch on to use this station' : s.running ? '' : 'Press Start';
    }
  }

  window.stationsRender = function (d) {
    const dup = new Set();
    const seen = {};
    d.stations.forEach(s => { if (s.enabled && s.camera !== 'video') { (seen[s.camera] = seen[s.camera] || []).push(s.id); } });
    Object.keys(seen).forEach(k => { if (seen[k].length > 1) dup.add(k); });
    d.stations.forEach(s => { update(cards[s.id] || build(s), s, dup); });
    renderCamera(d.camera);
  };

  /* ---- toolbar ---- */
  $('#btnStartAll').onclick = async () => { const r = await api('/api/stations/start_all', {}); toast(r.ok ? 'All enabled stations started' : 'Some stations could not start - see messages'); };
  $('#btnStopAll').onclick = async () => { await api('/api/stations/stop_all', {}); toast('All stations stopped - cameras released'); };
  $('#btnScan').onclick = async () => {
    $('#scanNote').textContent = 'Scanning cameras… (up to ~30 s)';
    $('#btnScan').disabled = true;
    const d = await api('/api/cameras?scan=1');
    $('#btnScan').disabled = false;
    if (d.cameras) {
      camInfo = d.cameras; videoOk = d.video_available;
      Object.values(cards).forEach(c => { const v = c.cam.value; c.cam.innerHTML = camOptions(); c.cam.value = v; });
      $('#scanNote').textContent = 'Found: ' + (d.cameras.filter(x => x.available).map(x => x.index).join(', ') || 'none');
    } else $('#scanNote').textContent = 'Scan failed';
  };

  /* ---- global Camera & Image panel ---- */
  const busy = {};
  function setv(id, v) { const e = $(id); if (e && !busy[id] && document.activeElement !== e) e.value = v; }
  function renderCamera(c) {
    if (!c) return;
    setv('#camRes', c.width + 'x' + c.height); setv('#camFps', String(c.fps)); setv('#camDrv', c.backend);
    setv('#swGamma', c.sw_gamma); setv('#swBri', c.sw_brightness); setv('#swCon', c.sw_contrast);
    $('#vGamma').textContent = c.sw_gamma; $('#vBri').textContent = c.sw_brightness; $('#vCon').textContent = c.sw_contrast;
    $('#swEnh').checked = c.sw_enhance;
  }
  const send = (b, m) => api('/api/camera-settings', b).then(d => { if (m) toast(m); });
  const timers = {};
  function slider(id, key) {
    const e = $(id);
    e.oninput = () => { busy[id] = true; $(id === '#swGamma' ? '#vGamma' : id === '#swBri' ? '#vBri' : '#vCon').textContent = e.value;
      clearTimeout(timers[id]); timers[id] = setTimeout(() => send({ [key]: +e.value }), 150); };
    e.onchange = () => setTimeout(() => { busy[id] = false; }, 500);
  }
  slider('#swGamma', 'sw_gamma'); slider('#swBri', 'sw_brightness'); slider('#swCon', 'sw_contrast');
  $('#camRes').onchange = e => { const [w, h] = e.target.value.split('x').map(Number); send({ width: w, height: h }, `Applying ${w}×${h} - restarting cameras…`); };
  $('#camFps').onchange = e => send({ fps: +e.target.value }, 'Applying FPS - restarting cameras…');
  $('#camDrv').onchange = e => send({ backend: e.target.value }, 'Switching driver - restarting cameras…');
  $('#swEnh').onchange = e => send({ sw_enhance: e.target.checked });
  $('#btnCamReset').onclick = () => api('/api/camera-settings/reset', {}).then(d => toast(d.message));
})();
