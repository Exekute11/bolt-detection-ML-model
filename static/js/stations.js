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
        <div class="zone-layer"><div class="zone-hint">Drag to add an exclusion zone · draw as many as needed · Esc / Done to finish</div><div class="zone-rect"></div><button class="btn zone-done">Done</button></div>
      </div>
      <div class="zone-list"></div>
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
        <button class="btn ghost b-zone">Add Exclusion Zone</button>
        <button class="btn ghost b-zoneundo">Undo Last Zone</button>
        <button class="btn ghost b-zoneclr">Clear All Zones</button>
        <button class="btn ghost b-fs">Fullscreen</button>
        <span class="sc-fps"></span>
      </div>`;
    grid.appendChild(el);
    const q = sel => el.querySelector(sel);
    const c = { el, id: s.id, img: q('.sc-img'), ov: q('.sc-ov'), stage: q('.sc-stage'), layer: q('.zone-layer'), rect: q('.zone-rect'),
      name: q('.sc-name'), en: q('.sc-en'), badge: q('.sbadge'), det: q('.m-det'), ref: q('.sc-ref'), miss: q('.m-miss'), extra: q('.m-extra'),
      msg: q('.sc-msg'), cam: q('.sc-cam'), conf: q('.sc-conf'), confv: q('.sc-confv'), smooth: q('.sc-smooth'),
      start: q('.b-start'), stop: q('.b-stop'), zone: q('.b-zone'), zoneundo: q('.b-zoneundo'), zoneclr: q('.b-zoneclr'), zdone: q('.zone-done'), zlist: q('.zone-list'), zones: [], fps: q('.sc-fps'), gen: -1 };
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
    const sv = (zones, msg) => save(c, { zones }, msg);
    c.zone.onclick = () => {
      c.layer.classList.add('on');
      toast(`Draw exclusion zones for ${c.name.value} - repeat for each, then press Done`);
    };
    c.zdone.onpointerdown = e => e.stopPropagation();
    c.zdone.onclick = e => { e.stopPropagation(); end(); };
    c.zoneundo.onclick = () => sv(c.zones.slice(0, -1), 'Last exclusion zone removed');
    c.zoneclr.onclick = () => sv([], 'All exclusion zones cleared');
    c.zlist.onclick = e => {
      const b = e.target.closest('[data-i]'); if (!b) return;
      sv(c.zones.filter((_, i) => i !== +b.dataset.i), `Exclusion zone ${+b.dataset.i + 1} removed`);
    };
    c.layer.onpointerdown = e => { if (e.target !== c.layer) return; st = norm(e); drawing = true; c.rect.style.display = 'block';
      Object.assign(c.rect.style, { left: st.x * 100 + '%', top: st.y * 100 + '%', width: '0%', height: '0%' });
      c.layer.setPointerCapture(e.pointerId); e.preventDefault(); };
    c.layer.onpointermove = e => {
      if (!drawing) return; const p = norm(e);
      Object.assign(c.rect.style, { left: Math.min(st.x, p.x) * 100 + '%', top: Math.min(st.y, p.y) * 100 + '%',
        width: Math.abs(p.x - st.x) * 100 + '%', height: Math.abs(p.y - st.y) * 100 + '%' });
    };
    c.layer.onpointerup = e => {
      if (!drawing) return; const p = norm(e), s0 = st;
      drawing = false; st = null; c.rect.style.display = 'none';      /* stay in draw mode for the next zone */
      if (Math.abs(p.x - s0.x) < 0.01 || Math.abs(p.y - s0.y) < 0.01) return;   /* ignore stray clicks */
      const zones = c.zones.concat([[s0.x, s0.y, p.x, p.y]]);
      c.zones = zones;                                                 /* optimistic, so rapid draws don't overwrite each other */
      sv(zones, `Exclusion zone ${zones.length} saved for ${c.name.value}`);
    };
    c.layer.onpointercancel = () => { drawing = false; st = null; c.rect.style.display = 'none'; };
    document.addEventListener('keydown', e => { if (e.key === 'Escape') end(); });
    c.end = end;
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
    const zs = s.zones || [];
    c.zones = zs;
    c.zone.classList.toggle('on', zs.length > 0);
    c.zone.textContent = zs.length ? `Add Exclusion Zone (${zs.length})` : 'Add Exclusion Zone';
    c.zoneundo.disabled = !zs.length;
    c.zoneclr.disabled = !zs.length;
    const zk = JSON.stringify(zs);
    if (c.zlist.dataset.k !== zk) {
      c.zlist.dataset.k = zk;
      c.zlist.innerHTML = zs.map((_, i) => `<span class="zchip">Zone ${i + 1}<button data-i="${i}" title="Remove zone ${i + 1}">×</button></span>`).join('');
    }
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
