/* Godrej Fastener Vision - shared helpers, polling, overall result, event timeline */
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
let lastEventId = -1, offline = false, lastOverall = null;

function toast(m) {
  const t = $('#toast');
  if (!t) return;
  t.textContent = m;
  t.classList.add('show');
  clearTimeout(toast.t);
  toast.t = setTimeout(() => t.classList.remove('show'), 2800);
}
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, body, method) {
  const r = await fetch(path, {
    method: method || (body === undefined ? 'GET' : 'POST'),
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body || {}),
    cache: 'no-store'
  });
  let d = {};
  try { d = await r.json(); } catch (e) { /* non-JSON error page */ }
  d._status = r.status;
  return d;
}

setInterval(() => { const c = $('#clock'); if (c) c.textContent = new Date().toLocaleTimeString(); }, 1000);

/* short beep when the overall result turns FAIL / CAMERA ERROR */
let audioCtx;
function beep() {
  if (!$('#sndOn') || !$('#sndOn').checked) return;
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.type = 'square'; o.frequency.value = 880; g.gain.value = 0.08;
    o.connect(g); g.connect(audioCtx.destination);
    o.start(); o.stop(audioCtx.currentTime + 0.35);
  } catch (e) { /* audio blocked until first click - ignore */ }
}

const OV_CLASS = { PASS: 'pass', FAIL: 'fail', WAITING: 'wait', CAMERA_ERROR: 'err', NO_STATIONS: 'off' };

function renderOverall(d) {
  const o = d.overall, ov = $('#overall');
  ov.className = 'overall ' + (OV_CLASS[o.status] || 'wait');
  $('#ovLabel').textContent = o.label;
  $('#ovReason').textContent = o.reasons.length ? o.reasons.join('  ·  ') : '';
  const c = o.counts || {};
  $('#kEnabled').textContent = o.enabled;
  $('#kPass').textContent = c.PASS || 0;
  $('#kFail').textContent = c.FAIL || 0;
  $('#kWait').textContent = (c.WAITING || 0) + (c.STOPPED || 0);
  $('#kErr').textContent = c.CAMERA_ERROR || 0;
  if (lastOverall !== null && lastOverall !== o.status && (o.status === 'FAIL' || o.status === 'CAMERA_ERROR')) beep();
  lastOverall = o.status;
}

function renderEvents(d) {
  const topId = d.events.length ? d.events[0].id : 0;
  if (topId === lastEventId) return;
  const first = lastEventId === -1;
  lastEventId = topId;
  $('#timeline').innerHTML = d.events.length
    ? d.events.map(e => `<li class="${e.level}"><i></i><div><b>${esc(e.msg)}</b><small>${e.t}</small></div></li>`).join('')
    : '<li class="empty">No events yet</li>';
  if (!first) { const f = $('#timeline li'); if (f) f.classList.add('new'); }
}

function render(d) {
  renderOverall(d);
  renderEvents(d);
  const run = d.stations.filter(s => s.enabled && s.running);
  $('#fps').textContent = Math.round(run.reduce((a, s) => a + (s.det_fps || 0), 0));
  $('#fpsPill').title = 'Total AI detection FPS (shared by all running stations)';
  $('#liveText').textContent = offline ? 'OFFLINE' : run.length ? 'LIVE' : 'IDLE';
  $('#livePill').classList.toggle('off', offline || !run.length);
  const wb = $('#warnBar');
  const html = d.warnings.map(w => `<div>⚠ ${esc(w)}</div>`).join('');
  if (wb.dataset.h !== html) { wb.dataset.h = html; wb.innerHTML = html; }
  if (window.stationsRender) window.stationsRender(d);
}

async function poll() {
  try {
    const r = await fetch('/status', { cache: 'no-store' });
    render(await r.json());
    offline = false;
  } catch (e) {
    offline = true;
    $('#liveText').textContent = 'OFFLINE';
    $('#livePill').classList.add('off');
  }
  setTimeout(poll, 300);
}
window.addEventListener('DOMContentLoaded', () => setTimeout(poll, 50));
