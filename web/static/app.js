"use strict";
const $ = (id) => document.getElementById(id);
const SUIT = { c: "♣", d: "♦", h: "♥", s: "♠" };
const SEND_MAX_W = 1280;       // 傳給後端的影格寬度上限（分割模型輸入 640，再大沒有意義）

const st = {
  mode: "cam", paused: false, ws: null, busy: false, sentAt: 0, fps: 0, debug: false,
  res: null, stream: null, files: [], fileIdx: -1, truthDirty: false, recording: null,
  settings: { conf: 0.5, value_min: 0.6, verify: true, scan: true, smooth: true },
  rp: { folder: "", frames: [], idx: -1, playing: false, timer: 0, lastIdx: -1 },
};
const video = $("video"), still = $("still"), overlay = $("overlay"), ctx = overlay.getContext("2d");
const grab = document.createElement("canvas");

// ---------------------------------------------------------------- 工具
const pretty = (l) => l.slice(0, -1) + SUIT[l.slice(-1)];
const isRed = (l) => "hd".includes(l.slice(-1));
const pct = (x) => (x * 100).toFixed(0) + "%";
function chip(l) { const s = document.createElement("span"); s.className = "pc" + (isRed(l) ? " red" : ""); s.textContent = pretty(l); return s; }
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// 框的狀態說明：沒有牌值時說明原因
function segReason(g) {
  const v = g.value || {};
  if (g.labels.length) return null;
  if (!v.label) return "讀不到牌角";
  if (v.multi) return "多張?";
  return `信心低 ${pct(v.conf)}`;
}
// 牌內切片中與主牌值點數不同的讀值（同點數的只是又讀到同一張牌的牌角）
const scanExtra = (g) => (g.scan || []).filter((s) => !g.value || !g.value.label || s[0].slice(0, -1) !== g.value.label.slice(0, -1));
function segText(g) {
  if (g.labels.length) return g.labels.map(pretty).join("+");
  return g.value && g.value.multi ? "多張?" : "?";
}

// ---------------------------------------------------------------- WebSocket
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { $("conn").textContent = "已連線"; $("conn").className = "pill on"; st.busy = false; sendSettings(); };
  ws.onclose = () => { $("conn").textContent = "未連線"; $("conn").className = "pill off"; setTimeout(connect, 1000); };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "state") onState(m);
    else if (m.type === "result") onResult(m);
    else if (m.type === "report") {
      const el = $("reportMsg"); el.className = "small " + (m.ok ? "ok" : "err");
      el.textContent = m.msg + (m.count ? `（共 ${m.count} 筆）` : ""); if (m.ok) $("reportNote").value = "";
    }
  };
  st.ws = ws;
}
const wsReady = () => st.ws && st.ws.readyState === 1;
function sendSettings() {
  if (wsReady()) st.ws.send(JSON.stringify({ type: "settings", ...st.settings, smooth: st.settings.smooth && st.mode !== "img" }));
}
function sendCmd(cmd, extra = {}) { if (wsReady()) st.ws.send(JSON.stringify({ type: "cmd", cmd, ...extra })); }

function onState(m) {
  const miss = Object.entries(m.models).filter(([, ok]) => !ok).map(([k]) => `models/${k}/`);
  if (miss.length) { $("note").textContent = `尚未下載模型：${miss.join("、")}（見 README 的權重下載說明）`; $("note").className = "note err"; }
  const wasRec = !!st.recording;
  setRecording(m.recording);
  if (wasRec && !m.recording) loadFolders();   // 錄製結束：重播與回測清單加入新資料夾
}
function setRecording(r) {
  st.recording = r;
  $("rec").hidden = !r;
  if (r) $("rec").textContent = `● 錄製中 ${r.s.toFixed(0)} 秒 · ${r.frames} 幀`;
  $("recBtn").classList.toggle("recording", !!r);
  $("recBtn").textContent = r ? "■ 停止錄製" : "● 錄製";
}

function onResult(m) {
  st.busy = false;
  if (m.error) { $("note").textContent = m.error; $("note").className = "note err"; return; }
  const dt = performance.now() - st.sentAt;
  st.fps = st.fps ? st.fps * 0.85 + (1000 / dt) * 0.15 : 1000 / dt;
  const ms = m.ms;
  $("perf").textContent = `分割 ${ms.seg} · 牌值 ${ms.value} · 切片 ${ms.scan} · 共 ${ms.total} ms · ${st.fps.toFixed(1)} FPS`;
  if (m.recording) setRecording(m.recording); else if (st.recording && st.mode === "cam") setRecording(null);
  st.res = m;
  renderPanel(m);
  drawOverlay();
  if (st.mode === "replay" && st.rp.playing) scheduleNext();
}

// ---------------------------------------------------------------- 影格來源
async function listCams() {
  const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const sel = $("camSel"), cur = sel.value;
  sel.innerHTML = "";
  devs.forEach((d, i) => sel.add(new Option(d.label || `攝影機 ${i + 1}`, d.deviceId)));
  if (cur) sel.value = cur;
}
async function startCam() {
  try {
    if (st.stream) st.stream.getTracks().forEach((t) => t.stop());
    const id = $("camSel").value;
    st.stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: id ? { exact: id } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
    video.srcObject = st.stream;
    await video.play();
    $("empty").hidden = true;
    await listCams();
  } catch (err) {
    $("empty").hidden = false;
    $("empty").textContent = "無法開啟攝影機：" + err.message + "（可改用「圖片」或「重播」模式）";
  }
}
function stopCam() { if (st.stream) { st.stream.getTracks().forEach((t) => t.stop()); st.stream = null; } }
function sourceSize() {
  return st.mode === "cam" ? [video.videoWidth, video.videoHeight] : [still.naturalWidth, still.naturalHeight];
}
function sendFrame(t) {
  const el = st.mode === "cam" ? video : still;
  const [w, h] = sourceSize();
  if (!w || !h || !wsReady()) return;
  const k = Math.min(1, SEND_MAX_W / w);
  grab.width = Math.round(w * k); grab.height = Math.round(h * k);
  grab.getContext("2d").drawImage(el, 0, 0, grab.width, grab.height);
  st.busy = true; st.sentAt = performance.now();
  grab.toBlob((b) => {
    if (!b) { st.busy = false; return; }
    b.arrayBuffer().then((buf) => {
      if (t !== undefined) st.ws.send(JSON.stringify({ type: "frame_t", t }));   // 重播：用原始時間戳
      st.ws.send(buf);
    });
  }, "image/jpeg", 0.9);
}
function loop() {
  if (st.mode === "cam" && !st.paused && !st.busy && video.readyState >= 2) sendFrame();
  if (st.busy && performance.now() - st.sentAt > 1500) { $("note").textContent = "模型載入／推論中…"; $("note").className = "note"; }
  requestAnimationFrame(loop);
}

function setMode(mode) {
  if (st.recording && mode !== "cam") sendCmd("record_stop");
  st.mode = mode;
  ["Cam", "Img", "Replay"].forEach((n) => $("mode" + n).classList.toggle("on", mode === n.toLowerCase()));
  document.querySelectorAll(".grp").forEach((g) => (g.hidden = g.dataset.mode !== mode));
  $("thumbs").hidden = mode !== "img" || st.files.length < 2;
  $("scrub").hidden = mode !== "replay";
  $("smoothChk").disabled = mode === "img";
  video.hidden = mode !== "cam";
  still.hidden = mode === "cam";
  still.removeAttribute("src");
  pauseReplay();
  st.res = null; st.busy = false; renderEmptyPanel(); drawOverlay(); sendCmd("reset"); sendSettings();
  if (mode === "cam") { if (!st.stream) startCam(); else $("empty").hidden = true; return; }
  stopCam();
  if (mode === "img") {
    $("empty").hidden = st.fileIdx >= 0; $("empty").textContent = "選擇一張或多張照片";
    if (st.fileIdx >= 0) showFile(st.fileIdx);
  } else {
    $("empty").hidden = false; $("empty").textContent = "選擇影格資料夾（DEBUG 監看存檔或錄製）";
    if ($("folderSel").value) openFolder($("folderSel").value);
  }
}
function showFile(i) {
  st.fileIdx = i;
  still.onload = () => { $("empty").hidden = true; st.busy = false; sendFrame(); };
  still.src = URL.createObjectURL(st.files[i]);
  [...$("thumbs").children].forEach((t, j) => t.classList.toggle("on", j === i));
}

// ---------------------------------------------------------------- 重播
async function loadFolders() {
  const j = await (await fetch("/api/frames")).json();
  for (const id of ["folderSel", "btSel"]) {
    const sel = $(id), cur = sel.value;
    sel.innerHTML = "";
    if (!j.folders.length) sel.add(new Option("（還沒有影格：用「錄製」或 CARD_DEBUG=1 監看）", ""));
    j.folders.forEach((f) => sel.add(new Option(`${f.id}（${f.frames} 幀）`, f.id)));
    if (cur) sel.value = cur;
  }
}
async function openFolder(id) {
  pauseReplay();
  st.rp.folder = id; st.rp.frames = []; st.rp.idx = -1;
  if (!id) return;
  const j = await (await fetch(`/api/frames/${id}/list`)).json();
  if (st.rp.folder !== id) return;   // 等待期間又換了資料夾
  st.rp.frames = j.frames;
  $("frameSlider").max = Math.max(0, j.frames.length - 1);
  showReplayFrame(0);
}
function showReplayFrame(i) {
  const f = st.rp.frames[i];
  if (!f) return;
  if (i !== st.rp.lastIdx + 1) sendCmd("reset");   // 跳格：追蹤紀錄不連續，先清除
  st.rp.idx = i; st.rp.lastIdx = i;
  $("frameSlider").value = i;
  $("frameInfo").textContent = `${i + 1} / ${st.rp.frames.length} · ${new Date(f.t * 1000).toLocaleTimeString()}`;
  still.onload = () => { $("empty").hidden = true; st.busy = false; sendFrame(f.t); };
  still.src = `/api/frames/${st.rp.folder}/img/${encodeURIComponent(f.name)}`;
}
function scheduleNext() {
  const i = st.rp.idx, fr = st.rp.frames;
  if (i >= fr.length - 1) { pauseReplay(); return; }
  const speed = parseFloat($("speedSel").value);
  const elapsed = performance.now() - st.sentAt;
  const wait = speed > 0 ? Math.min(2000, (fr[i + 1].t - fr[i].t) * 1000 / speed) - elapsed : 0;
  st.rp.timer = setTimeout(() => showReplayFrame(i + 1), Math.max(0, wait));
}
function playReplay() {
  if (!st.rp.frames.length) return;
  if (st.rp.idx >= st.rp.frames.length - 1) { st.rp.lastIdx = -2; showReplayFrame(0); }
  st.rp.playing = true; $("playBtn").textContent = "⏸ 暫停";
  if (!st.busy) scheduleNext();
}
function pauseReplay() { st.rp.playing = false; clearTimeout(st.rp.timer); $("playBtn").textContent = "▶ 播放"; }

// ---------------------------------------------------------------- 繪圖
function contentRect() {
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  const [w, h] = st.res ? [st.res.w, st.res.h] : sourceSize();
  if (!w || !h) return { s: 1, x: 0, y: 0 };
  const s = Math.min(cw / w, ch / h);
  return { s, x: (cw - w * s) / 2, y: (ch - h * s) / 2 };
}
function path(pts, P) { ctx.beginPath(); pts.forEach((p, k) => { const [x, y] = P(p); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.closePath(); }
function label(text, x, y, bg, fg = "#fff", font = "bold 15px 'Segoe UI Symbol', 'Microsoft JhengHei', sans-serif") {
  ctx.font = font; const tw = ctx.measureText(text).width + 10;
  ctx.fillStyle = bg; ctx.beginPath(); ctx.roundRect(x - tw / 2, y - 11, tw, 22, 5); ctx.fill();
  ctx.fillStyle = fg; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, x, y + 1);
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
}
function drawOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = overlay.clientWidth * dpr; overlay.height = overlay.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight);
  if (!st.res) return;
  const r = contentRect(), P = (p) => [r.x + p[0] * r.s, r.y + p[1] * r.s];
  // 背景過濾剔除的框：灰色虛線（多半是門、畫框、書櫃）
  (st.res.segs_rejected || []).forEach((g) => {
    path(g.poly, P); ctx.setLineDash([4, 6]); ctx.strokeStyle = "rgba(170,170,170,.8)"; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
  });
  st.res.segs.forEach((g, i) => {
    const id = g.track ?? i, hue = (id * 67) % 360, on = g.confirmed;
    const col = on ? `hsl(${hue},80%,55%)` : "hsl(0,0%,70%)";
    path(g.poly, P);
    ctx.fillStyle = on ? `hsla(${hue},80%,55%,.22)` : "rgba(200,200,200,.12)"; ctx.fill();
    ctx.strokeStyle = col; ctx.lineWidth = on ? 2.5 : 1.5; ctx.stroke();
    // 標籤放在框最上方的點
    const top = g.rbox.reduce((a, b) => (b[1] < a[1] ? b : a));
    const [tx, ty] = P(top), v = g.value || {};
    const conf = g.labels.length && v.label === g.labels[0] ? ` ${pct(v.conf)}` : "";
    label(segText(g) + conf, tx, Math.max(12, ty - 14), on ? `hsl(${hue},80%,35%)` : "rgba(60,60,60,.85)");
    if (!st.debug) return;
    path(g.rbox, P); ctx.setLineDash([6, 4]); ctx.strokeStyle = "rgba(255,255,255,.8)"; ctx.lineWidth = 1; ctx.stroke(); ctx.setLineDash([]);
    (v.corner_pts || []).forEach((p) => { const [x, y] = P(p); ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.strokeStyle = "#56ccf2"; ctx.lineWidth = 2; ctx.stroke(); });
    scanExtra(g).forEach(([lab, q, x0, y0]) => {
      const [x, y] = P([x0, y0]);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, 7); ctx.fillStyle = "#f2994a"; ctx.fill();
      label(`${pretty(lab)} ${pct(q)}`, x, y - 14, "rgba(242,153,74,.9)", "#111", "bold 11px sans-serif");
    });
    const [cx, cy] = P(g.center);
    label(`#${id} ${g.verify || ""} 白${pct(v.white || 0)} 邊${pct(v.border || 0)} 彩${pct(v.color || 0)}`, cx, cy,
      "rgba(0,0,0,.65)", "#fff", "11px sans-serif");
  });
}

// ---------------------------------------------------------------- 面板
function renderEmptyPanel() {
  $("hand").textContent = "—"; $("handInfo").textContent = ""; $("segTable").tBodies[0].innerHTML = "";
}
function renderPanel(m) {
  const el = $("hand"); el.innerHTML = "";
  m.hand.length ? m.hand.forEach((l) => el.appendChild(chip(l))) : (el.textContent = "—");
  const rej = (m.segs_rejected || []).length;
  $("handInfo").textContent = `${m.hand.length} 張 · ${m.segs.length} 個框` + (rej ? `（剔除背景 ${rej}）` : "");
  if (!$("note").classList.contains("err")) $("note").textContent = "";
  const rows = m.segs.map((g, i) => {
    const v = g.value || {}, why = segReason(g);
    const status = why ? `<span class="tag ${why === "讀不到牌角" ? "bad" : "warn"}">${why}</span>`
      : `<span class="tag ${g.confirmed ? "ok" : "warn"}">${g.confirmed ? "已確認" : "確認中"}</span>`;
    const via = g.verify && g.verify !== "card" ? ` <span class="tag">${g.verify === "chain" ? "扇形串接" : "記憶"}</span>` : "";
    const ex = scanExtra(g);
    const scan = ex.length ? `<div class="muted">切片另見：${[...new Set(ex.map((s) => pretty(s[0])))].join(" ")}</div>` : "";
    return `<tr class="${g.confirmed ? "" : "dim"}"><td>${g.track ?? i + 1}</td><td>${esc(segText(g))}${scan}</td>` +
      `<td>${v.label ? pct(v.conf) : "—"}</td><td>${g.track !== null ? pct(g.agree) : "—"}</td><td>${status}${via}</td></tr>`;
  });
  $("segTable").tBodies[0].innerHTML = rows.join("") || `<tr><td colspan="5" class="muted">畫面中沒有牌</td></tr>`;
  if (!st.truthDirty && document.activeElement !== $("truth")) $("truth").value = currentTruth();
}
const currentTruth = () => st.res ? ([...st.res.hand].sort().join("-") || "none") : "";

// ---------------------------------------------------------------- 回測 / 存檔 / 評估
async function runBacktest() {
  const id = $("btSel").value, out = $("btOut");
  if (!id) return;
  out.textContent = "回測中…（每幀約 30~40 ms）"; $("btBtn").disabled = true;
  try {
    const r = await fetch("/api/backtest", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder: id, settings: { ...st.settings } }) });
    const j = await r.json();
    if (!r.ok) { out.innerHTML = `<span class="err">${esc(j.detail)}</span>`; return; }
    const n = (k) => j[k] || 0, kept = n("kept") || 1;
    out.innerHTML = `<table class="kv">
      <tr><td>影格</td><td>${n("frames")}</td><td>每幀耗時</td><td>${j.ms_median} ms</td></tr>
      <tr><td>保留框</td><td>${n("kept")}</td><td>剔除背景</td><td>${n("rejected")}</td></tr>
      <tr><td>單張牌值</td><td>${n("single")}（${pct(n("single") / kept)}）</td><td>一框多張</td><td>${n("multi")}</td></tr>
      <tr><td>沒有牌值</td><td>${n("unknown")}（${pct(n("unknown") / kept)}）</td><td>手牌變動</td><td>${n("hand_changes")} 次</td></tr></table>
      <div class="muted">最常見手牌：${j.top_hands.map(([h, c]) => `${esc(h)}×${c}`).join("、")}</div>`;
  } finally { $("btBtn").disabled = false; }
}
async function saveCapture() {
  const src = st.mode === "cam" ? video : still, [w, h] = sourceSize();
  if (!w) return;
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  c.getContext("2d").drawImage(src, 0, 0);
  const r = await fetch("/api/capture", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: c.toDataURL("image/jpeg", 0.92), truth: $("truth").value.trim() || "none" }) });
  const j = await r.json(), msg = $("saveMsg");
  if (r.ok) { msg.className = "small ok"; msg.textContent = `已存 ${j.saved}`; $("rtCount").textContent = `已存 ${j.count} 張`; st.truthDirty = false; }
  else { msg.className = "small err"; msg.textContent = j.detail || "存檔失敗"; }
}
async function runEval() {
  const out = $("evalOut"); out.textContent = "評估中…";
  const r = await fetch("/api/evaluate", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings: { ...st.settings } }) });
  const j = await r.json();
  if (!r.ok) { out.innerHTML = `<span class="err">${esc(j.detail)}</span>`; return; }
  out.innerHTML = `<table class="kv">
    <tr><td>照片數</td><td>${j.n}</td><td>整張全對</td><td>${pct(j.exact)}</td></tr>
    <tr><td>精確率</td><td>${pct(j.precision)}</td><td>召回率</td><td>${pct(j.recall)}</td></tr></table>
    ${j.errors.length ? j.errors.map((e) => `<div class="muted">${esc(e)}</div>`).join("") : '<p class="ok">全部正確</p>'}`;
}
function report() { sendCmd("report", { note: $("reportNote").value.trim() }); }
function toggleRecord() { if (st.mode === "cam") sendCmd(st.recording ? "record_stop" : "record_start"); }
const resend = () => {
  if (st.mode === "img" && st.fileIdx >= 0) { st.busy = false; sendFrame(); }
  else if (st.mode === "replay" && !st.rp.playing && st.rp.idx >= 0) { st.rp.lastIdx = -2; showReplayFrame(st.rp.idx); }
};

// ---------------------------------------------------------------- 事件
function bindRange(id, key, outId) {
  $(id).addEventListener("input", (e) => {
    st.settings[key] = parseFloat(e.target.value);
    $(outId).textContent = parseFloat(e.target.value).toFixed(2);
    sendSettings(); resend();
  });
}
bindRange("conf", "conf", "confOut");
bindRange("valueMin", "value_min", "valueOut");
for (const [id, key] of [["verifyChk", "verify"], ["scanChk", "scan"], ["smoothChk", "smooth"]])
  $(id).addEventListener("change", (e) => { st.settings[key] = e.target.checked; sendSettings(); resend(); });
$("debugChk").addEventListener("change", (e) => { st.debug = e.target.checked; drawOverlay(); });

$("modeCam").onclick = () => setMode("cam");
$("modeImg").onclick = () => setMode("img");
$("modeReplay").onclick = () => setMode("replay");
$("camSel").onchange = startCam;
$("pauseBtn").onclick = () => { st.paused = !st.paused; $("pauseBtn").textContent = st.paused ? "▶ 繼續" : "⏸ 暫停"; };
$("recBtn").onclick = toggleRecord;
$("fileIn").onchange = (e) => {
  st.files = [...e.target.files]; const th = $("thumbs"); th.innerHTML = "";
  st.files.forEach((f, i) => { const im = document.createElement("img"); im.src = URL.createObjectURL(f); im.onclick = () => showFile(i); th.appendChild(im); });
  th.hidden = st.files.length < 2; if (st.files.length) showFile(0);
};
$("folderSel").onchange = (e) => openFolder(e.target.value);
$("playBtn").onclick = () => (st.rp.playing ? pauseReplay() : playReplay());
$("frameSlider").addEventListener("input", (e) => { pauseReplay(); showReplayFrame(parseInt(e.target.value)); });
$("resetBtn").onclick = () => sendCmd("reset");
$("btBtn").onclick = runBacktest;
$("reportBtn").onclick = report;
$("saveBtn").onclick = saveCapture;
$("fillBtn").onclick = () => { st.truthDirty = false; $("truth").value = currentTruth(); };
$("truth").addEventListener("input", () => (st.truthDirty = true));
$("evalBtn").onclick = runEval;
window.addEventListener("resize", drawOverlay);
window.addEventListener("keydown", (e) => {
  const a = document.activeElement;
  if ((a.tagName === "INPUT" && a.type !== "range" && a.type !== "checkbox") || a.tagName === "SELECT") return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); st.mode === "replay" ? $("playBtn").click() : $("pauseBtn").click(); }
  else if (k === "r") toggleRecord();
  else if (k === "s") saveCapture();
  else if (k === "c") sendCmd("reset");
  else if (k === "e") report();
  else if (k === "d") $("debugChk").click();
  else if (k === "arrowright" || k === "arrowleft") {
    const d = k === "arrowright" ? 1 : -1;
    if (st.mode === "img" && st.files[st.fileIdx + d]) showFile(st.fileIdx + d);
    if (st.mode === "replay" && st.rp.frames[st.rp.idx + d]) { pauseReplay(); showReplayFrame(st.rp.idx + d); }
  }
});

async function init() {
  connect();
  await loadFolders();
  const rt = await (await fetch("/api/realtest")).json();
  $("rtCount").textContent = `已存 ${rt.count} 張`;
  if (rt.hard_cases) $("reportMsg").textContent = `已回報 ${rt.hard_cases} 筆`;
  setMode("cam");
  requestAnimationFrame(loop);
}
init();
