"use strict";
const $ = (id) => document.getElementById(id);
const SUIT = { c: "♣", d: "♦", h: "♥", s: "♠" };
const SEND_MAX_W = 1280;       // 傳給後端的影格寬度上限（推論解析度最高 1024，再大沒有意義）

const st = {
  mode: "cam", paused: false, ws: null, busy: false, sentAt: 0, fps: 0,
  res: null, stream: null, files: [], fileIdx: -1, truthDirty: false,
  settings: { model: null, conf: 0.45, window: 8, dist_factor: 6.0, imgsz: 832, ensemble: true, refine: true, view: "corner", seg_verify: true, seg_value: true },
};
const video = $("video"), still = $("still"), overlay = $("overlay"), ctx = overlay.getContext("2d");
const grab = document.createElement("canvas");

// ---------------------------------------------------------------- 工具
const pretty = (l) => l.slice(0, -1) + SUIT[l.slice(-1)];
const isRed = (l) => "hd".includes(l.slice(-1));
const pct = (x) => (x * 100).toFixed(0) + "%";
const confColor = (c) => (c >= 0.8 ? "#3ccf7a" : c >= 0.5 ? "#f2c14e" : "#e5534b");
function chip(l) { const s = document.createElement("span"); s.className = "pc" + (isRed(l) ? " red" : ""); s.textContent = pretty(l); return s; }

// ---------------------------------------------------------------- WebSocket
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { $("conn").textContent = "已連線"; $("conn").className = "pill on"; st.busy = false; sendSettings(); };
  ws.onclose = () => { $("conn").textContent = "未連線"; $("conn").className = "pill off"; setTimeout(connect, 1000); };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type === "state") {
      $("refine").disabled = !m.refiner;
      [...$("viewSel").options].forEach((o) => { if (o.value !== "corner") o.disabled = !m.card_seg; });
      $("viewHint").textContent = m.card_seg ? "" : "整張牌框模型訓練中，下載到 models/card_seg/ 後即可選";
      $("refineHint").textContent = m.refiner ? "" : "（尚未下載分類模型 models/corner_cls/）";
      if (!st.settings.model && m.settings.model) { st.settings.model = m.settings.model; $("modelSel").value = m.settings.model; sendSettings(); }
    }
    else if (m.type === "result") onResult(m);
    else if (m.type === "report") {
      const el = $("reportMsg"); el.className = "small " + (m.ok ? "ok" : "err");
      el.textContent = m.msg + (m.count ? `（共 ${m.count} 筆）` : ""); if (m.ok) $("reportNote").value = "";
    }
  };
  st.ws = ws;
}
const wsReady = () => st.ws && st.ws.readyState === 1;
function sendSettings() { if (wsReady()) st.ws.send(JSON.stringify({ type: "settings", ...st.settings, smooth: st.mode === "cam" })); }
function sendCmd(cmd) { if (wsReady()) st.ws.send(JSON.stringify({ type: "cmd", cmd })); }

function onResult(m) {
  st.busy = false;
  if (m.error) { $("note").textContent = m.error; $("note").className = "note err"; return; }
  const dt = performance.now() - st.sentAt;
  st.fps = st.fps ? st.fps * 0.85 + (1000 / dt) * 0.15 : 1000 / dt;
  const sc = m.scales.length ? `${m.scales.join("+")}px · ` : "整張牌框 · ";
  $("perf").textContent = `${sc}推論 ${m.infer_ms} ms · ${st.fps.toFixed(1)} FPS`;
  st.res = m;
  renderPanel(m);
  drawOverlay();
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
    $("empty").textContent = "無法開啟攝影機：" + err.message + "（可改用「圖片」模式）";
  }
}
function sourceSize() {
  return st.mode === "cam" ? [video.videoWidth, video.videoHeight] : [still.naturalWidth, still.naturalHeight];
}
function sendFrame() {
  const el = st.mode === "cam" ? video : still;
  const [w, h] = sourceSize();
  if (!w || !h || !wsReady()) return;
  const k = Math.min(1, SEND_MAX_W / w);
  grab.width = Math.round(w * k); grab.height = Math.round(h * k);
  grab.getContext("2d").drawImage(el, 0, 0, grab.width, grab.height);
  st.busy = true; st.sentAt = performance.now();
  grab.toBlob((b) => b ? b.arrayBuffer().then((buf) => st.ws.send(buf)) : (st.busy = false), "image/jpeg", 0.85);
}
function loop() {
  if (st.mode === "cam" && !st.paused && !st.busy && video.readyState >= 2) sendFrame();
  if (st.busy && performance.now() - st.sentAt > 1500) { $("note").textContent = "模型載入／推論中…"; $("note").className = "note"; }
  requestAnimationFrame(loop);
}

function setMode(mode) {
  st.mode = mode;
  $("modeCam").classList.toggle("on", mode === "cam");
  $("modeImg").classList.toggle("on", mode === "img");
  $("camSel").hidden = mode !== "cam";
  $("pauseBtn").hidden = mode !== "cam";
  $("fileWrap").hidden = mode !== "img";
  $("thumbs").hidden = mode !== "img" || st.files.length < 2;
  video.hidden = mode !== "cam";
  still.hidden = mode !== "img";
  st.res = null; drawOverlay(); sendSettings();
  if (mode === "cam") { st.busy = false; if (!st.stream) startCam(); else $("empty").hidden = true; }
  else { if (st.stream) { st.stream.getTracks().forEach((t) => t.stop()); st.stream = null; }
         $("empty").hidden = st.fileIdx >= 0; $("empty").textContent = "選擇一張或多張照片"; }
}
function showFile(i) {
  st.fileIdx = i;
  const f = st.files[i];
  still.onload = () => { $("empty").hidden = true; st.busy = false; sendFrame(); };
  still.src = URL.createObjectURL(f);
  [...$("thumbs").children].forEach((t, j) => t.classList.toggle("on", j === i));
}

// ---------------------------------------------------------------- 繪圖
function contentRect() {
  const cw = overlay.clientWidth, ch = overlay.clientHeight;
  const [w, h] = st.res ? [st.res.w, st.res.h] : sourceSize();
  if (!w || !h) return { s: 1, x: 0, y: 0 };
  const s = Math.min(cw / w, ch / h);
  return { s, x: (cw - w * s) / 2, y: (ch - h * s) / 2 };
}
function drawOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = overlay.clientWidth * dpr; overlay.height = overlay.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight);
  if (!st.res) return;
  const r = contentRect();
  // 被牌角交叉驗證剔除的整張牌框：灰色虛線輪廓（框內沒有牌角，多半是門、畫框等背景）
  (st.res.segs_rejected || []).forEach((g) => {
    ctx.beginPath(); g.poly.forEach((p, k) => { const x = r.x + p[0] * r.s, y = r.y + p[1] * r.s; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.closePath();
    ctx.setLineDash([4, 6]); ctx.strokeStyle = "rgba(160,160,160,.8)"; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
  });
  // 整張牌框：露出區域（半透明填色）＋貼合的旋轉框（虛線）＋編號與信心度
  (st.res.segs || []).forEach((g, i) => {
    const hue = (i * 67) % 360, P = (p) => [r.x + p[0] * r.s, r.y + p[1] * r.s];
    ctx.beginPath(); g.poly.forEach((p, k) => { const [x, y] = P(p); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.closePath();
    ctx.fillStyle = `hsla(${hue},85%,55%,.28)`; ctx.fill();
    ctx.strokeStyle = `hsl(${hue},85%,55%)`; ctx.lineWidth = 2; ctx.stroke();
    ctx.beginPath(); g.rbox.forEach((p, k) => { const [x, y] = P(p); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.closePath();
    ctx.setLineDash([6, 4]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
    const [cx, cy] = P(g.center), t = `#${i + 1} ${segValueText(g)}`;
    ctx.font = "bold 14px sans-serif"; const tw = ctx.measureText(t).width + 8;
    ctx.fillStyle = `hsl(${hue},85%,40%)`; ctx.fillRect(cx - tw / 2, cy - 10, tw, 20);
    ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.fillText(t, cx, cy + 5); ctx.textAlign = "left";
  });
  ctx.font = "bold 12px sans-serif";
  for (const d of st.res.detections) {
    const [x1, y1, x2, y2] = d.box, col = confColor(d.conf);
    const X = r.x + x1 * r.s, Y = r.y + y1 * r.s;
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.strokeRect(X, Y, (x2 - x1) * r.s, (y2 - y1) * r.s);
    const fixed = d.det_label && d.det_label !== d.label;   // 二次分類改判了偵測器的標籤
    const t = `${fixed ? pretty(d.det_label) + "→" : ""}${pretty(d.label)} ${pct(d.conf)}`, tw = ctx.measureText(t).width + 6;
    if (fixed) { ctx.strokeStyle = "#bb6bd9"; ctx.lineWidth = 3; ctx.strokeRect(X - 2, Y - 2, (x2 - x1) * r.s + 4, (y2 - y1) * r.s + 4); }
    ctx.fillStyle = col; ctx.fillRect(X, Y - 16, tw, 16);
    ctx.fillStyle = "#111"; ctx.fillText(t, X + 3, Y - 4);
  }
  // 合併後的實體牌：中心標籤＋連到各牌角的線
  ctx.font = "bold 22px 'Segoe UI Symbol', 'Microsoft JhengHei', sans-serif";
  for (const c of st.res.cards) {
    const x = r.x + c.center[0] * r.s, y = r.y + c.center[1] * r.s, t = pretty(c.label);
    if (c.corners > 1) {
      ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      for (const d of st.res.detections.filter((d) => d.label === c.raw)) {
        const dx = r.x + (d.box[0] + d.box[2]) / 2 * r.s, dy = r.y + (d.box[1] + d.box[3]) / 2 * r.s;
        const lim = st.settings.dist_factor * Math.max(d.box[2] - d.box[0], d.box[3] - d.box[1]) * r.s;
        // 只連到屬於這張牌的牌角（同名牌可能有多張）
        if (Math.hypot(dx - x, dy - y) < lim) { ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(dx, dy); ctx.stroke(); }
      }
      ctx.setLineDash([]);
    }
    const tw = ctx.measureText(t).width + 14;
    ctx.globalAlpha = c.confirmed ? 1 : 0.45;           // 尚未確認的軌跡（出現不到 3 幀）半透明
    ctx.fillStyle = "rgba(255,255,255,.95)";
    ctx.beginPath(); ctx.roundRect(x - tw / 2, y - 16, tw, 30, 6); ctx.fill();
    ctx.fillStyle = isRed(c.label) ? "#c62828" : "#111"; ctx.textAlign = "center"; ctx.fillText(t, x, y + 7);
    if (c.raw !== c.label) {                            // 本幀原始判讀與追蹤投票不同：顯示被修正的原始結果
      ctx.font = "bold 12px sans-serif"; ctx.fillStyle = "#e5534b";
      ctx.fillText(`本幀:${pretty(c.raw)} → 已修正`, x, y + 30);
      ctx.font = "bold 22px 'Segoe UI Symbol', 'Microsoft JhengHei', sans-serif";
    }
    ctx.textAlign = "left"; ctx.globalAlpha = 1;
  }
}

// ---------------------------------------------------------------- 面板
// 整張牌框的標籤文字：B-2 牌值達門檻才顯示，否則「?」；兩角認成不同牌時提示「多張?」；沒有牌值時顯示框的信心
function segValueText(g) {
  if (!g.value) return pct(g.conf);
  if (g.value.shown === false) return g.value.multi ? "多張?" : "?";
  return `${pretty(g.value.label)} ${pct(g.value.conf)}`;
}

function renderPanel(m) {
  const el = $("stableCards"); el.innerHTML = "";
  m.stable.length ? m.stable.forEach((l) => el.appendChild(chip(l))) : (el.textContent = "—");
  $("detCount").textContent = st.settings.view === "card" ? "" : `${m.detections.length} 個牌角 → ${m.cards.length} 張牌（穩定 ${m.stable.length}）`;
  $("segInfo").textContent = st.settings.view === "corner" ? "" :
    `整張牌框：偵測到 ${(m.segs || []).length} 張（${m.seg_ms} ms）` + ((m.segs || []).length ? "　信心 " + m.segs.map((g) => pct(g.conf)).join(" / ") : "") +
    ((m.segs_rejected || []).length ? `　｜無牌角剔除 ${m.segs_rejected.length}` : "") +
    ((m.segs || []).some((g) => g.value) ? `　｜轉正辨識（${m.seg_value_ms} ms）：` +
      m.segs.map((g, i) => `#${i + 1} ${segValueText(g)}`).join(" ") : "");
  $("note").textContent = ""; $("note").className = "note";
  const dl = $("detList"); dl.innerHTML = "";
  [...m.detections].sort((x, y) => y.conf - x.conf).forEach((d) => {
    const div = document.createElement("div");
    const fx = d.det_label && d.det_label !== d.label ? ` <span style="color:#bb6bd9">（偵測 ${pretty(d.det_label)} → 分類改判）</span>` : "";
    const cl = d.cls_conf ? ` <span class="muted">分類 ${(d.cls_conf * 100).toFixed(0)}%</span>` : "";
    div.innerHTML = `<span>${pretty(d.label)}${fx}</span><span>${cl} <span style="color:${confColor(d.conf)}">${(d.conf * 100).toFixed(1)}%</span></span>`;
    dl.appendChild(div);
  });
  if (!st.truthDirty && document.activeElement !== $("truth")) $("truth").value = currentTruth();
}
const currentTruth = () => st.res ? ([...st.res.stable].sort().join("-") || "none") : "";

// ---------------------------------------------------------------- 存檔 / 評估
async function saveCapture() {
  const src = st.mode === "cam" ? video : still;
  const w = st.mode === "cam" ? video.videoWidth : still.naturalWidth, h = st.mode === "cam" ? video.videoHeight : still.naturalHeight;
  if (!w || (st.mode === "img" && st.fileIdx < 0)) return;
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  c.getContext("2d").drawImage(src, 0, 0);
  const truth = $("truth").value.trim() || "none";
  const r = await fetch("/api/capture", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: c.toDataURL("image/jpeg", 0.92), truth }) });
  const j = await r.json();
  const msg = $("saveMsg");
  if (r.ok) { msg.className = "small ok"; msg.textContent = `已存 ${j.saved}`; $("rtCount").textContent = `共 ${j.count} 張`; st.truthDirty = false; }
  else { msg.className = "small err"; msg.textContent = j.detail || "存檔失敗"; }
}
async function runEval() {
  const out = $("evalOut"); out.textContent = "評估中…";
  const r = await fetch("/api/evaluate", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: st.settings.model, conf: st.settings.conf, imgsz: st.settings.imgsz, ensemble: st.settings.ensemble, refine: st.settings.refine }) });
  const j = await r.json();
  if (!r.ok) { out.innerHTML = `<span class="err">${j.detail}</span>`; return; }
  out.innerHTML = `<table>
    <tr><td>照片數</td><td>${j.n}</td><td>整張全對</td><td>${pct(j.exact)}</td></tr>
    <tr><td>Precision</td><td>${pct(j.precision)}</td><td>Recall</td><td>${pct(j.recall)}</td></tr>
    <tr><td>F1</td><td>${pct(j.f1)}</td><td>推論</td><td>${j.ms.toFixed(1)} ms</td></tr>
    <tr><td>尺度</td><td colspan="3">${j.scales.join(" + ")} px</td></tr></table>
    ${j.errors.length ? "<p>錯誤樣本：</p>" + j.errors.map((e) => `<div class="muted">${e}</div>`).join("") : '<p class="ok">全部正確</p>'}`;
}
async function loadModels() {
  const j = await (await fetch("/api/models")).json();
  const sel = $("modelSel"); sel.innerHTML = "";
  const names = Object.keys(j.models);
  if (!names.length) { sel.add(new Option("（models/ 底下沒有權重）", "")); return; }
  names.forEach((n) => sel.add(new Option(n, n)));
  st.settings.model = st.settings.model || names[0]; sel.value = st.settings.model;
  const info = () => { const m = j.models[sel.value] || {}; $("modelInfo").textContent = m.mAP50 ? `合成 test：mAP50 ${m.mAP50.toFixed(3)} · mAP50-95 ${m["mAP50-95"].toFixed(3)}` : ""; };
  sel.onchange = () => { st.settings.model = sel.value; info(); sendSettings(); resendStill(); };
  info(); sendSettings();
  const rt = await (await fetch("/api/realtest")).json(); $("rtCount").textContent = `共 ${rt.count} 張`;
  $("reportMsg").textContent = rt.hard_cases ? `已回報 ${rt.hard_cases} 筆` : "";
}
const resendStill = () => { if (st.mode === "img" && st.fileIdx >= 0) { st.busy = false; sendFrame(); } };

// ---------------------------------------------------------------- 事件
function bindSetting(id, key, parse, outId) {
  $(id).addEventListener("input", (e) => {
    st.settings[key] = parse(e.target.value);
    if (outId) $(outId).textContent = e.target.value;
    sendSettings(); resendStill();
  });
}
bindSetting("conf", "conf", parseFloat, "confOut");
bindSetting("win", "window", (v) => parseInt(v), "winOut");
bindSetting("dist", "dist_factor", parseFloat, "distOut");
bindSetting("imgsz", "imgsz", (v) => parseInt(v));
$("ensemble").addEventListener("change", (e) => { st.settings.ensemble = e.target.checked; sendSettings(); resendStill(); });
$("refine").addEventListener("change", (e) => { st.settings.refine = e.target.checked; sendSettings(); resendStill(); });
$("viewSel").addEventListener("change", (e) => { st.settings.view = e.target.value; sendSettings(); resendStill(); });
$("segVerify").addEventListener("change", (e) => { st.settings.seg_verify = e.target.checked; sendSettings(); resendStill(); });
$("segValue").addEventListener("change", (e) => { st.settings.seg_value = e.target.checked; sendSettings(); resendStill(); });
function report() {
  if (!wsReady()) return;
  st.ws.send(JSON.stringify({ type: "cmd", cmd: "report", note: $("reportNote").value.trim() }));
}
$("reportBtn").onclick = report;

$("modeCam").onclick = () => setMode("cam");
$("modeImg").onclick = () => setMode("img");
$("camSel").onchange = startCam;
$("pauseBtn").onclick = () => { st.paused = !st.paused; $("pauseBtn").textContent = st.paused ? "▶ 繼續" : "⏸ 暫停"; };
$("fileIn").onchange = (e) => {
  st.files = [...e.target.files]; const th = $("thumbs"); th.innerHTML = "";
  st.files.forEach((f, i) => { const im = document.createElement("img"); im.src = URL.createObjectURL(f); im.onclick = () => showFile(i); th.appendChild(im); });
  th.hidden = st.files.length < 2; if (st.files.length) showFile(0);
};
$("resetBtn").onclick = () => sendCmd("reset");
$("saveBtn").onclick = saveCapture;
$("fillBtn").onclick = () => { st.truthDirty = false; $("truth").value = currentTruth(); };
$("truth").addEventListener("input", () => (st.truthDirty = true));
$("evalBtn").onclick = runEval;
window.addEventListener("resize", drawOverlay);
window.addEventListener("keydown", (e) => {
  const a = document.activeElement;
  if ((a.tagName === "INPUT" && a.type !== "range") || a.tagName === "SELECT") return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); $("pauseBtn").click(); }
  else if (k === "s") saveCapture();
  else if (k === "c") sendCmd("reset");
  else if (k === "e") report();
  else if (k === "arrowright" && st.mode === "img" && st.fileIdx < st.files.length - 1) showFile(st.fileIdx + 1);
  else if (k === "arrowleft" && st.mode === "img" && st.fileIdx > 0) showFile(st.fileIdx - 1);
});

connect();
loadModels();
setMode("cam");
requestAnimationFrame(loop);
