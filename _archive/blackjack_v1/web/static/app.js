"use strict";
const $ = (id) => document.getElementById(id);
const SUIT = { c: "♣", d: "♦", h: "♥", s: "♠" };
const ACT = { stand: "停牌", hit: "要牌", double: "加倍" };
const STRAT_COLOR = { H: "#56ccf2", S: "#3ccf7a", D: "#f2c14e", P: "#bb6bd9" };
const SEND_MAX_W = 960;        // 傳給後端的影格寬度上限（模型輸入 416，不需全解析度）

const st = {
  mode: "cam", paused: false, ws: null, busy: false, sentAt: 0, fps: 0,
  res: null, stream: null, files: [], fileIdx: -1, truthDirty: false,
  settings: { model: null, conf: 0.45, window: 8, decks: 6, h17: false, peek: true, dealer_top: true, split: 0.45, smooth: true },
};
const video = $("video"), still = $("still"), overlay = $("overlay"), ctx = overlay.getContext("2d");
const grab = document.createElement("canvas");

// ---------------------------------------------------------------- 工具
const pretty = (l) => l.slice(0, -1) + SUIT[l.slice(-1)];
const isRed = (l) => "hd".includes(l.slice(-1));
const pct = (x) => (x * 100).toFixed(0) + "%";
function chip(l) { const s = document.createElement("span"); s.className = "pc" + (isRed(l) ? " red" : ""); s.textContent = pretty(l); return s; }

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
  };
  st.ws = ws;
}
const wsReady = () => st.ws && st.ws.readyState === 1;
function sendSettings() { if (wsReady()) st.ws.send(JSON.stringify({ type: "settings", ...st.settings, smooth: st.mode === "cam" })); }
function sendCmd(cmd) { if (wsReady()) st.ws.send(JSON.stringify({ type: "cmd", cmd })); }

function onState(m) {
  $("shoeInfo").textContent = `已記 ${m.seen} 張 · 真數 ${m.true_count >= 0 ? "+" : ""}${m.true_count.toFixed(1)}`;
  if (!st.settings.model && m.settings.model) { st.settings.model = m.settings.model; $("modelSel").value = m.settings.model; sendSettings(); }
}

function onResult(m) {
  st.busy = false;
  if (m.error) { $("note").textContent = m.error; $("note").className = "note err"; return; }
  const dt = performance.now() - st.sentAt;
  st.fps = st.fps ? st.fps * 0.85 + (1000 / dt) * 0.15 : 1000 / dt;
  $("perf").textContent = `推論 ${m.infer_ms} ms · ${st.fps.toFixed(1)} FPS`;
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
  if (!w || !h) return { s: 1, x: 0, y: 0, w: cw, h: ch, cw, ch, iw: cw, ih: ch };
  const s = Math.min(cw / w, ch / h);
  return { s, x: (cw - w * s) / 2, y: (ch - h * s) / 2, w: w * s, h: h * s, cw, ch, iw: w, ih: h };
}
function drawOverlay() {
  const dpr = window.devicePixelRatio || 1;
  overlay.width = overlay.clientWidth * dpr; overlay.height = overlay.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight);
  const r = contentRect();
  const ly = r.y + st.settings.split * r.h;
  ctx.strokeStyle = "#f2c14e"; ctx.lineWidth = 2; ctx.setLineDash([8, 6]);
  ctx.beginPath(); ctx.moveTo(r.x, ly); ctx.lineTo(r.x + r.w, ly); ctx.stroke(); ctx.setLineDash([]);
  ctx.font = "bold 13px 'Microsoft JhengHei', sans-serif"; ctx.fillStyle = "#f2c14e";
  const top = st.settings.dealer_top ? "莊家區" : "玩家區", bot = st.settings.dealer_top ? "玩家區" : "莊家區";
  ctx.fillText(`▲ ${top}`, r.x + 8, ly - 8); ctx.fillText(`▼ ${bot}`, r.x + 8, ly + 18);
  if (!st.res) return;
  for (const d of st.res.detections) {
    const [x1, y1, x2, y2] = d.box;
    ctx.strokeStyle = "rgba(60,207,122,.9)"; ctx.lineWidth = 2;
    ctx.strokeRect(r.x + x1 * r.s, r.y + y1 * r.s, (x2 - x1) * r.s, (y2 - y1) * r.s);
  }
  ctx.font = "bold 22px 'Segoe UI Symbol', 'Microsoft JhengHei', sans-serif";
  for (const c of st.res.cards) {
    const x = r.x + c.center[0] * r.s, y = r.y + c.center[1] * r.s, t = pretty(c.label);
    const tw = ctx.measureText(t).width + 14;
    ctx.fillStyle = c.zone === "dealer" ? "#f2994a" : "#56ccf2";
    ctx.beginPath(); ctx.roundRect(x - tw / 2, y - 16, tw, 30, 6); ctx.fill();
    ctx.fillStyle = "#111"; ctx.textAlign = "center"; ctx.fillText(t, x, y + 7); ctx.textAlign = "left";
    ctx.font = "11px sans-serif"; ctx.fillStyle = "#fff";
    ctx.fillText(`${(c.conf * 100).toFixed(0)}%·${c.corners}角`, x - tw / 2, y + 28);
    ctx.font = "bold 22px 'Segoe UI Symbol', 'Microsoft JhengHei', sans-serif";
  }
}

// ---------------------------------------------------------------- 面板
function renderPanel(m) {
  const fill = (el, labels) => { el.innerHTML = ""; labels.length ? labels.forEach((l) => el.appendChild(chip(l))) : (el.textContent = "—"); };
  fill($("dealerCards"), m.dealer); fill($("playerCards"), m.player);
  const a = m.advice;
  $("total").textContent = a.total ? (a.soft ? "軟 " : "") + a.total : "";
  const s = $("strategy");
  if (a.strategy) { s.textContent = `基本策略：${a.strategy_name}`; s.style.color = STRAT_COLOR[a.strategy]; }
  else { s.textContent = a.total > 21 ? "爆牌" : "等待牌面"; s.style.color = ""; }
  const box = $("odds"); box.innerHTML = "";
  for (const k of ["stand", "hit", "double"]) {
    const o = a.odds[k]; if (!o) continue;
    const row = document.createElement("div");
    row.className = "act" + (a.best === k ? " best" : "");
    row.innerHTML = `<span class="name">${a.best === k ? "★ " : ""}${ACT[k]}${k === "double" ? "<small>（×2）</small>" : ""}</span>
      <div class="bar"><span class="w" style="flex:${o.win}">${o.win > .12 ? "勝 " + pct(o.win) : ""}</span><span class="p" style="flex:${o.push}">${o.push > .12 ? pct(o.push) : ""}</span><span class="l" style="flex:${o.lose}">${o.lose > .12 ? "負 " + pct(o.lose) : ""}</span></div>
      <span class="ev">EV ${o.ev >= 0 ? "+" : ""}${o.ev.toFixed(3)}</span>`;
    row.title = `勝 ${pct(o.win)} · 和 ${pct(o.push)} · 負 ${pct(o.lose)}`;
    box.appendChild(row);
  }
  $("note").textContent = a.note || ""; $("note").className = "note";
  $("shoeInfo").textContent = `已記 ${m.seen} 張 · 真數 ${m.true_count >= 0 ? "+" : ""}${m.true_count.toFixed(1)}`;
  const dl = $("detList"); dl.innerHTML = "";
  [...m.detections].sort((x, y) => y.conf - x.conf).forEach((d) => {
    const div = document.createElement("div"); div.innerHTML = `<span>${pretty(d.label)}</span><span>${(d.conf * 100).toFixed(1)}%</span>`; dl.appendChild(div);
  });
  $("detCount").textContent = `${m.detections.length} 個牌角 → ${m.cards.length} 張牌`;
  if (!st.truthDirty && document.activeElement !== $("truth")) $("truth").value = currentTruth();
}
const currentTruth = () => st.res ? ([...st.res.dealer, ...st.res.player].sort().join("-") || "none") : "";

// ---------------------------------------------------------------- 存檔 / 評估
async function fileDataURL(f) { return new Promise((ok) => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(f); }); }
async function saveCapture() {
  let img;
  if (st.mode === "cam") {
    const c = document.createElement("canvas"); c.width = video.videoWidth; c.height = video.videoHeight;
    if (!c.width) return;
    c.getContext("2d").drawImage(video, 0, 0); img = c.toDataURL("image/jpeg", 0.92);
  } else {
    if (st.fileIdx < 0) return;
    const c = document.createElement("canvas"); c.width = still.naturalWidth; c.height = still.naturalHeight;
    c.getContext("2d").drawImage(still, 0, 0); img = c.toDataURL("image/jpeg", 0.92);
  }
  const truth = $("truth").value.trim() || "none";
  const r = await fetch("/api/capture", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: img, truth }) });
  const j = await r.json();
  const msg = $("saveMsg");
  if (r.ok) { msg.className = "small ok"; msg.textContent = `已存 ${j.saved}`; $("rtCount").textContent = `共 ${j.count} 張`; st.truthDirty = false; }
  else { msg.className = "small err"; msg.textContent = j.detail || "存檔失敗"; }
}
async function runEval() {
  const out = $("evalOut"); out.textContent = "評估中…";
  const r = await fetch("/api/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: st.settings.model, conf: st.settings.conf }) });
  const j = await r.json();
  if (!r.ok) { out.innerHTML = `<span class="err">${j.detail}</span>`; return; }
  out.innerHTML = `<table>
    <tr><td>照片數</td><td>${j.n}</td><td>整張全對</td><td>${pct(j.exact)}</td></tr>
    <tr><td>Precision</td><td>${pct(j.precision)}</td><td>Recall</td><td>${pct(j.recall)}</td></tr>
    <tr><td>F1</td><td>${pct(j.f1)}</td><td>推論</td><td>${j.ms.toFixed(1)} ms</td></tr></table>
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
}
const resendStill = () => { if (st.mode === "img" && st.fileIdx >= 0) { st.busy = false; sendFrame(); } };

// ---------------------------------------------------------------- 事件
function bindSetting(id, key, parse, outId) {
  $(id).addEventListener("input", (e) => {
    const el = e.target; st.settings[key] = parse(el.type === "checkbox" ? el.checked : el.value);
    if (outId) $(outId).textContent = el.value;
    sendSettings(); resendStill(); drawOverlay();
  });
}
bindSetting("conf", "conf", parseFloat, "confOut");
bindSetting("win", "window", (v) => parseInt(v), "winOut");
bindSetting("decks", "decks", (v) => parseInt(v));
bindSetting("h17", "h17", Boolean);
bindSetting("peek", "peek", Boolean);
bindSetting("dealerTop", "dealer_top", Boolean);

let dragging = false;
function setSplitFromEvent(e) {
  const r = contentRect(), rect = overlay.getBoundingClientRect();
  st.settings.split = Math.min(0.95, Math.max(0.05, (e.clientY - rect.top - r.y) / r.h));
  drawOverlay();
}
overlay.addEventListener("pointerdown", (e) => { dragging = true; overlay.setPointerCapture(e.pointerId); setSplitFromEvent(e); });
overlay.addEventListener("pointermove", (e) => dragging && setSplitFromEvent(e));
overlay.addEventListener("pointerup", () => { dragging = false; sendSettings(); resendStill(); });

$("modeCam").onclick = () => setMode("cam");
$("modeImg").onclick = () => setMode("img");
$("camSel").onchange = startCam;
$("pauseBtn").onclick = () => { st.paused = !st.paused; $("pauseBtn").textContent = st.paused ? "▶ 繼續" : "⏸ 暫停"; };
$("fileIn").onchange = (e) => {
  st.files = [...e.target.files]; const th = $("thumbs"); th.innerHTML = "";
  st.files.forEach((f, i) => { const im = document.createElement("img"); im.src = URL.createObjectURL(f); im.onclick = () => showFile(i); th.appendChild(im); });
  th.hidden = st.files.length < 2; if (st.files.length) showFile(0);
};
$("nextBtn").onclick = () => sendCmd("next_round");
$("resetShoeBtn").onclick = () => sendCmd("reset_shoe");
$("saveBtn").onclick = saveCapture;
$("fillBtn").onclick = () => { st.truthDirty = false; $("truth").value = currentTruth(); };
$("truth").addEventListener("input", () => (st.truthDirty = true));
$("evalBtn").onclick = runEval;
window.addEventListener("resize", drawOverlay);
window.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT"].includes(document.activeElement.tagName) && document.activeElement.type !== "range" && document.activeElement.type !== "checkbox") return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); $("pauseBtn").click(); }
  else if (k === "n") sendCmd("next_round");
  else if (k === "r") sendCmd("reset_shoe");
  else if (k === "s") saveCapture();
  else if (k === "f") { $("dealerTop").click(); }
  else if (k === "arrowright" && st.mode === "img" && st.fileIdx < st.files.length - 1) showFile(st.fileIdx + 1);
  else if (k === "arrowleft" && st.mode === "img" && st.fileIdx > 0) showFile(st.fileIdx - 1);
});

connect();
loadModels();
setMode("cam");
requestAnimationFrame(loop);
