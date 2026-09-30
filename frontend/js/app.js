const COL = {LOW: "#2e9e4f", MODERATE: "#f2b01e", HIGH: "#d7301f"};
let metrics = {}, layer = null;
const $ = id => document.getElementById(id);
const map = L.map("map").setView([21, 82], 5);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {attribution: "&copy; OpenStreetMap"}).addTo(map);
const occLayer = L.layerGroup().addTo(map);
const lg = L.control({position: "bottomright"});
lg.onAdd = () => { const d = L.DomUtil.create("div", "legend"); d.innerHTML = "<b>Predicted Manganese Prospectivity</b><br>" +
  ["LOW", "MODERATE", "HIGH"].map(k => `<i style="background:${COL[k]}"></i>${k} PROSPECTIVITY`).join("<br>"); return d; };
lg.addTo(map);
function msg(t, ok) { const m = $("msg"); m.hidden = !t; m.textContent = t || ""; m.className = "msg" + (ok ? " ok" : ""); if (t) scrollTo(0, 0); }
async function api(url, body) {
  const r = await fetch(url, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const j = await r.json(); if (!r.ok) { msg(j.error || "Request failed"); throw new Error(j.error); } msg(""); return j;
}
async function busy(btn, fn) { btn.disabled = true; const t = btn.textContent; btn.textContent = "Working..."; try { await fn(); } catch (e) {} btn.disabled = false; btn.textContent = t; }
async function init() {
  const s = await api("/api/status");
  $("period").textContent = s.s2_period.join(" to ");
  $("status").innerHTML = `GSI dataset: <b>${s.dataset_present ? "found" : "MISSING"}</b> | Earth Engine project: <b>${s.ee_project_set ? "set" : "NOT set"}</b>`;
  if (s.message) msg(s.message);
  if (s.features_built) api("/api/summary").then(j => $("summary").textContent = JSON.stringify(j, null, 2));
  if (s.metrics_ready) api("/api/metrics").then(showMetrics);
  if (s.prediction_ready) showMap();
  if (s.dataset_present) loadOcc();
}
async function loadOcc() {
  const j = await api("/api/occurrences"); occLayer.clearLayers();
  j.points.forEach(p => L.circleMarker([p.lat, p.lon], {radius: 4, color: "#900", weight: 1, fillOpacity: .8})
    .bindPopup(`<b>${p.locality}</b><br>${p.state}<br>${p.host_rock}`).addTo(occLayer));
  $("occ").textContent = JSON.stringify(j.report, null, 2);
  map.fitBounds(L.latLngBounds(j.points.map(p => [p.lat, p.lon])));
}
function buildFeat() { busy(event.target, async () => { $("summary").textContent = JSON.stringify(await api("/api/build-features", {state: $("state").value}), null, 2); }); }
function train() { busy(event.target, async () => showMetrics(await api("/api/train", {models: [...document.querySelectorAll(".m:checked")].map(c => c.value)}))); }
function showMetrics(m) {
  metrics = m; const f = v => v == null ? "n/a" : v.toFixed(3);
  $("metrics").innerHTML = "<table><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>Confusion [[TN,FP],[FN,TP]]</th></tr>" +
    Object.entries(m).map(([k, v]) => `<tr><td>${k}</td><td>${f(v.accuracy)}</td><td>${f(v.precision)}</td><td>${f(v.recall)}</td><td>${f(v.f1)}</td><td>${f(v.roc_auc)}</td><td>${JSON.stringify(v.confusion_matrix)}</td></tr>`).join("") + "</table>";
  $("fiModel").innerHTML = Object.keys(m).map(k => `<option>${k}</option>`).join(""); showFI();
}
function showFI() {
  const m = metrics[$("fiModel").value]; if (!m) return;
  const e = Object.entries(m.feature_importance).sort((a, b) => b[1] - a[1]), mx = Math.max(...e.map(x => x[1]), 1e-9);
  $("fi").innerHTML = e.map(([k, v]) => `<div>${k.padEnd(12, "\u00a0")} <span class="bar" style="width:${Math.max(0, v) / mx * 300}px"></span> ${v.toFixed(3)}</div>`).join("");
}
function predict() { busy(event.target, async () => { const r = await api("/api/predict", {model: $("pm").value}); $("pinfo").textContent = `${r.cells} cells: ` + JSON.stringify(r.counts); await showMap(); }); }
async function showMap() {
  const g = await api("/api/prospectivity"); if (layer) map.removeLayer(layer);
  layer = L.geoJSON(g, {style: f => ({color: COL[f.properties.class], weight: 0, fillOpacity: .5}),
    onEachFeature: (f, l) => l.bindPopup(`${f.properties.class} (p=${f.properties.probability})`)}).addTo(map);
  occLayer.bringToFront && occLayer.eachLayer(l => l.bringToFront());
}
init();
