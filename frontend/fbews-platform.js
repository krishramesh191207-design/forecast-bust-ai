/* ═══════════════════════════════════════════════════════════════════════
   FBEWS platform module - shared selection state, shared metric semantics,
   and the Forecast Intelligence / Operations Center panes.

   Loaded BEFORE the main dashboard script so both share one global scope.
   Nothing here redefines a dashboard helper: where the dashboard already has
   a renderer (stabilityHTML, trajectorySVG, bandLabel, fmtPct, ...) this
   module calls it instead of duplicating it.
   ═══════════════════════════════════════════════════════════════════════ */

/* ───────────────────────── shared selection state ───────────────────── */
const FBP = {
  cycle: null,                 // "YYYY-MM-DD" or null (resolve from dashboard)
  lead: 3,
  region: null,                // analysis region id
  loc: {lat: 19.0, lon: 72.9, label: "Mumbai"},
  intelTab: "overview",
  opsTab: "situation",
  userRules: [],               // user alert rules (localStorage-backed)
  cache: new Map()
};
const FBP_SUBS = [];
function fbpSub(fn) { FBP_SUBS.push(fn); }
function setFBP(patch) {
  Object.assign(FBP, patch);
  FBP_SUBS.forEach(fn => { try { fn(FBP, patch); } catch (e) { console.warn("fbp", e); } });
}
function fbpCycle() {
  return FBP.cycle || (typeof currentCycle === "function" ? currentCycle() : null);
}
function fbpLead() {
  return (typeof S !== "undefined" && S && S.lead) ? S.lead : FBP.lead;
}
function fbpCache(key, fn) {
  if (FBP.cache.has(key)) return FBP.cache.get(key);
  const p = fn();
  FBP.cache.set(key, p);
  p.catch(() => FBP.cache.delete(key));
  return p;
}
async function fbpJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    let msg = r.status + " " + r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

/* ───────────────────────── metric semantics ─────────────────────────────
   ONE registry for "what does this number mean".  Mirrors
   fbews.derived.change.METRICS (label, units, decimals, higherIsBetter,
   threshold) and adds the human direction/risk wording every pane uses, so
   no pane can quietly disagree with another about a metric.
   ─────────────────────────────────────────────────────────────────────── */
const METRIC_SEM = {
  confidence: {
    key: "confidence", label: "Reliability Score", units: "0-100", decimals: 0,
    higherIsBetter: true, threshold: 5,
    direction: "Higher is better", unitSuffix: " points",
    valueRisk: v => (v >= 80 ? "high" : v >= 60 ? "moderate" : v >= 40 ? "low" :
                     v >= 20 ? "very_low" : "critical"),
    valueRiskLabel: v => (v >= 80 ? "High" : v >= 60 ? "Moderate" : v >= 40 ? "Low" :
                          v >= 20 ? "Very Low" : "High Bust Risk")
  },
  bust_probability: {
    key: "bust_probability", label: "Bust Probability", units: "0-1", decimals: 3,
    higherIsBetter: false, threshold: 0.05,
    direction: "Lower is better", unitSuffix: "",
    valueRisk: v => (v > 0.5 ? "critical" : v > 0.25 ? "high" : v > 0.1 ? "moderate" : "low"),
    valueRiskLabel: v => (v > 0.5 ? "Critical" : v > 0.25 ? "High" : v > 0.1 ? "Moderate" : "Low")
  },
  volatility: {
    key: "volatility", label: "Forecast Volatility Index", units: "index", decimals: 3,
    higherIsBetter: false, threshold: 0.1,
    direction: "Lower is better", unitSuffix: "",
    valueRisk: () => "unknown", valueRiskLabel: () => "—"
  },
  err_precip: {
    key: "err_precip", label: "Rainfall Error", units: "mm/day", decimals: 2,
    higherIsBetter: false, threshold: 0.5,
    direction: "Lower is better", unitSuffix: " mm/day",
    valueRisk: () => "unknown", valueRiskLabel: () => "—"
  },
  err_t2m: {
    key: "err_t2m", label: "Temperature Error", units: "K", decimals: 2,
    higherIsBetter: false, threshold: 0.2,
    direction: "Lower is better", unitSuffix: " K",
    valueRisk: () => "unknown", valueRiskLabel: () => "—"
  },
  err_wind: {
    key: "err_wind", label: "Wind Error", units: "m/s", decimals: 2,
    higherIsBetter: false, threshold: 0.5,
    direction: "Lower is better", unitSuffix: " m/s",
    valueRisk: () => "unknown", valueRiskLabel: () => "—"
  },
  err_mslp: {
    key: "err_mslp", label: "Pressure Error", units: "hPa", decimals: 2,
    higherIsBetter: false, threshold: 0.5,
    direction: "Lower is better", unitSuffix: " hPa",
    valueRisk: () => "unknown", valueRiskLabel: () => "—"
  },
  ens_spread_precip: {
    key: "ens_spread_precip", label: "Ensemble Rainfall Spread", units: "mm/day",
    decimals: 2, higherIsBetter: false, threshold: 0.5,
    direction: "Lower is better", unitSuffix: " mm/day",
    valueRisk: () => "unknown", valueRiskLabel: () => "—"
  },
  analogue_bust_rate: {
    key: "analogue_bust_rate", label: "Analogue Bust Rate", units: "0-1", decimals: 3,
    higherIsBetter: false, threshold: 0.05,
    direction: "Lower is better", unitSuffix: "",
    valueRisk: v => (v > 0.5 ? "critical" : v > 0.25 ? "high" : "moderate"),
    valueRiskLabel: v => (v > 0.5 ? "Critical" : v > 0.25 ? "High" : "Moderate")
  },
  fc_precip: {
    key: "fc_precip", label: "Forecast Rainfall", units: "mm/day", decimals: 2,
    higherIsBetter: null, threshold: null,
    direction: "No universal direction", unitSuffix: " mm/day",
    valueRisk: () => "unknown", valueRiskLabel: () => "—"
  }
};
/* Metrics whose cycle-to-cycle change is meaningful (mirrors change_is_defined). */
function semChangeDefined(key) {
  const m = METRIC_SEM[key];
  return !!(m && m.higherIsBetter !== null && m.threshold !== null);
}
function semFmt(key, v) {
  const m = METRIC_SEM[key] || {decimals: 2, units: ""};
  if (v === null || v === undefined || !isFinite(v)) return "—";
  if (key === "bust_probability" || key === "analogue_bust_rate") {
    return (v <= 1 ? Math.round(v * 100) + "%" : Math.round(v) + "%");
  }
  return Number(v).toFixed(m.decimals);
}
function semDelta(key, d) {
  if (d === null || d === undefined || !isFinite(d)) return "—";
  const m = METRIC_SEM[key] || {decimals: 2, units: ""};
  const sign = d > 0 ? "+" : "";
  if (key === "bust_probability" || key === "analogue_bust_rate") {
    const pp = Math.round(Math.abs(d) * 100);
    return (d > 0 ? "+" : d < 0 ? "−" : "") + pp + (Math.abs(d) <= 1 ? " pp" : "%");
  }
  if (key === "confidence") return sign + Number(d).toFixed(0) + " points";
  return sign + Number(d).toFixed(m.decimals) + m.unitSuffix;
}
/* Is a raw delta an improvement, a deterioration or noise? */
function semDirection(key, delta) {
  const m = METRIC_SEM[key];
  if (!m || m.higherIsBetter === null || m.threshold === null) return 0;
  if (Math.abs(delta) <= m.threshold) return 0;
  const better = m.higherIsBetter ? delta > 0 : delta < 0;
  return better ? 1 : -1;
}

/* ───────────────────────── tiny DOM helpers ─────────────────────────── */
function fbEsc(s) {
  return String(s === null || s === undefined ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
function fb$(sel, root) { return (root || document).querySelector(sel); }
function fbLoading(label) {
  return `<div class="fb-loading"><span class="boot-spinner"></span>
    <span>${fbEsc(label || "Loading…")}</span></div>`;
}
function fbEmpty(title, body) {
  return `<div class="fb-empty"><div class="fb-empty-t">${fbEsc(title)}</div>
    ${body ? `<div class="fb-empty-b">${body}</div>` : ""}</div>`;
}
function fbErr(title, e) {
  return `<div class="fb-empty fb-empty-err"><div class="fb-empty-t">${fbEsc(title)}</div>
    <div class="fb-empty-b">${fbEsc(e && e.message ? e.message : String(e))}</div></div>`;
}
function fbPill(text, cls) {
  return `<span class="fb-pill ${fbEsc(cls || "")}">${fbEsc(text)}</span>`;
}

/* ───────────────────────── sub-navigation ───────────────────────────── */
function fbSubnav(hostId, items, active, onPick) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const keys = items.map(i => i[0]).join(",");
  const sig = keys + "|" + active;
  if (host.dataset.sig === sig) return;              // nothing changed
  if (host.dataset.keys === keys) {                  // only the active tab moved
    Array.prototype.forEach.call(host.children, b => {
      if (b.dataset) b.classList.toggle("active", b.dataset.k === active);
    });
    host.dataset.sig = sig;
    return;
  }
  // Rebuilding the nav detaches the button under the pointer, so a click
  // issued while a render is in flight can be lost.  Only rebuild when the
  // tab list itself changes.
  host.dataset.keys = "";
  host.innerHTML = items.map(([k, label]) =>
    `<button class="fb-tab${k === active ? " active" : ""}" data-k="${fbEsc(k)}"
       onclick="fbpPick('${hostId}','${fbEsc(k)}')">${fbEsc(label)}</button>`).join("");
  host.dataset.keys = keys;
  host.dataset.sig = sig;
}
function fbpPick(hostId, key) {
  if (hostId === "intelTabs") {
    setFBP({intelTab: key});
    if (typeof fbpRenderIntel === "function") fbpRenderIntel();
  } else if (hostId === "opsTabs") {
    setFBP({opsTab: key});
    if (typeof fbpRenderOps === "function") fbpRenderOps();
  }
}

/* ───────────────────────── generic line chart ───────────────────────── */
/* series: [{label, points:[{x,y|null}], color, dashed}] */
function fbLineChart(series, opts) {
  opts = opts || {};
  const W = opts.width || 620, H = opts.height || 210;
  const pL = opts.padLeft || 44, pR = 12, pT = 16, pB = 30;
  const all = [];
  series.forEach(s => s.points.forEach(p => { if (p.y !== null && isFinite(p.y)) all.push(p.y); }));
  if (!all.length) return fbEmpty("No chart data", "Values unavailable for this selection.");
  let lo = opts.min !== undefined ? opts.min : Math.min(...all);
  let hi = opts.max !== undefined ? opts.max : Math.max(...all);
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const pad = (hi - lo) * 0.12;
  if (opts.min === undefined) lo -= pad;
  if (opts.max === undefined) hi += pad;
  const xs = opts.xs || series[0].points.map(p => p.x);
  const x = i => pL + (W - pL - pR) * (xs.length > 1 ? i / (xs.length - 1) : 0.5);
  const y = v => H - pB - (H - pB - pT) * (v - lo) / ((hi - lo) || 1);
  const gid = "fblc" + Math.random().toString(36).slice(2, 7);

  const grid = [0, 0.25, 0.5, 0.75, 1].map(f => {
    const yy = pT + (H - pB - pT) * f, v = hi - (hi - lo) * f;
    return `<line x1="${pL}" y1="${yy.toFixed(1)}" x2="${W - pR}" y2="${yy.toFixed(1)}"
      stroke="var(--bd)" stroke-dasharray="2 4"/>
      <text x="${pL - 6}" y="${(yy + 3.5).toFixed(1)}" text-anchor="end"
        fill="var(--t3)" font-size="10.5">${opts.fmtY ? opts.fmtY(v) : (Math.round(v * 100) / 100)}</text>`;
  }).join("");

  const paths = series.map((s, si) => {
    const segs = [];
    let cur = [];
    s.points.forEach((p, i) => {
      if (p.y === null || !isFinite(p.y)) { if (cur.length) segs.push(cur); cur = []; return; }
      cur.push(`${x(i).toFixed(1)},${y(p.y).toFixed(1)}`);
    });
    if (cur.length) segs.push(cur);
    const lines = segs.map(seg => `<polyline points="${seg.join(" ")}" fill="none"
      stroke="${s.color}" stroke-width="${s.width || 2}" stroke-linejoin="round"
      stroke-linecap="round"${s.dashed ? ' stroke-dasharray="5 4"' : ""}/>`).join("");
    const dots = s.points.map((p, i) => (p.y === null || !isFinite(p.y)) ? "" :
      `<circle cx="${x(i).toFixed(1)}" cy="${y(p.y).toFixed(1)}" r="3.4"
         fill="${s.color}" stroke="var(--bg2)" stroke-width="1.4"><title>${fbEsc(s.label)} ${fbEsc(xs[i])}: ${p.y}</title></circle>`).join("");
    return lines + dots;
  }).join("");

  const labels = xs.map((l, i) => {
    const anc = i === 0 ? "start" : (i === xs.length - 1 ? "end" : "middle");
    return `<text x="${x(i).toFixed(1)}" y="${H - 9}" text-anchor="${anc}"
      fill="var(--t3)" font-size="10.5">${fbEsc(opts.fmtX ? opts.fmtX(l) : l)}</text>`;
  }).join("");

  const legend = series.map(s => `<span class="fb-legend-i" style="--c:${s.color}">
    <i></i>${fbEsc(s.label)}</span>`).join("");

  return `<div class="fb-chart">
    <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block"
      role="img" aria-label="${fbEsc(opts.aria || "chart")}">
      <defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#FF8A00" stop-opacity=".20"/>
        <stop offset="100%" stop-color="#FF8A00" stop-opacity="0"/>
      </linearGradient></defs>
      ${grid}${paths}${labels}
    </svg>
    <div class="fb-legend">${legend}</div>
  </div>`;
}

/* ───────────────────────── shared fetchers ──────────────────────────── */
function fbRegionsPayload() {
  return fbpCache("regions|" + (fbpCycle() || ""),
    () => fbpJSON(`/api/regions?cycle=${fbpCycle() || ""}`));
}
function fbComparison(metric, lead) {
  const c = fbpCycle(), l = lead || fbpLead();
  return fbpCache(`cmp|${c}|${l}|${metric}`,
    () => fbpJSON(`/api/forecast-comparison?cycle=${c || ""}&lead=${l}&metric=${metric}`));
}
function fbStabilityRegions(lead) {
  const c = fbpCycle(), l = lead || fbpLead();
  return fbpCache(`stabR|${c}|${l}`,
    () => fbpJSON(`/api/forecast-stability-regions?cycle=${c || ""}&lead=${l}`));
}
function fbEnsemble(lat, lon, lead) {
  const c = fbpCycle(), l = lead || fbpLead();
  return fbpCache(`ens|${c}|${l}|${lat.toFixed(2)}|${lon.toFixed(2)}`,
    () => fbpJSON(`/api/ensemble?cycle=${c || ""}&lead=${l}&lat=${lat}&lon=${lon}`));
}
function fbGridPoint(lat, lon, lead) {
  const c = fbpCycle(), l = lead || fbpLead();
  return fbpCache(`pt|${c}|${l}|${lat.toFixed(2)}|${lon.toFixed(2)}`,
    () => fbpJSON(`/api/grid/${lat}/${lon}?cycle=${c || ""}&lead=${l}`));
}
function fbAnalogs(lat, lon, lead) {
  const c = fbpCycle(), l = lead || fbpLead();
  return fbpCache(`ana|${c}|${l}|${lat.toFixed(2)}|${lon.toFixed(2)}`,
    () => fbpJSON(`/api/analogs?cycle=${c || ""}&lead=${l}&lat=${lat}&lon=${lon}`));
}

/* Region id -> {name, centroid} from /api/regions definitions. */
function fbRegionIndex(defs) {
  const out = {};
  (defs || []).forEach(r => {
    const bb = r.bbox;
    if (!bb || bb.length !== 4) return;
    out[r.id] = {
      id: r.id, name: r.name, kind: r.kind, bbox: bb,
      lat: (bb[0] + bb[1]) / 2, lon: (bb[2] + bb[3]) / 2
    };
  });
  return out;
}
function fbRegionName(id) {
  if (!id) return "—";
  const r = (FBP.regionIndex || {})[id];
  return r ? r.name : id;
}
function fbRegionSelect(id, idx) {
  const opts = Object.values(idx || FBP.regionIndex || {}).map(r =>
    `<option value="${fbEsc(r.id)}"${r.id === FBP.region ? " selected" : ""}>${fbEsc(r.name)}</option>`).join("");
  return `<select class="fb-select" id="${id}" onchange="fbpPickRegion(this.value)">${opts}</select>`;
}
function fbpPickRegion(v) {
  setFBP({region: v});
  const r = (FBP.regionIndex || {})[v];
  if (r) setFBP({loc: {lat: r.lat, lon: r.lon, label: r.name}});
  fbpRenderActive();
}
function fbpRenderActive() {
  const active = document.querySelector(".view.active");
  if (!active) return;
  if (active.id === "view-intel" && typeof fbpRenderIntel === "function") fbpRenderIntel();
  if (active.id === "view-ops" && typeof fbpRenderOps === "function") fbpRenderOps();
}
/* Lead changes raised from inside the panes must not yank the user back to
   the Overview the way the map's own lead bar does. */
function fbSetLead(l) {
  if (typeof S !== "undefined") S.lead = l;
  document.querySelectorAll(".lead-btn").forEach(b =>
    b.classList.toggle("on", +b.dataset.l === l));
  FBP.cache.clear();
  setFBP({lead: l});
  if (typeof refreshDerived === "function") { try { refreshDerived(); } catch (e) { console.warn(e); } }
  fbpRenderActive();
}
function fbpOnLeadChange() { FBP.cache.clear(); setFBP({lead: fbpLead()}); fbpRenderActive(); }
function fbpOnCycleChange() { FBP.cache.clear(); setFBP({cycle: fbpCycle()}); fbpRenderActive(); }

/* ───────────────────────── stale-render guard ─────────────────────────
   Every pane renderer is async. Without a guard a slow earlier request can
   overwrite a faster later one and leave the previous pane's markup on
   screen. Each render bumps FB_TOK.n; innerHTML writes from an older render
   are dropped. */
const FB_TOK = {n: 0};
function fbGuardedEl(el, tok) {
  if (!el) return el;
  return new Proxy(el, {
    set(t, p, v) {
      if (p === "innerHTML") {
        if (tok === FB_TOK.n) t.innerHTML = v;
        return true;
      }
      t[p] = v;
      return true;
    },
    get(t, p) {
      const v = t[p];
      return typeof v === "function" ? v.bind(t) : v;
    }
  });
}
