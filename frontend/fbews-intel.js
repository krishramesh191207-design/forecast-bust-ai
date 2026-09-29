/* ═══════════════════════════════════════════════════════════════════════
   FORECAST INTELLIGENCE pane
   Tabs: Overview · Drift · Stability · Ensemble · Analogues · Drivers ·
         Scenarios · Risk Evolution
   Reuses the dashboard's own renderers wherever they already exist:
   stabilityHTML, trajectorySVG, bandLabel, fmtPct, show/setMode.
   ═══════════════════════════════════════════════════════════════════════ */

const FBP_INTEL_TABS = [
  ["overview", "Overview"],
  ["drift", "Drift"],
  ["stability", "Stability"],
  ["ensemble", "Ensemble"],
  ["analogues", "Analogues"],
  ["drivers", "Drivers"],
  ["scenarios", "Scenarios"],
  ["risk", "Risk Evolution"]
];

/* Metrics offered on the Drift tab - only those whose change is defined. */
const FBP_DRIFT_METRICS = ["confidence", "bust_probability", "err_precip",
  "err_t2m", "err_wind", "err_mslp", "ens_spread_precip", "volatility",
  "analogue_bust_rate"];

/* ── shared aggregation ──────────────────────────────────────────────────
   Mean direction-signed delta per region (bounding boxes) and per state
   (the dashboard's own boundary index).  Same sign convention as the map:
   positive always means "towards better".                            */
function fbAggregate(cmp, groupBy) {
  if (!cmp || !cmp.available || !cmp.change || !S.products) return null;
  const g = S.products.grid;
  const dir = (cmp.higherIsBetter === false) ? -1 : 1;
  const out = new Map();
  const defs = groupBy === "region"
    ? Object.values(FBP.regionIndex || {})
    : null;
  if (groupBy === "state" && typeof B !== "undefined" && !B.cellState
      && typeof buildCellStateIndex === "function") buildCellStateIndex();
  let k = 0;
  for (let i = 0; i < g.lat.length; i++) {
    for (let j = 0; j < g.lon.length; j++, k++) {
      const v = cmp.change[k];
      if (v === null || v === undefined || !isFinite(v)) continue;
      const la = g.lat[i], lo = g.lon[j];
      let key = null;
      if (groupBy === "region") {
        for (const d of defs) {
          if (la >= d.bbox[0] && la <= d.bbox[1] && lo >= d.bbox[2] && lo <= d.bbox[3]) {
            key = d.id; break;
          }
        }
      } else {
        const si = (B.cellState && B.cellState[k] >= 0) ? B.cellState[k] : -1;
        key = si < 0 ? null : ("s" + si);
      }
      if (key === null) continue;
      const e = out.get(key) || {sum: 0, n: 0};
      e.sum += dir * v; e.n++;
      out.set(key, e);
    }
  }
  const list = [];
  out.forEach((e, key) => {
    list.push({key, mean: e.sum / e.n, n: e.n,
      name: groupBy === "state"
        ? (B.features[+key.slice(1)] || {}).name || key
        : fbRegionName(key)});
  });
  list.sort((a, b) => a.mean - b.mean);   // worst first
  return list;
}

function fbStatusPill(cmp, meanDelta) {
  /* meanDelta is already direction-signed (positive = better). */
  const th = cmp && cmp.threshold != null ? cmp.threshold : 0;
  if (Math.abs(meanDelta) <= th) return fbPill("No meaningful change", "neutral");
  return meanDelta > 0 ? fbPill("Improved", "up") : fbPill("Deteriorated", "down");
}

/* ── state-level drift choropleth (SVG, reuses the dashboard boundary index) ── */
function fbDriftSVG(cmp, list) {
  if (!list || !list.length) return "";
  const g = S.products.grid;
  const latMin = Math.min(...g.lat), latMax = Math.max(...g.lat);
  const lonMin = Math.min(...g.lon), lonMax = Math.max(...g.lon);
  const W = 560, H = 460, pad = 10;
  const sx = lo => pad + (W - 2 * pad) * (lo - lonMin) / ((lonMax - lonMin) || 1);
  const sy = la => pad + (H - 2 * pad) * (latMax - la) / ((latMax - latMin) || 1);
  const byKey = {};
  list.forEach(e => { byKey[e.key] = e; });
  const mag = Math.max(...list.map(e => Math.abs(e.mean))) || 1;
  const th = cmp.threshold || 0;

  const paths = B.features.map((f, i) => {
    const e = byKey["s" + i];
    let fill = "var(--bg4)";
    if (e) {
      if (Math.abs(e.mean) <= th) fill = scaleColor(DIVERGE.stops, 0.5);
      else {
        const t = (Math.max(-1, Math.min(1, e.mean / mag)) + 1) / 2;
        fill = scaleColor(DIVERGE.stops, t);
      }
    }
    const d = f.rings.map(r =>
      r.map((p, n) => `${n ? "L" : "M"}${sx(p[0]).toFixed(1)} ${sy(p[1]).toFixed(1)}`).join("") + "Z"
    ).join(" ");
    const title = e
      ? `${f.name}: ${semDelta(cmp.metric, e.mean)} (n=${e.n} cells)`
      : `${f.name}: no comparable data`;
    return `<path d="${d}" fill="${fill}" stroke="var(--bg)" stroke-width="0.8">
      <title>${fbEsc(title)}</title></path>`;
  }).join("");

  const legend = `<div class="fb-drift-scale">
      <span>Deteriorated</span>
      <i style="background:linear-gradient(90deg,rgb(${DIVERGE.stops.map(s => s.join(",")).join("),rgb(")}))"></i>
      <span>Improved</span>
    </div>
    <div class="fb-drift-note">${fbEsc(cmp.label)} · ${fbEsc(cmp.direction || METRIC_SEM[cmp.metric].direction)}
      · |Δ| ≤ ${cmp.threshold} counts as no meaningful change ·
      ${cmp.cycle} Day ${cmp.lead} vs ${cmp.previousCycle || "—"} Day ${cmp.previousLead || "—"}
      · same valid time ${cmp.validTime || "—"}</div>`;

  return `<div class="fb-driftmap"><svg viewBox="0 0 ${W} ${H}"
    style="width:100%;height:auto;display:block" role="img"
    aria-label="State level forecast drift map">${paths}</svg>${legend}</div>`;
}

function fbUnavailableBox(cmp) {
  const leads = (cmp && cmp.availableLeads && cmp.availableLeads.length)
    ? `<div class="fb-empty-b">Comparable leads: ${cmp.availableLeads.map(l => "Day " + l).join(", ")}.</div>`
    : "";
  return fbEmpty("Forecast Change unavailable",
    `${fbEsc((cmp && cmp.message) || "Comparison unavailable for this selection.")}${leads}
     <div class="fb-cta"><button class="fb-btn" onclick="fbpPick('intelTabs','stability')">Open Stability instead</button></div>`);
}

/* ═════════════════════════ TAB: OVERVIEW ═════════════════════════ */
async function fbIntelOverview(el) {
  el.innerHTML = fbLoading("Building Forecast Intelligence overview…");
  const [cmpR, stabR, regsR] = await Promise.allSettled([
    fbComparison("confidence"), fbStabilityRegions(), fbRegionsPayload()
  ]);
  if (regsR.status === "fulfilled") {
    FBP.regionIndex = fbRegionIndex(regsR.value.definitions);
    if (!FBP.region) {
      const first = Object.keys(FBP.regionIndex)[0];
      if (first) { FBP.region = first; const r = FBP.regionIndex[first];
        FBP.loc = {lat: r.lat, lon: r.lon, label: r.name}; }
    }
  }
  const cmp = cmpR.status === "fulfilled" ? cmpR.value : null;
  const stab = stabR.status === "fulfilled" ? stabR.value : null;
  const regs = regsR.status === "fulfilled" ? regsR.value : null;

  const err = [];
  if (cmpR.status === "rejected") err.push("Forecast Change: " + cmpR.reason.message);
  if (stabR.status === "rejected") err.push("Stability: " + stabR.reason.message);
  if (regsR.status === "rejected") err.push("Regions: " + regsR.reason.message);
  if (err.length) { el.innerHTML = fbErr("Forecast Intelligence unavailable", new Error(err.join(" · "))); return; }

  /* --- change summary ------------------------------------------------ */
  let changeBlock;
  if (cmp.available && cmp.change) {
    const counts = {0: 0, 1: 0, 2: 0, 3: 0};
    cmp.change.forEach(() => {});
    let total = 0;
    for (let k = 0; k < cmp.change.length; k++) {
      const v = cmp.change[k];
      if (v === null || v === undefined || !isFinite(v)) { counts[3]++; continue; }
      const st = semDirection(cmp.metric, v);
      counts[st === 1 ? 1 : st === -1 ? 2 : 0]++;
      total++;
    }
    const pct = n => total ? Math.round(100 * n / total) : 0;
    changeBlock = `
      <div class="fb-bars">
        ${[["Improved", counts[1], "up"], ["No meaningful change", counts[0], "flat"],
           ["Deteriorated", counts[2], "down"], ["No comparable data", counts[3], "na"]]
          .map(([lab, n, cls]) => `
          <div class="fb-bar-row">
            <div class="fb-bar-lab">${lab}</div>
            <div class="fb-bar-track"><div class="fb-bar-fill ${cls}"
              style="width:${pct(n)}%"></div></div>
            <div class="fb-bar-n">${n}<small> (${pct(n)}%)</small></div>
          </div>`).join("")}
      </div>
      <div class="fb-note">${fbEsc(cmp.label)} · Day ${cmp.lead} of ${cmp.cycle} vs
        Day ${cmp.previousLead} of ${cmp.previousCycle} · same valid time ${cmp.validTime} ·
        |Δ| ≤ ${cmp.threshold} = no meaningful change ·
        ${total.toLocaleString()} comparable grid cells</div>`;
  } else {
    changeBlock = fbUnavailableBox(cmp);
  }

  /* --- stability summary -------------------------------------------- */
  let stabBlock;
  if (stab && stab.regions && stab.regions.length) {
    const by = {};
    stab.regions.forEach(r => {
      const c = (r.readout && r.readout.stability && r.readout.stability.classification) || "unknown";
      by[c] = (by[c] || 0) + 1;
    });
    const order = ["deteriorating", "unstable", "improving", "stable"];
    stabBlock = `<div class="fb-chips">${order.filter(k => by[k]).map(k =>
      fbPill(`${by[k]} ${k}`, k)).join("")}</div>
      <div class="fb-note">Stability classification at Day ${stab.lead} of ${stab.cycle},
        computed at each analysis-region centre. Open <strong>Stability</strong> for the
        full readout (trajectory, volatility, cycle revision).</div>`;
  } else {
    stabBlock = fbEmpty("Stability unavailable", "No region readouts for this cycle.");
  }

  /* --- weakest regions ---------------------------------------------- */
  const lead = fbpLead();
  let weakest = [];
  if (regs && regs.rows) {
    weakest = regs.rows.filter(r => r.lead_day === lead)
      .sort((a, b) => a.confidence - b.confidence).slice(0, 6);
  }
  const weakTbl = weakest.length ? `
    <table class="fb-tbl">
      <thead><tr><th>Region</th><th>Confidence</th><th>Bust Prob.</th>
        <th>Rainfall Error</th><th>Regime</th></tr></thead>
      <tbody>${weakest.map(r => `
        <tr>
          <td>${fbEsc(r.region_name)}</td>
          <td>${r.confidence} ${typeof pillFor === "function" ? pillFor(bandOf(r.confidence)) : ""}</td>
          <td style="color:${r.bust_probability > 0.5 ? "var(--red)" : r.bust_probability > 0.25 ? "var(--amber)" : "var(--t1)"}">${fmtPct(r.bust_probability)}</td>
          <td>${r.expected_error.precipitation_mm_day}</td>
          <td>${fbEsc(r.regime)}</td>
        </tr>`).join("")}</tbody>
    </table>` : fbEmpty("No region data", "Unavailable for this cycle.");

  el.innerHTML = `
    <div class="fb-kpis">
      <div class="fb-kpi"><div class="fb-kpi-k">Metrics compared</div>
        <div class="fb-kpi-v">${cmp && cmp.available ? 1 : 0}<small> / ${FBP_DRIFT_METRICS.length} change layers</small></div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Regions on watch</div>
        <div class="fb-kpi-v">${weakest.filter(r => r.confidence < 60).length}<small> below 60</small></div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Weakest region (D${lead})</div>
        <div class="fb-kpi-v">${weakest[0] ? fbEsc(weakest[0].region_name) : "—"}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Weakest confidence</div>
        <div class="fb-kpi-v">${weakest[0] ? weakest[0].confidence : "—"}<small> / 100</small></div></div>
    </div>

    <div class="fb-grid2">
      <div class="fb-card">
        <div class="fb-card-h">What changed since the previous cycle</div>
        <div class="fb-card-b">${changeBlock}</div>
        <div class="fb-card-f">
          <button class="fb-btn" onclick="fbpPick('intelTabs','drift')">Open Drift</button>
          <button class="fb-btn ghost" onclick="fbpOpenMap()">Open the drift map</button>
        </div>
      </div>
      <div class="fb-card">
        <div class="fb-card-h">Forecast stability</div>
        <div class="fb-card-b">${stabBlock}</div>
        <div class="fb-card-f">
          <button class="fb-btn" onclick="fbpPick('intelTabs','stability')">Open Stability</button>
        </div>
      </div>
    </div>

    <div class="fb-card">
      <div class="fb-card-h">Weakest regions at Day ${lead} of ${fbpCycle()}</div>
      <div class="fb-card-b">${weakTbl}</div>
      <div class="fb-card-f">
        <button class="fb-btn" onclick="fbpPick('intelTabs','risk')">Open Risk Evolution</button>
        <button class="fb-btn ghost" onclick="fbpPick('intelTabs','ensemble')">Open Ensemble</button>
      </div>
    </div>`;
}

function fbpOpenMap() {
  if (typeof setMode === "function") setMode("change");
  if (typeof show === "function") show("overview");
}

/* ═════════════════════════ TAB: DRIFT ═════════════════════════ */
async function fbIntelDrift(el) {
  const metric = FBP.driftMetric || "confidence";
  const sem = METRIC_SEM[metric];
  const sel = `<div class="fb-ctl">
    <label>Change layer</label>
    <select class="fb-select" onchange="setFBP({driftMetric:this.value});fbpRenderIntel()">
      ${FBP_DRIFT_METRICS.filter(semChangeDefined).map(k =>
        `<option value="${k}"${k === metric ? " selected" : ""}>${fbEsc(METRIC_SEM[k].label)}</option>`).join("")}
    </select>
    <label>Lead time</label>
    ${fbLeadBar()}
    <button class="fb-btn" onclick="fbpOpenMap()">Open full map</button>
  </div>`;

  el.innerHTML = sel + fbLoading("Loading drift…");
  let cmp;
  try { cmp = await fbComparison(metric); }
  catch (e) { el.innerHTML = sel + fbErr("Forecast Change unavailable", e); return; }

  if (!cmp.available || !cmp.change) { el.innerHTML = sel + fbUnavailableBox(cmp); return; }

  const regions = fbAggregate(cmp, "region");
  const states = (typeof B !== "undefined" && B.ready) ? fbAggregate(cmp, "state") : null;
  FBP.driftRegions = regions;

  const regionTbl = regions && regions.length ? `
    <table class="fb-tbl">
      <thead><tr><th>Region</th><th>Mean change</th><th>Status</th><th>Cells</th></tr></thead>
      <tbody>${regions.map(r => `<tr>
        <td>${fbEsc(r.name)}</td>
        <td style="font-weight:700">${semDelta(metric, r.mean)}</td>
        <td>${fbStatusPill(cmp, r.mean)}</td>
        <td>${r.n}</td></tr>`).join("")}</tbody>
    </table>` : fbEmpty("No region aggregation", "Unavailable for this cycle.");

  const stateBlock = states
    ? `<div class="fb-card"><div class="fb-card-h">State-level drift</div>
         <div class="fb-card-b">${fbDriftSVG(cmp, states)}</div></div>`
    : `<div class="fb-card"><div class="fb-card-h">State-level drift</div>
         <div class="fb-card-b">${fbEmpty("Boundaries unavailable",
           "State aggregation needs /api/boundaries; region aggregation above still works.")}</div></div>`;

  el.innerHTML = sel + `
    <div class="fb-grid2">
      <div class="fb-card">
        <div class="fb-card-h">${fbEsc(cmp.label)} by analysis region</div>
        <div class="fb-card-b">${regionTbl}</div>
        <div class="fb-card-f"><div class="fb-note">Direction-signed mean over grid cells inside
          each region box. ${fbEsc(sem.direction)}. Positive always means "towards better".</div></div>
      </div>
      ${stateBlock}
    </div>`;
}
function fbLeadBar() {
  const leads = (S.products && S.products.meta && S.products.meta.lead_days) || [1,2,3,4,5,6,7,8,9,10];
  return `<span class="fb-leadbar">${leads.map(l =>
    `<button class="fb-mini${l === fbpLead() ? " on" : ""}" onclick="fbSetLead(${l})">D${l}</button>`).join("")}</span>`;
}

/* ═════════════════════════ TAB: STABILITY ═════════════════════════ */
async function fbIntelStability(el) {
  el.innerHTML = fbLoading("Loading stability readouts…");
  let stab;
  try { stab = await fbStabilityRegions(); }
  catch (e) { el.innerHTML = fbErr("Stability unavailable", e); return; }
  FBP.stabRegions = stab.regions;
  if (!FBP.region || !stab.regions.some(r => r.region === FBP.region)) {
    FBP.region = stab.regions.length ? stab.regions[0].region : null;
  }

  const chips = {};
  stab.regions.forEach(r => {
    const c = (r.readout.stability || {}).classification || "unknown";
    chips[c] = (chips[c] || 0) + 1;
  });

  const tbl = `<table class="fb-tbl fb-tbl-click">
    <thead><tr><th>Region</th><th>Class</th><th>Score</th><th>Trend</th>
      <th>Volatility</th><th>Cycle revision</th></tr></thead>
    <tbody>${stab.regions.map(r => {
      const ro = r.readout, st = ro.stability || {}, t = ro.trajectory || {},
            v = ro.volatility || {}, cc = ro.cycleChange || {};
      const sel = r.region === FBP.region ? ' class="sel"' : "";
      return `<tr${sel} onclick="setFBP({region:'${fbEsc(r.region)}'});fbpRenderIntel()">
        <td>${fbEsc(r.name)}</td>
        <td>${fbPill(st.label || "—", st.classification || "")}</td>
        <td>${st.score == null ? "—" : st.score + " / 100"}</td>
        <td>${fbEsc(t.classification || "—")}</td>
        <td>${v.index == null ? "—" : v.index} <small>(${fbEsc(v.signal || "—")})</small></td>
        <td>${cc.available ? semDelta("confidence", cc.change) : "Unavailable"}</td>
      </tr>`;
    }).join("")}</tbody></table>`;

  el.innerHTML = `
    <div class="fb-ctl">
      <label>Region</label>
      ${fbRegionSelect("stabRegionSel")}
      <span class="fb-chips">${Object.keys(chips).map(k =>
        fbPill(`${chips[k]} ${k}`, k)).join("")}</span>
    </div>
    <div class="fb-card"><div class="fb-card-h">Stability by analysis region · Day ${stab.lead} of ${stab.cycle}</div>
      <div class="fb-card-b">${tbl}</div></div>
    <div class="fb-card"><div class="fb-card-h">Selected region detail — ${fbEsc(fbRegionName(FBP.region))}</div>
      <div class="fb-card-b" id="fbStabDetail">${fbLoading("Loading readout…")}</div></div>`;

  const r = stab.regions.find(x => x.region === FBP.region);
  const host = document.getElementById("fbStabDetail");
  if (!r || !host) return;
  try {
    const full = await fbpJSON(`/api/forecast-stability?cycle=${fbpCycle() || ""}` +
      `&lead=${fbpLead()}&lat=${r.centroid.lat}&lon=${r.centroid.lon}`);
    host.innerHTML = stabilityHTML(full);
  } catch (e) {
    host.innerHTML = fbErr("Stability readout unavailable", e);
  }
}

/* ═════════════════════════ TAB: ENSEMBLE ═════════════════════════ */
function fbVarCard(v) {
  const rows = [
    ["Control (deterministic)", v.control],
    ["Ensemble mean", v.ens_mean],
    ["Ensemble spread (σ)", v.ens_std],
    ["q25", v.q25], ["q50 (median)", v.q50], ["q75", v.q75], ["q90", v.q90],
    ["Interquartile range", v.iqr],
    ["Verifying observation", v.observed]
  ].filter(r => r[1] !== null && r[1] !== undefined);
  const extras = Object.entries(v.extras || {})
    .map(([k, val]) => [k.replace(/_/g, " "), val])
    .filter(r => r[1] !== null && r[1] !== undefined);
  return `<div class="fb-var">
    <div class="fb-var-h">${fbEsc(v.label)} <small>${fbEsc(v.unit)}</small></div>
    ${v.available ? `
      <div class="fb-var-rows">${rows.concat(extras).map(([k, val]) => `
        <div class="fb-var-r"><span>${fbEsc(k)}</span><b>${val}</b></div>`).join("")}</div>`
      : `<div class="fb-empty-b">Unavailable${v.reason ? ": " + fbEsc(v.reason) : ""}</div>`}
  </div>`;
}

async function fbIntelEnsemble(el) {
  if (!FBP.region) {
    try {
      const defs = await fbRegionsPayload();
      FBP.regionIndex = fbRegionIndex(defs.definitions);
      const first = Object.keys(FBP.regionIndex)[0];
      if (first) { FBP.region = first; const r = FBP.regionIndex[first];
        FBP.loc = {lat: r.lat, lon: r.lon, label: r.name}; }
    } catch (e) { /* handled below */ }
  }
  const head = `<div class="fb-ctl">
    <label>Region</label>${fbRegionSelect("ensRegionSel")}
    <label>Lead time</label>${fbLeadBar()}
  </div>`;
  el.innerHTML = head + fbLoading("Loading ensemble summary…");

  const loc = FBP.loc;
  let ens;
  try { ens = await fbEnsemble(loc.lat, loc.lon); }
  catch (e) { el.innerHTML = head + fbErr("Ensemble unavailable", e); return; }

  const v = ens.variables.precipitation;
  let chart = "";
  if (v && v.available) {
    const pts = [
      ["q25", v.q25], ["q50", v.q50], ["q75", v.q75], ["q90", v.q90]
    ].filter(p => p[1] !== null);
    if (pts.length) {
      chart = fbLineChart([{
        label: "Rainfall quantiles (mm/day)", color: "var(--accent)",
        points: pts.map(p => ({x: p[0], y: p[1]}))
      }], {xs: pts.map(p => p[0]), fmtX: x => x, fmtY: y => Math.round(y),
        aria: "Ensemble rainfall quantiles"});
    }
  }

  const memberBlock = `
    <div class="fb-card">
      <div class="fb-card-h">Ensemble member viewer</div>
      <div class="fb-card-b">
        ${ens.members_available
          ? `<div class="fb-note">Member dimension present (${ens.member_count} members).</div>`
          : fbEmpty("Members unavailable in current data mode",
              `${fbEsc(ens.members_unavailable_reason)}
               <div class="fb-empty-b">Declared member count: ${ens.member_count == null ? "unknown" : ens.member_count}.
               Statistics shown above (mean, σ, quantiles, PoP, control) are real values read from the
               forecast file - only the per-member traces cannot be drawn.</div>`)}
      </div>
    </div>`;

  el.innerHTML = head + `
    <div class="fb-kpis">
      <div class="fb-kpi"><div class="fb-kpi-k">Cycle</div><div class="fb-kpi-v">${fbEsc(ens.cycle)}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Lead</div><div class="fb-kpi-v">D${ens.lead}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Valid time</div><div class="fb-kpi-v">${fbEsc(ens.valid_time || "—")}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Location</div>
        <div class="fb-kpi-v">${ens.location.lat}, ${ens.location.lon}${ens.location.region ? " · " + fbEsc(fbRegionName(ens.location.region)) : ""}</div></div>
    </div>
    <div class="fb-grid2">
      <div class="fb-card"><div class="fb-card-h">Ensemble summary</div>
        <div class="fb-card-b fb-var-grid">
          ${Object.values(ens.variables).map(fbVarCard).join("")}
        </div>
        <div class="fb-card-f"><div class="fb-note">${fbEsc(ens.observed_available
          ? "Verifying observation is the truth file value for the valid time."
          : (ens.observed_reason || "Verification unavailable."))}</div></div>
      </div>
      <div class="fb-card"><div class="fb-card-h">Rainfall distribution</div>
        <div class="fb-card-b">${chart || fbEmpty("Quantiles unavailable", "No precipitation quantiles in this file.")}</div>
      </div>
    </div>
    ${memberBlock}`;
}

/* ═════════════════════════ TAB: ANALOGUES ═════════════════════════ */
async function fbIntelAnalogues(el) {
  if (!FBP.region) {
    try {
      const defs = await fbRegionsPayload();
      FBP.regionIndex = fbRegionIndex(defs.definitions);
      const first = Object.keys(FBP.regionIndex)[0];
      if (first) { FBP.region = first; const r = FBP.regionIndex[first];
        FBP.loc = {lat: r.lat, lon: r.lon, label: r.name}; }
    } catch (e) { /* handled below */ }
  }
  const head = `<div class="fb-ctl"><label>Region</label>${fbRegionSelect("anaRegionSel")}
    <label>Lead time</label>${fbLeadBar()}</div>`;
  el.innerHTML = head + fbLoading("Loading historical analogues…");

  let a;
  try { a = await fbAnalogs(FBP.loc.lat, FBP.loc.lon); }
  catch (e) { el.innerHTML = head + fbErr("Analogue lookup unavailable", e); return; }
  const an = a.analogue || {};
  const rows = an.most_similar || [];

  el.innerHTML = head + `
    <div class="fb-kpis">
      <div class="fb-kpi"><div class="fb-kpi-k">Similarity</div>
        <div class="fb-kpi-v">${an.similarity == null ? "—" : an.similarity}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Analogues searched</div>
        <div class="fb-kpi-v">${an.n_analogues == null ? "—" : an.n_analogues}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Historical bust rate</div>
        <div class="fb-kpi-v">${an.historical_bust_rate == null ? "—" : fmtPct(an.historical_bust_rate)}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Historical mean error</div>
        <div class="fb-kpi-v">${an.historical_mean_error_mm_day == null ? "—" : an.historical_mean_error_mm_day}<small> mm/day</small></div></div>
    </div>
    <div class="fb-card">
      <div class="fb-card-h">Most similar past situations</div>
      <div class="fb-card-b">
        ${rows.length ? `<table class="fb-tbl">
          <thead><tr><th>Cycle</th><th>Valid date</th><th>Lead</th><th>Position</th>
            <th>Regime</th><th>Similarity</th><th>Observed error</th><th>Was a bust</th></tr></thead>
          <tbody>${rows.map(r2 => `<tr>
            <td>${fbEsc(r2.cycle)}</td><td>${fbEsc(r2.valid_date)}</td><td>D${r2.lead_day}</td>
            <td>${r2.lat}, ${r2.lon}</td><td>${fbEsc(r2.regime)}</td><td>${r2.similarity}</td>
            <td>${r2.observed_error == null ? "—" : r2.observed_error}</td>
            <td>${r2.was_bust === null || r2.was_bust === undefined ? "—"
              : (r2.was_bust ? fbPill("Yes", "down") : fbPill("No", "up"))}</td>
          </tr>`).join("")}</tbody></table>`
          : fbEmpty("No analogues", "Unavailable for this selection.")}
      </div>
      <div class="fb-card-f"><div class="fb-note">Analogues are retrieved from the stored feature
        sample for this forecast cycle; they are historical look-alikes, not a re-forecast.</div></div>
    </div>`;
}

/* ═════════════════════════ TAB: DRIVERS ═════════════════════════ */
async function fbIntelDrivers(el) {
  if (!FBP.region) {
    try {
      const defs = await fbRegionsPayload();
      FBP.regionIndex = fbRegionIndex(defs.definitions);
      const first = Object.keys(FBP.regionIndex)[0];
      if (first) { FBP.region = first; const r = FBP.regionIndex[first];
        FBP.loc = {lat: r.lat, lon: r.lon, label: r.name}; }
    } catch (e) { /* handled below */ }
  }
  const head = `<div class="fb-ctl"><label>Region</label>${fbRegionSelect("drvRegionSel")}
    <label>Lead time</label>${fbLeadBar()}</div>`;
  el.innerHTML = head + fbLoading("Loading drivers…");

  let p;
  try { p = await fbGridPoint(FBP.loc.lat, FBP.loc.lon); }
  catch (e) { el.innerHTML = head + fbErr("Drivers unavailable", e); return; }

  const ud = p.uncertainty_decomposition || {};
  const ed = p.explanation_detail || {};
  const contribs = (ed.top_positive_contributors || []).concat(ed.top_negative_contributors || []);
  const vb = p.variable_bust_probability || {};
  const s = p.series || {};

  const contribBlock = contribs.length ? `
    <div class="fb-bars">${contribs.map(c => {
      const mag = Math.min(1, Math.abs(c.contribution || 0));
      const pos = (c.contribution || 0) >= 0;
      return `<div class="fb-bar-row">
        <div class="fb-bar-lab">${fbEsc(c.label)} <small>(${fbEsc(c.block)})</small></div>
        <div class="fb-bar-track"><div class="fb-bar-fill ${pos ? "up" : "down"}"
          style="width:${Math.round(mag * 100)}%"></div></div>
        <div class="fb-bar-n">${(c.contribution == null ? "—" : c.contribution)}</div>
      </div>`;
    }).join("")}</div>` : fbEmpty("Attribution unavailable", "No contributor breakdown for this cell.");

  const varBlock = Object.keys(vb).length ? `
    <table class="fb-tbl"><thead><tr><th>Variable</th><th>Predicted bust probability</th></tr></thead>
    <tbody>${Object.entries(vb).map(([k, v]) => `<tr><td>${fbEsc(k)}</td>
      <td style="color:${v > 0.5 ? "var(--red)" : v > 0.25 ? "var(--amber)" : "var(--t1)"}">${fmtPct(v)}</td></tr>`).join("")}
      <tr class="fb-tbl-em"><td>Dominant variable</td><td>${fbEsc(p.dominant_variable || "—")}</td></tr>
    </tbody></table>` : fbEmpty("Variable decomposition unavailable", "Not present for this cell.");

  const confSeries = (s.lead || []).map((l, i) => ({x: "D" + l, y: s.confidence ? s.confidence[i] : null}));
  const volSeries = (s.lead || []).map((l, i) => ({x: "D" + l, y: s.volatility_index ? s.volatility_index[i] : null}));

  const udRows = [];
  Object.entries(ud).forEach(([group, obj]) => {
    if (!obj || typeof obj !== "object") return;
    Object.entries(obj).forEach(([k, v]) => {
      if (k === "note") return;
      udRows.push([group.replace(/_/g, " "), k.replace(/_/g, " "), v]);
    });
  });

  el.innerHTML = head + `
    <div class="fb-kpis">
      <div class="fb-kpi"><div class="fb-kpi-k">Confidence</div>
        <div class="fb-kpi-v">${p.confidence}<small> / 100</small></div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Bust probability</div>
        <div class="fb-kpi-v">${fmtPct(p.bust_probability)}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Dominant variable</div>
        <div class="fb-kpi-v">${fbEsc(p.dominant_variable || "—")}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Regime</div>
        <div class="fb-kpi-v">${fbEsc(p.regime || "—")}</div></div>
    </div>

    <div class="fb-grid2">
      <div class="fb-card"><div class="fb-card-h">Why this cell looks like this</div>
        <div class="fb-card-b">
          <ul class="fb-list">${(p.explanations || []).map(x => `<li>${fbEsc(x)}</li>`).join("")}</ul>
          ${ed.summary ? `<div class="fb-note">${fbEsc(ed.summary)}</div>` : ""}
        </div></div>
      <div class="fb-card"><div class="fb-card-h">Contribution ranking</div>
        <div class="fb-card-b">${contribBlock}</div></div>
    </div>

    <div class="fb-grid2">
      <div class="fb-card"><div class="fb-card-h">Confidence across lead times</div>
        <div class="fb-card-b">${fbLineChart([{label: "Confidence", color: "var(--accent)",
          points: confSeries}], {xs: confSeries.map(p2 => p2.x), fmtY: v => Math.round(v),
          min: 0, max: 100, aria: "Confidence across lead times"})}</div></div>
      <div class="fb-card"><div class="fb-card-h">Volatility across lead times</div>
        <div class="fb-card-b">${fbLineChart([{label: "Volatility index", color: "var(--amber)",
          points: volSeries}], {xs: volSeries.map(p2 => p2.x), fmtY: v => Math.round(v * 100) / 100,
          min: 0, aria: "Volatility across lead times"})}</div></div>
    </div>

    <div class="fb-grid2">
      <div class="fb-card"><div class="fb-card-h">Variable bust probability</div>
        <div class="fb-card-b">${varBlock}</div></div>
      <div class="fb-card"><div class="fb-card-h">Uncertainty indicators</div>
        <div class="fb-card-b">
          <table class="fb-tbl"><thead><tr><th>Source</th><th>Indicator</th><th>Value</th></tr></thead>
          <tbody>${udRows.length ? udRows.map(([g, k, v]) =>
            `<tr><td>${fbEsc(g)}</td><td>${fbEsc(k)}</td><td>${v === null ? "—" : v}</td></tr>`).join("")
            : `<tr><td colspan="3">Unavailable</td></tr>`}</tbody></table>
          ${ud.note ? `<div class="fb-note">${fbEsc(ud.note)}</div>` : ""}
        </div></div>
    </div>`;
}

/* ═════════════════════════ TAB: SCENARIOS ═════════════════════════ */
async function fbIntelScenarios(el) {
  if (!FBP.region) {
    try {
      const defs = await fbRegionsPayload();
      FBP.regionIndex = fbRegionIndex(defs.definitions);
      const first = Object.keys(FBP.regionIndex)[0];
      if (first) { FBP.region = first; const r = FBP.regionIndex[first];
        FBP.loc = {lat: r.lat, lon: r.lon, label: r.name}; }
    } catch (e) { /* handled below */ }
  }
  const head = `<div class="fb-ctl"><label>Region</label>${fbRegionSelect("scnRegionSel")}
    <label>Lead time</label>${fbLeadBar()}</div>`;
  el.innerHTML = head + fbLoading("Loading scenarios…");

  let ens;
  try { ens = await fbEnsemble(FBP.loc.lat, FBP.loc.lon); }
  catch (e) { el.innerHTML = head + fbErr("Scenarios unavailable", e); return; }
  const v = ens.variables.precipitation || {};

  const scenarios = [
    ["Drier than usual (q25)", v.q25, "drier"],
    ["Most likely (q50 / median)", v.q50, "likely"],
    ["Wetter than usual (q75)", v.q75, "wetter"],
    ["Heavy-rain tail (q90)", v.q90, "tail"]
  ].filter(s => s[1] !== null && s[1] !== undefined);

  const cards = scenarios.length ? `<div class="fb-scen">${
    scenarios.map(([label, val, cls]) => `
      <div class="fb-scen-c ${cls}">
        <div class="fb-scen-k">${fbEsc(label)}</div>
        <div class="fb-scen-v">${val}<small> mm/day</small></div>
      </div>`).join("")}
    <div class="fb-scen-c ctrl">
      <div class="fb-scen-k">Control forecast</div>
      <div class="fb-scen-v">${v.control == null ? "—" : v.control}<small> mm/day</small></div>
    </div>
    <div class="fb-scen-c obs">
      <div class="fb-scen-k">Verifying observation</div>
      <div class="fb-scen-v">${v.observed == null ? "—" : v.observed}<small> mm/day</small></div>
    </div></div>`
    : fbEmpty("Scenario quantiles unavailable",
        "This forecast file carries no precipitation quantiles for the selected cell.");

  const pop = v.extras || {};
  el.innerHTML = head + `
    <div class="fb-card">
      <div class="fb-card-h">Rainfall scenarios · ${fbEsc(ens.cycle)} D${ens.lead} · valid ${fbEsc(ens.valid_time || "—")}</div>
      <div class="fb-card-b">
        ${cards}
        <div class="fb-var-rows" style="margin-top:12px">
          <div class="fb-var-r"><span>P(rain &gt; 10 mm/day)</span><b>${pop.pop_gt10 == null ? "—" : pop.pop_gt10}</b></div>
          <div class="fb-var-r"><span>P(rain &gt; 25 mm/day)</span><b>${pop.pop_gt25 == null ? "—" : pop.pop_gt25}</b></div>
          <div class="fb-var-r"><span>P(rain &gt; 50 mm/day)</span><b>${pop.pop_gt50 == null ? "—" : pop.pop_gt50}</b></div>
        </div>
        <div class="fb-note">Scenarios are the stored ensemble quantiles of the forecast file -
          not a re-run of the model and not a downscaling. The control and the verifying
          observation are shown alongside so the spread can be judged honestly.</div>
      </div>
    </div>`;
}

/* ═════════════════════════ TAB: RISK EVOLUTION ═════════════════════════ */
async function fbIntelRisk(el) {
  el.innerHTML = fbLoading("Building risk evolution…");
  const [regsR, cmpR] = await Promise.allSettled([
    fbRegionsPayload(), fbComparison("bust_probability")
  ]);
  if (regsR.status === "rejected") {
    el.innerHTML = fbErr("Risk evolution unavailable", regsR.reason); return;
  }
  const regs = regsR.value;
  FBP.regionIndex = fbRegionIndex(regs.definitions);
  const cmp = cmpR.status === "fulfilled" ? cmpR.value : null;
  const leads = (regs.meta && regs.meta.lead_days) || [...new Set(regs.rows.map(r => r.lead_day))].sort();
  const ids = [...new Set(regs.rows.map(r => r.region))];
  const nameOf = id => (regs.rows.find(r => r.region === id) || {}).region_name || id;

  const series = ids.slice(0, 8).map((id, i) => {
    const pts = leads.map(l => {
      const row = regs.rows.find(r => r.region === id && r.lead_day === l);
      return {x: "D" + l, y: row ? (row.bust_probability <= 1 ? row.bust_probability * 100 : row.bust_probability) : null};
    });
    const cols = ["var(--accent)", "var(--amber)", "var(--red)", "var(--green)",
                  "var(--gold)", "var(--cyan,#4FC3F7)", "#B39DDB", "#4DB6AC"];
    return {label: nameOf(id), color: cols[i % cols.length], points: pts};
  });

  /* persistence: how many leads each region stays at or above 25% bust risk */
  const persist = ids.map(id => {
    const vals = leads.map(l => {
      const row = regs.rows.find(r => r.region === id && r.lead_day === l);
      return row ? row.bust_probability : null;
    });
    const at = vals.filter(v => v !== null);
    const elevated = at.filter(v => v > 0.25).length;
    const worst = at.length ? Math.max(...at) : null;
    const first = at.findIndex(v => v > 0.25);
    const last = at.length - 1 - [...at].reverse().findIndex(v => v > 0.25);
    return {id, name: nameOf(id), elevated, n: at.length, worst,
      firstElevated: first >= 0 ? leads[first] : null,
      lastElevated: (first >= 0 && last >= 0) ? leads[last] : null};
  }).sort((a, b) => (b.elevated - a.elevated) || ((b.worst || 0) - (a.worst || 0)));

  const persistTbl = `<table class="fb-tbl">
    <thead><tr><th>Region</th><th>Leads ≥ 25% risk</th><th>First</th><th>Last</th>
      <th>Worst bust probability</th></tr></thead>
    <tbody>${persist.map(p => `<tr>
      <td>${fbEsc(p.name)}</td>
      <td>${p.elevated} / ${p.n}</td>
      <td>${p.firstElevated == null ? "—" : "D" + p.firstElevated}</td>
      <td>${p.lastElevated == null ? "—" : "D" + p.lastElevated}</td>
      <td style="color:${(p.worst || 0) > 0.5 ? "var(--red)" : (p.worst || 0) > 0.25 ? "var(--amber)" : "var(--t1)"}">${p.worst == null ? "—" : fmtPct(p.worst)}</td>
    </tr>`).join("")}</tbody></table>`;

  let emergence = fbEmpty("Cycle-to-cycle emergence unavailable", "No comparison for this cycle.");
  if (cmp && cmp.available && cmp.change) {
    const agg = fbAggregate(cmp, "region");
    if (agg && agg.length) {
      /* direction-signed mean of bust probability change (negative = improvement) */
      const worse = agg.filter(a => a.mean > (cmp.threshold || 0));
      const better = agg.filter(a => a.mean < -(cmp.threshold || 0));
      emergence = `
        <div class="fb-chips">
          ${fbPill(`${worse.length} regions deteriorating`, "deteriorating")}
          ${fbPill(`${better.length} regions improving`, "improving")}
          ${fbPill(`${agg.length - worse.length - better.length} unchanged`, "stable")}
        </div>
        <table class="fb-tbl" style="margin-top:10px">
          <thead><tr><th>Region</th><th>Δ bust probability</th><th>Status</th></tr></thead>
          <tbody>${agg.map(a => `<tr>
            <td>${fbEsc(a.name)}</td>
            <td style="font-weight:700">${semDelta("bust_probability", -a.mean)}</td>
            <td>${fbStatusPill({threshold: cmp.threshold}, -a.mean)}</td></tr>`).join("")}</tbody>
        </table>
        <div class="fb-note">Bust probability is a lower-is-better metric, so the table shows
          −(direction-signed mean): a positive value means risk rose since ${fbEsc(cmp.previousCycle)}.</div>`;
    }
  }

  el.innerHTML = `
    <div class="fb-card">
      <div class="fb-card-h">Bust probability by lead time, by region</div>
      <div class="fb-card-b">${fbLineChart(series, {xs: leads.map(l => "D" + l),
        fmtX: x => x, fmtY: v => Math.round(v) + "%", min: 0, max: 100,
        aria: "Bust probability across lead times"})}</div>
      <div class="fb-card-f"><div class="fb-note">Derived from /api/regions - mean predicted bust
        probability per analysis region and lead time. Top ${series.length} regions by lead coverage.</div></div>
    </div>
    <div class="fb-grid2">
      <div class="fb-card"><div class="fb-card-h">Risk persistence</div>
        <div class="fb-card-b">${persistTbl}</div>
        <div class="fb-card-f"><div class="fb-note">A region "persists" while its predicted bust
          probability stays above 25%. Threshold is a display convention, not a model output.</div></div>
      </div>
      <div class="fb-card"><div class="fb-card-h">Emergence and dissipation vs the previous cycle</div>
        <div class="fb-card-b">${emergence}</div>
      </div>
    </div>`;
}

/* ═════════════════════════ pane wiring ═════════════════════════ */
const FBP_INTEL_RENDER = {
  overview: fbIntelOverview, drift: fbIntelDrift, stability: fbIntelStability,
  ensemble: fbIntelEnsemble, analogues: fbIntelAnalogues, drivers: fbIntelDrivers,
  scenarios: fbIntelScenarios, risk: fbIntelRisk
};
let _fbIntelTok = 0;
function fbpRenderIntel() {
  const tabs = document.getElementById("intelTabs");
  const raw = document.getElementById("intelBody");
  if (!tabs || !raw) return;
  fbSubnav("intelTabs", FBP_INTEL_TABS, FBP.intelTab);
  const fn = FBP_INTEL_RENDER[FBP.intelTab] || fbIntelOverview;
  const tok = ++FB_TOK.n; _fbIntelTok = tok;
  // The guard must capture THIS render's token: comparing against the live
  // counter would let a stale render write through, which is exactly the
  // race we are trying to close.
  const body = fbGuardedEl(raw, tok);
  fn(body).catch(e => {
    if (tok === FB_TOK.n) raw.innerHTML = fbErr("Forecast Intelligence unavailable", e);
  });
}
