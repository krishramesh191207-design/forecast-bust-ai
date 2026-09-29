/* ═══════════════════════════════════════════════════════════════════════
   OPERATIONS CENTER pane
   Tabs: Situation · Active Alerts · Systems · Regions · Rules · History ·
         Performance · Trust · Brief
   ═══════════════════════════════════════════════════════════════════════ */

const FBP_OPS_TABS = [
  ["situation", "Situation"],
  ["alerts", "Active Alerts"],
  ["systems", "Systems"],
  ["regions", "Regions"],
  ["rules", "Rules"],
  ["history", "History"],
  ["performance", "Performance"],
  ["trust", "Trust"],
  ["brief", "Brief"]
];

const FBP_SEVERITY_CLS = {critical: "critical", warning: "down",
  advisory: "moderate", info: "neutral"};
const FBP_STATUS_CLS = {new: "down", escalated: "critical",
  ongoing: "moderate", resolved: "up"};

function fbSev(sev) { return fbPill(sev, FBP_SEVERITY_CLS[sev] || "neutral"); }
function fbStatus(st) { return fbPill(st, FBP_STATUS_CLS[st] || "neutral"); }

/* ── user alert rules (localStorage - no auth backend in this sandbox) ── */
function fbLoadUserRules() {
  try { FBP.userRules = JSON.parse(localStorage.getItem("fbews.alertRules") || "[]"); }
  catch (e) { FBP.userRules = []; }
  if (!Array.isArray(FBP.userRules)) FBP.userRules = [];
}
function fbSaveUserRules() {
  try { localStorage.setItem("fbews.alertRules", JSON.stringify(FBP.userRules)); }
  catch (e) { /* private mode: rules stay in memory for this session */ }
}

/* ═════════════════════════ TAB: SITUATION ═════════════════════════ */
async function fbOpsSituation(el) {
  el.innerHTML = fbLoading("Loading situation report…");
  const [regsR, cmpR, sysR] = await Promise.allSettled([
    fbRegionsPayload(), fbComparison("confidence"),
    fbpJSON(`/api/systems?cycle=${fbpCycle() || ""}&lead=${fbpLead()}`)
  ]);
  if (regsR.status === "rejected") {
    el.innerHTML = fbErr("Situation report unavailable", regsR.reason); return;
  }
  const regs = regsR.value;
  FBP.regionIndex = fbRegionIndex(regs.definitions);
  const cmp = cmpR.status === "fulfilled" ? cmpR.value : null;
  const sys = sysR.status === "fulfilled" ? sysR.value : null;
  const lead = fbpLead();
  const rows = regs.rows.filter(r => r.lead_day === lead)
    .sort((a, b) => a.confidence - b.confidence);
  const worst = rows[0];
  const atRisk = rows.filter(r => r.confidence < 60);

  const sysBlock = sys && sys.available ? `
    <div class="fb-chips">
      ${fbPill(`${sys.systems.length} system${sys.systems.length === 1 ? "" : "s"} detected`, "moderate")}
      ${sys.systems.filter(s => s.severity === "high").length
        ? fbPill(`${sys.systems.filter(s => s.severity === "high").length} high impact`, "critical") : ""}
      ${fbPill(`${sys.regimeSummary.length} regimes`, "neutral")}
    </div>
    <div class="fb-note">Detected from the forecast pressure field for Day ${sys.lead}
      (${fbEsc(sys.validTime)}).</div>`
    : fbEmpty("Systems unavailable", fbEsc((sys && sys.message) || "Unavailable in current data mode."));

  el.innerHTML = `
    <div class="fb-kpis">
      <div class="fb-kpi"><div class="fb-kpi-k">Cycle</div><div class="fb-kpi-v">${fbpCycle() || "—"}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Lead</div><div class="fb-kpi-v">D${lead}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Regions below 60</div>
        <div class="fb-kpi-v">${atRisk.length}<small> / ${rows.length}</small></div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Lowest confidence</div>
        <div class="fb-kpi-v">${worst ? worst.confidence : "—"}<small>${worst ? " · " + fbEsc(worst.region_name) : ""}</small></div></div>
    </div>
    <div class="fb-grid2">
      <div class="fb-card">
        <div class="fb-card-h">Regional situation · Day ${lead} of ${fbpCycle()}</div>
        <div class="fb-card-b">
          <table class="fb-tbl">
            <thead><tr><th>Region</th><th>Confidence</th><th>Bust prob.</th><th>Rain error</th>
              <th>Regime</th></tr></thead>
            <tbody>${rows.map(r => `<tr>
              <td>${fbEsc(r.region_name)}</td>
              <td>${r.confidence} ${typeof pillFor === "function" ? pillFor(bandOf(r.confidence)) : ""}</td>
              <td style="color:${r.bust_probability > 0.5 ? "var(--red)" : r.bust_probability > 0.25 ? "var(--amber)" : "var(--t1)"}">${fmtPct(r.bust_probability)}</td>
              <td>${r.expected_error.precipitation_mm_day}</td>
              <td>${fbEsc(r.regime)}</td></tr>`).join("")}</tbody>
          </table>
        </div>
        <div class="fb-card-f"><div class="fb-note">${cmp && cmp.available
          ? `Forecast Change available for Day ${cmp.lead} (${fbEsc(cmp.label)}).`
          : "Forecast Change unavailable for this lead time."}</div></div>
      </div>
      <div class="fb-card">
        <div class="fb-card-h">Synoptic situation</div>
        <div class="fb-card-b">${sysBlock}</div>
        <div class="fb-card-f">
          <button class="fb-btn" onclick="fbpPick('opsTabs','systems')">Open Systems</button>
          <button class="fb-btn ghost" onclick="fbpPick('opsTabs','alerts')">Open Active Alerts</button>
        </div>
      </div>
    </div>`;
}

/* ═════════════════════════ TAB: ACTIVE ALERTS ═════════════════════════ */
async function fbOpsAlerts(el) {
  const f = FBP.alertFilter || (FBP.alertFilter = {view: "groups", severity: "all",
    status: "all", lead: "all"});
  const head = `<div class="fb-ctl">
    <label>View</label>
    <select class="fb-select" onchange="FBP.alertFilter.view=this.value;fbpRenderOps()">
      <option value="groups"${f.view === "groups" ? " selected" : ""}>Grouped</option>
      <option value="records"${f.view === "records" ? " selected" : ""}>Individual alerts</option>
    </select>
    <label>Severity</label>
    <select class="fb-select" onchange="FBP.alertFilter.severity=this.value;fbpRenderOps()">
      ${["all", "critical", "warning", "advisory", "info"].map(s =>
        `<option value="${s}"${f.severity === s ? " selected" : ""}>${s === "all" ? "All" : s}</option>`).join("")}
    </select>
    <label>Status</label>
    <select class="fb-select" onchange="FBP.alertFilter.status=this.value;fbpRenderOps()">
      ${["all", "new", "escalated", "ongoing", "resolved"].map(s =>
        `<option value="${s}"${f.status === s ? " selected" : ""}>${s === "all" ? "All" : s}</option>`).join("")}
    </select>
    <label>Lead</label>
    <select class="fb-select" onchange="FBP.alertFilter.lead=this.value;fbpRenderOps()">
      <option value="all"${f.lead === "all" ? " selected" : ""}>All leads</option>
      ${(S.products && S.products.meta ? S.products.meta.lead_days : [1,2,3,4,5,6,7,8,9,10])
        .map(l => `<option value="${l}"${String(f.lead) === String(l) ? " selected" : ""}>Day ${l}</option>`).join("")}
    </select>
    <button class="fb-btn ghost" onclick="fbpPick('opsTabs','rules')">Edit rules</button>
  </div>`;
  el.innerHTML = head + fbLoading("Evaluating alert rules…");

  let d;
  try {
    const cyc = fbpCycle();
    d = await fbpJSON(`/api/alerts?cycle=${cyc || ""}&t=${Date.now()}`);
  } catch (e) { el.innerHTML = head + fbErr("Active alerts unavailable", e); return; }

  const match = a =>
    (f.severity === "all" || a.severity === f.severity) &&
    (f.status === "all" || a.status === f.status) &&
    (f.lead === "all" || String(a.lead) === String(f.lead));
  const alerts = d.alerts.filter(match);
  const groups = d.groups.filter(g =>
    (f.severity === "all" || g.severity === f.severity) &&
    (f.lead === "all" || String(g.lead) === String(f.lead)) &&
    (f.status === "all" || g.statuses.includes(f.status)));
  const resolved = d.resolved.filter(match);

  const kpis = `
    <div class="fb-kpis">
      <div class="fb-kpi"><div class="fb-kpi-k">New</div><div class="fb-kpi-v">${d.counts.new}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Escalated</div><div class="fb-kpi-v">${d.counts.escalated}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Ongoing</div><div class="fb-kpi-v">${d.counts.ongoing}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Resolved since ${fbEsc(d.previousCycle || "—")}</div>
        <div class="fb-kpi-v">${d.counts.resolved}</div></div>
    </div>`;

  const sevChips = `<div class="fb-chips">
      ${fbSev("critical")} ${d.bySeverity.critical} critical ·
      ${fbSev("warning")} ${d.bySeverity.warning} ·
      ${fbSev("advisory")} ${d.bySeverity.advisory} ·
      ${fbSev("info")} ${d.bySeverity.info}
    </div>`;

  let body;
  if (f.view === "groups") {
    body = groups.length ? `
      <table class="fb-tbl">
        <thead><tr><th>Severity</th><th>Rule</th><th>Lead</th><th>Regions</th>
          <th>Worst value</th><th>Status</th></tr></thead>
        <tbody>${groups.map(g => `<tr>
          <td>${fbSev(g.severity)}</td>
          <td>${fbEsc(g.label)}<br><small style="color:var(--t3)">${fbEsc(g.condition)}</small></td>
          <td>D${g.lead}</td>
          <td>${g.count} <small style="color:var(--t3)">(${fbEsc(g.regionNames.slice(0, 3).join(", "))}${g.regionNames.length > 3 ? ", +" + (g.regionNames.length - 3) : ""})</small></td>
          <td style="font-weight:700">${g.worstValue == null ? "—" : g.worstValue}</td>
          <td>${g.statuses.map(fbStatus).join(" ")}</td>
        </tr>`).join("")}</tbody>
      </table>` : fbEmpty("No alert groups match this filter",
        "Adjust the severity, status or lead filter.");
  } else {
    body = alerts.length ? `
      <table class="fb-tbl">
        <thead><tr><th>Severity</th><th>Status</th><th>Rule</th><th>Region</th>
          <th>Lead</th><th>Value</th><th>Condition</th></tr></thead>
        <tbody>${alerts.map(a => `<tr>
          <td>${fbSev(a.severity)}</td>
          <td>${fbStatus(a.status)}</td>
          <td>${fbEsc(a.ruleLabel)}</td>
          <td>${fbEsc(a.regionName)}</td>
          <td>D${a.lead}</td>
          <td style="font-weight:700">${a.value}</td>
          <td>${fbEsc(a.condition)}</td>
        </tr>`).join("")}</tbody>
      </table>` : fbEmpty("No alerts match this filter",
        "Adjust the severity, status or lead filter.");
  }

  const resolvedBlock = resolved.length ? `
    <div class="fb-card">
      <div class="fb-card-h">Resolved since ${fbEsc(d.previousCycle || "—")}</div>
      <div class="fb-card-b"><table class="fb-tbl">
        <thead><tr><th>Severity</th><th>Rule</th><th>Region</th><th>Lead</th><th>Value</th></tr></thead>
        <tbody>${resolved.slice(0, 20).map(a => `<tr>
          <td>${fbSev(a.severity)}</td><td>${fbEsc(a.ruleLabel)}</td>
          <td>${fbEsc(a.regionName)}</td><td>D${a.lead}</td><td>${a.value}</td>
        </tr>`).join("")}</tbody></table></div>
    </div>` : "";

  const suppressedNote = d.suppressed.length
    ? `<div class="fb-note">${d.suppressed.length} lower-severity breach${d.suppressed.length === 1 ? "" : "es"}
        suppressed as already covered by a higher-severity rule on the same metric, region and lead.
        Open <strong>Rules</strong> to see the rule list.</div>` : "";

  el.innerHTML = head + kpis + sevChips + `
    <div class="fb-card">
      <div class="fb-card-h">${f.view === "groups" ? "Alert groups" : "Individual alerts"}
        · ${f.view === "groups" ? groups.length : alerts.length} shown
        · cycle ${fbEsc(d.cycle)} vs ${fbEsc(d.previousCycle || "—")}</div>
      <div class="fb-card-b">${body}${suppressedNote}</div>
      <div class="fb-card-f"><div class="fb-note">${fbEsc(d.note)}</div></div>
    </div>
    ${resolvedBlock}`;
}

/* ═════════════════════════ TAB: SYSTEMS ═════════════════════════ */
async function fbOpsSystems(el) {
  const lead = fbpLead();
  const head = `<div class="fb-ctl"><label>Lead time</label>${fbLeadBar()}
    <button class="fb-btn ghost" onclick="fbpPick('opsTabs','alerts')">Open Active Alerts</button></div>`;
  el.innerHTML = head + fbLoading("Detecting synoptic systems…");

  let sys;
  try { sys = await fbpJSON(`/api/systems?cycle=${fbpCycle() || ""}&lead=${lead}`); }
  catch (e) { el.innerHTML = head + fbErr("Systems board unavailable", e); return; }
  if (!sys.available) { el.innerHTML = head + fbEmpty(fbEsc(sys.reason || "Systems unavailable"), fbEsc(sys.message || "")); return; }

  const cards = sys.systems.length ? sys.systems.map(s => `
    <div class="fb-card">
      <div class="fb-card-h">${fbEsc(s.label)} <span style="float:right">${fbSev(
        s.severity === "high" ? "critical" : s.severity === "moderate" ? "warning" :
        s.severity === "low" ? "advisory" : "info")} ${fbEsc(s.labelImpact)}</span></div>
      <div class="fb-card-b">
        <div class="fb-var-rows">
          <div class="fb-var-r"><span>Position</span><b>${s.lat}, ${s.lon}</b></div>
          <div class="fb-var-r"><span>Depth below background</span><b>${s.depthHpa == null ? "—" : s.depthHpa + " hPa"}</b></div>
          <div class="fb-var-r"><span>Influence radius</span><b>${s.influenceRadiusDeg}°</b></div>
          <div class="fb-var-r"><span>Nearest region</span><b>${fbEsc(fbRegionName(s.nearestRegion))}</b></div>
          <div class="fb-var-r"><span>Mean confidence of affected cells</span><b>${s.meanConfidence == null ? "—" : s.meanConfidence}</b></div>
        </div>
        <table class="fb-tbl" style="margin-top:10px">
          <thead><tr><th>Region affected</th><th>Cells</th><th>Mean conf.</th>
            <th>Max bust</th><th>Impact</th></tr></thead>
          <tbody>${s.regionDetail.map(r2 => `<tr>
            <td>${fbEsc(fbRegionName(r2.region))}</td>
            <td>${r2.cells}</td>
            <td>${r2.meanConfidence == null ? "—" : r2.meanConfidence}</td>
            <td style="color:${r2.maxBustProbability > 0.5 ? "var(--red)" : r2.maxBustProbability > 0.25 ? "var(--amber)" : "var(--t1)"}">${fmtPct(r2.maxBustProbability)}</td>
            <td>${fbPill(r2.label, r2.severity === "high" ? "critical" : r2.severity === "moderate" ? "down" : "neutral")}</td>
          </tr>`).join("") || `<tr><td colspan="5">No analysis region within the influence radius.</td></tr>`}</tbody>
        </table>
      </div>
      <div class="fb-card-f"><div class="fb-note">${fbEsc(s.detectedFrom)}. Regional association and
        impact are read from the grid cells inside the influence radius - no manual assignment.</div></div>
    </div>`).join("")
    : fbEmpty("No closed lows detected",
        `No closed low was found in the forecast pressure field for Day ${lead}.`);

  const regime = sys.regimeSummaryAvailable && sys.regimeSummary.length ? `
    <div class="fb-card">
      <div class="fb-card-h">Distinct atmospheric regimes (share of grid cells)</div>
      <div class="fb-card-b">
        <div class="fb-bars">${sys.regimeSummary.map(r2 => `
          <div class="fb-bar-row">
            <div class="fb-bar-lab">${fbEsc(r2.regime)}</div>
            <div class="fb-bar-track"><div class="fb-bar-fill" style="width:${Math.max(1, r2.share)}%"></div></div>
            <div class="fb-bar-n">${r2.share}%</div>
          </div>`).join("")}</div>
      </div>
    </div>` : "";

  el.innerHTML = head + `
    <div class="fb-kpis">
      <div class="fb-kpi"><div class="fb-kpi-k">Systems detected</div>
        <div class="fb-kpi-v">${sys.systems.length}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">High impact</div>
        <div class="fb-kpi-v">${sys.systems.filter(s => s.severity === "high").length}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Valid time</div>
        <div class="fb-kpi-v">${fbEsc(sys.validTime)}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Regions covered</div>
        <div class="fb-kpi-v">${new Set(sys.systems.flatMap(s => s.regionsAffected)).size}</div></div>
    </div>
    ${sys.systems.length ? `<div class="fb-grid2">${cards}</div>` : cards}
    <div class="fb-grid2">
      ${regime}
      <div class="fb-card">
        <div class="fb-card-h">Not identified by the current detector</div>
        <div class="fb-card-b">
          <div class="fb-chips">${sys.notDetected.map(n => fbPill(n, "neutral")).join(" ")}</div>
          <div class="fb-note">${fbEsc(sys.notDetectedReason)}</div>
        </div>
      </div>
    </div>`;
}

/* ═════════════════════════ TAB: REGIONS ═════════════════════════ */
async function fbOpsRegions(el) {
  const f = FBP.regionBoard || (FBP.regionBoard = {metric: "confidence"});
  const head = `<div class="fb-ctl">
    <label>Sort by</label>
    <select class="fb-select" onchange="FBP.regionBoard.metric=this.value;fbpRenderOps()">
      <option value="confidence"${f.metric === "confidence" ? " selected" : ""}>Confidence (worst first)</option>
      <option value="bust_probability"${f.metric === "bust_probability" ? " selected" : ""}>Bust probability (worst first)</option>
      <option value="volatility_index"${f.metric === "volatility_index" ? " selected" : ""}>Volatility (highest first)</option>
      <option value="rain_error"${f.metric === "rain_error" ? " selected" : ""}>Rainfall error (highest first)</option>
    </select>
    <label>Lead</label>${fbLeadBar()}
    <button class="fb-btn ghost" onclick="fbpPick('opsTabs','trust')">Open Region profile</button>
  </div>`;
  el.innerHTML = head + fbLoading("Loading region board…");

  const [regsR, impR] = await Promise.allSettled([fbRegionsPayload(),
    fbpJSON("/api/event-impact")]);
  if (regsR.status === "rejected") {
    el.innerHTML = head + fbErr("Region board unavailable", regsR.reason); return;
  }
  const regs = regsR.value;
  const imp = impR.status === "fulfilled" ? impR.value : null;
  const evByRegion = {};
  if (imp && imp.available) {
    imp.rows.forEach(r => Object.keys(r.regionCounts || {}).forEach(id => {
      evByRegion[id] = (evByRegion[id] || 0) + r.regionCounts[id];
    }));
  }
  FBP.regionIndex = fbRegionIndex(regs.definitions);
  const lead = fbpLead();
  const val = r => f.metric === "rain_error" ? r.expected_error.precipitation_mm_day
    : (f.metric === "volatility_index" ? r.volatility_index : r[f.metric]);
  const lowerBetter = f.metric === "confidence" ? false : true;
  const rows = regs.rows.filter(r => r.lead_day === lead).slice().sort((a, b) => {
    const va = val(a), vb = val(b);
    return lowerBetter ? vb - va : va - vb;
  });

  const leads = (regs.meta && regs.meta.lead_days) || [];
  const matrix = `<div class="fb-card">
    <div class="fb-card-h">Confidence by lead time</div>
    <div class="fb-card-b"><div class="scroll"><table class="fb-tbl">
      <thead><tr><th>Region</th>${leads.map(l => `<th>D${l}</th>`).join("")}</tr></thead>
      <tbody>${[...new Set(regs.rows.map(r => r.region))].map(rid => {
        const name = (regs.rows.find(r => r.region === rid) || {}).region_name || rid;
        return `<tr><td>${fbEsc(name)}</td>${leads.map(l => {
          const c = regs.rows.find(r => r.region === rid && r.lead_day === l);
          const v = c ? c.confidence : null;
          return `<td style="background:${v !== null && v !== undefined ? colorCell("confidence", v, [0, 100]) : "transparent"};
            text-align:center;font-size:11px;font-weight:700">${v === null || v === undefined ? "—" : v}</td>`;
        }).join("")}</tr>`;
      }).join("")}</tbody>
    </table></div></div>
    <div class="fb-card-f"><div class="fb-note">Reliability Score per analysis region and lead
      time, straight from /api/regions. Colour uses the same scale as the map.</div></div>
  </div>`;

  el.innerHTML = head + `
    <div class="fb-card">
      <div class="fb-card-h">Region board · Day ${lead} of ${fbpCycle()} · sorted by
        ${fbEsc(f.metric.replace(/_/g, " "))}</div>
      <div class="fb-card-b">
        <table class="fb-tbl">
          <thead><tr><th>#</th><th>Region</th><th>Confidence</th><th>Bust prob.</th>
            <th>Max bust</th><th>Volatility</th><th>Rain error</th><th>Dominant</th>
            <th>Regime</th><th>Catalogue events</th></tr></thead>
          <tbody>${rows.map((r, i) => `<tr>
            <td>${i + 1}</td>
            <td>${fbEsc(r.region_name)}</td>
            <td>${r.confidence} ${typeof pillFor === "function" ? pillFor(bandOf(r.confidence)) : ""}</td>
            <td style="color:${r.bust_probability > 0.5 ? "var(--red)" : r.bust_probability > 0.25 ? "var(--amber)" : "var(--t1)"}">${fmtPct(r.bust_probability)}</td>
            <td>${fmtPct(r.max_bust_probability)}</td>
            <td>${r.volatility_index}</td>
            <td>${r.expected_error.precipitation_mm_day}</td>
            <td>${fbEsc(r.dominant_variable)}</td>
            <td>${fbEsc(r.regime)}</td>
            <td>${evByRegion[r.region] !== undefined ? evByRegion[r.region]
              : `<span class="fb-note">0</span>`}</td></tr>`).join("")}</tbody>
        </table>
      </div>
      <div class="fb-card-f"><div class="fb-note">${imp && imp.available
        ? `Catalogue events joined onto the configured bounding boxes - ` +
          `${imp.rows.reduce((a, r) => a + r.events, 0)} events across ` +
          `${imp.rows.length} types (${fbEsc(imp.source)}).`
        : "Event catalogue unavailable in this data mode."}</div></div>
    </div>
    ${matrix}`;
}

/* ═════════════════════════ TAB: RULES ═════════════════════════ */
async function fbOpsRules(el) {
  fbLoadUserRules();
  const head = `<div class="fb-ctl">
    <label>Add rule</label>
    <select class="fb-select" id="ruleMetric">
      <option value="confidence">Confidence</option>
      <option value="bust_probability">Bust probability</option>
      <option value="volatility">Volatility index</option>
      <option value="rain_error">Rainfall error</option>
    </select>
    <select class="fb-select" id="ruleOp">
      <option value="lt">&lt;</option><option value="lte">≤</option>
      <option value="gt">&gt;</option><option value="gte">≥</option>
    </select>
    <input class="fb-select" id="ruleThr" type="number" step="any" value="50" style="width:90px">
    <select class="fb-select" id="ruleSev">
      <option value="advisory">advisory</option>
      <option value="warning" selected>warning</option>
      <option value="critical">critical</option>
      <option value="info">info</option>
    </select>
    <select class="fb-select" id="ruleLead">
      <option value="">Any lead</option>
      ${(S.products && S.products.meta ? S.products.meta.lead_days : [1,2,3,4,5,6,7,8,9,10])
        .map(l => `<option value="${l}">Day ${l}</option>`).join("")}
    </select>
    <button class="fb-btn" onclick="fbAddRule()">Add rule</button>
    <button class="fb-btn ghost" onclick="fbResetRules()">Reset to built-in only</button>
  </div>`;
  el.innerHTML = head + fbLoading("Loading rules…");

  let d;
  try {
    d = await fbpJSON(`/api/alerts?cycle=${fbpCycle() || ""}&t=${Date.now()}`);
  } catch (e) { el.innerHTML = head + fbErr("Rules unavailable", e); return; }

  const builtin = d.rules;
  const user = FBP.userRules;

  const ruleRow = (r, isUser) => `<tr>
    <td>${fbPill(r.severity, FBP_SEVERITY_CLS[r.severity] || "neutral")}</td>
    <td>${fbEsc(r.label)}</td>
    <td><code>${fbEsc(r.metric)} ${({lt: "<", lte: "≤", gt: ">", gte: "≥"})[r.operator]} ${r.threshold}</code>
      ${r.lead == null ? "" : `<small style="color:var(--t3)"> · Day ${r.lead}</small>`}</td>
    <td style="font-size:11px;color:var(--t3)">${fbEsc(r.source || "user supplied")}</td>
    <td>${isUser ? `<button class="fb-btn ghost" onclick="fbRemoveRule('${fbEsc(r.id)}')">Remove</button>`
                 : `<span style="color:var(--t3)">built-in</span>`}</td>
  </tr>`;

  const evalBlock = FBP.ruleEval ? `
    <div class="fb-card">
      <div class="fb-card-h">Evaluation with ${FBP.ruleEval.rules.length} rule(s)</div>
      <div class="fb-card-b">
        <div class="fb-chips">
          ${fbPill(`${FBP.ruleEval.counts.total} alerts`, "moderate")}
          ${fbPill(`${FBP.ruleEval.counts.new} new`, "down")}
          ${fbPill(`${FBP.ruleEval.counts.escalated} escalated`, "critical")}
          ${fbPill(`${FBP.ruleEval.counts.ongoing} ongoing`, "moderate")}
          ${fbPill(`${FBP.ruleEval.suppressed.length} suppressed`, "neutral")}
        </div>
        <table class="fb-tbl" style="margin-top:10px">
          <thead><tr><th>Severity</th><th>Status</th><th>Rule</th><th>Region</th>
            <th>Lead</th><th>Value</th></tr></thead>
          <tbody>${FBP.ruleEval.alerts.slice(0, 40).map(a => `<tr>
            <td>${fbSev(a.severity)}</td><td>${fbStatus(a.status)}</td>
            <td>${fbEsc(a.ruleLabel)}</td><td>${fbEsc(a.regionName)}</td>
            <td>D${a.lead}</td><td>${a.value}</td></tr>`).join("")}</tbody>
        </table>
        ${FBP.ruleEval.alerts.length > 40
          ? `<div class="fb-note">Showing the first 40 of ${FBP.ruleEval.counts.total}.</div>` : ""}
      </div>
    </div>` : "";

  el.innerHTML = head + `
    <div class="fb-grid2">
      <div class="fb-card">
        <div class="fb-card-h">Built-in rules (${builtin.length})</div>
        <div class="fb-card-b">
          <table class="fb-tbl">
            <thead><tr><th>Severity</th><th>Label</th><th>Condition</th><th>Threshold source</th><th></th></tr></thead>
            <tbody>${builtin.map(r => ruleRow(r, false)).join("")}</tbody>
          </table>
        </div>
        <div class="fb-card-f"><div class="fb-note">Every built-in threshold already exists
          elsewhere in this project; the source column says where. Nothing is invented here.</div></div>
      </div>
      <div class="fb-card">
        <div class="fb-card-h">Your rules (${user.length}) · stored in this browser</div>
        <div class="fb-card-b">
          ${user.length ? `<table class="fb-tbl">
            <thead><tr><th>Severity</th><th>Label</th><th>Condition</th><th>Source</th><th></th></tr></thead>
            <tbody>${user.map(r => ruleRow(r, true)).join("")}</tbody>
          </table>` : fbEmpty("No custom rules",
            "Add a rule with the controls above. Custom rules are evaluated the same way as the built-in ones and are kept in this browser only - there is no account system in this sandbox.")}
        </div>
        <div class="fb-card-f">
          <button class="fb-btn" onclick="fbEvaluateRules()">Evaluate now</button>
        </div>
      </div>
    </div>
    ${evalBlock}`;
}

function fbRuleFromForm() {
  const metric = (document.getElementById("ruleMetric") || {}).value || "confidence";
  const op = (document.getElementById("ruleOp") || {}).value || "lt";
  const thr = parseFloat((document.getElementById("ruleThr") || {}).value);
  const sev = (document.getElementById("ruleSev") || {}).value || "warning";
  const leadRaw = (document.getElementById("ruleLead") || {}).value || "";
  if (!isFinite(thr)) return null;
  const labels = {confidence: "Confidence", bust_probability: "Bust probability",
    volatility: "Volatility index", rain_error: "Rainfall error"};
  const ops = {lt: "<", lte: "≤", gt: ">", gte: "≥"};
  return {
    id: "user-" + Date.now().toString(36),
    label: `${labels[metric]} ${ops[op]} ${thr}`,
    metric, operator: op, threshold: thr,
    lead: leadRaw === "" ? null : parseInt(leadRaw, 10),
    severity: sev, enabled: true, source: "user supplied (this browser)"
  };
}
function fbAddRule() {
  const r = fbRuleFromForm();
  if (!r) return;
  fbLoadUserRules();
  FBP.userRules.push(r);
  fbSaveUserRules();
  FBP.ruleEval = null;
  fbpRenderOps();
}
function fbRemoveRule(id) {
  fbLoadUserRules();
  FBP.userRules = FBP.userRules.filter(r => r.id !== id);
  fbSaveUserRules();
  FBP.ruleEval = null;
  fbpRenderOps();
}
function fbResetRules() {
  FBP.userRules = [];
  fbSaveUserRules();
  FBP.ruleEval = null;
  fbpRenderOps();
}
async function fbEvaluateRules() {
  const body = document.getElementById("opsBody");
  if (body) body.insertAdjacentHTML("afterbegin", fbLoading("Evaluating…"));
  try {
    fbLoadUserRules();
    const res = await fetch("/api/alerts/evaluate", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({cycle: fbpCycle(), rules: FBP.userRules})
    });
    if (!res.ok) throw new Error(res.status + " " + res.statusText);
    FBP.ruleEval = await res.json();
  } catch (e) {
    FBP.ruleEval = null;
    FBP.ruleEvalError = e.message;
  }
  fbpRenderOps();
}

/* ═════════════════════════ pane wiring ═════════════════════════ */
const FBP_OPS_RENDER = {
  situation: fbOpsSituation, alerts: fbOpsAlerts, systems: fbOpsSystems,
  regions: fbOpsRegions, rules: fbOpsRules, history: fbOpsHistory,
  performance: fbOpsPerformance, trust: fbOpsTrust, brief: fbOpsBrief
};
let _fbOpsTok = 0;
function fbpRenderOps() {
  const tabs = document.getElementById("opsTabs");
  const raw = document.getElementById("opsBody");
  if (!tabs || !raw) return;
  fbSubnav("opsTabs", FBP_OPS_TABS, FBP.opsTab);
  const fn = FBP_OPS_RENDER[FBP.opsTab];
  if (!fn) {
    raw.innerHTML = fbEmpty("Not available yet",
      "This Operations Center section has not been built in this build.");
    return;
  }
  const tok = ++FB_TOK.n; _fbOpsTok = tok;
  // Capture this render's token - see fbews-intel.js.
  const body = fbGuardedEl(raw, tok);
  const p = fn(body);
  if (p && p.catch) p.catch(e => {
    if (tok === FB_TOK.n) raw.innerHTML = fbErr("Operations Center unavailable", e);
  });
}

/* ═══════════════════════════ TAB: HISTORY ═══════════════════════════ */
function fbOpenCase(date, lat, lon) {
  const type = document.getElementById("csEventType");
  const sel = document.getElementById("csEventSel");
  if (!sel) return;
  if (type) type.value = "";
  if (typeof populateCsEventSel === "function") populateCsEventSel();
  const want = `${date}|${lat}|${lon}`;
  let opt = Array.from(sel.options).find(o => o.value === want);
  if (!opt) {                       // event outside the top-80 list: add it
    sel.insertAdjacentHTML("afterbegin",
      `<option value="${want}">${fbEsc(date)} (${lat}, ${lon})</option>`);
    opt = sel.options[0];
  }
  sel.value = want;
  if (typeof show === "function") show("case");
  if (typeof runCaseStudy === "function") runCaseStudy();
}

async function fbOpsHistory(el) {
  const f = FBP.historyFilter || (FBP.historyFilter = {kind: "all"});
  el.innerHTML = fbLoading("Loading event history…");
  const [impR, wpR, evR] = await Promise.allSettled([
    fbpJSON("/api/event-impact"),
    fbpJSON("/api/warning-performance?sample=12&limit=60"),
    fbpJSON("/api/historical-events?limit=200")
  ]);
  if (impR.status === "rejected") {
    el.innerHTML = fbErr("Event history unavailable", impR.reason); return;
  }
  const imp = impR.value;
  const wp = wpR.status === "fulfilled" ? wpR.value : null;
  const ev = evR.status === "fulfilled" ? evR.value : {events: []};
  if (!imp.available) {
    el.innerHTML = fbEmpty("Event history unavailable",
      fbEsc(imp.message || "No event catalogue in this data mode.")); return;
  }
  const kinds = ["all"].concat(imp.rows.map(r => r.eventType));
  const rows = imp.rows;
  const total = rows.reduce((a, r) => a + r.events, 0);
  const regionsCovered = new Set(rows.flatMap(r => r.regions)).size;

  const head = `<div class="fb-ctl">
    <label>Event type</label>
    <select class="fb-select" onchange="FBP.historyFilter.kind=this.value;fbpRenderOps()">
      ${kinds.map(k => `<option value="${fbEsc(k)}"${f.kind === k ? " selected" : ""}>${
        k === "all" ? "All event types" : fbEsc(k.replace(/_/g, " "))}</option>`).join("")}
    </select>
    <span class="fb-note">Source: ${fbEsc(imp.source)}</span></div>`;

  const typeCards = rows
    .filter(r => f.kind === "all" || r.eventType === f.kind)
    .map(r => `<div class="fb-card">
      <div class="fb-card-h">${fbEsc(r.eventType.replace(/_/g, " "))}</div>
      <div class="fb-card-b">
        <div class="fb-kpis" style="grid-template-columns:repeat(3,1fr)">
          <div class="fb-kpi"><div class="fb-kpi-k">Events</div>
            <div class="fb-kpi-v">${r.events}</div></div>
          <div class="fb-kpi"><div class="fb-kpi-k">Regions hit</div>
            <div class="fb-kpi-v">${r.regions.length}</div></div>
          <div class="fb-kpi"><div class="fb-kpi-k">Mean depth</div>
            <div class="fb-kpi-v">${r.meanDepthHpa ?? "—"}<small> hPa</small></div></div>
        </div>
        <div class="fb-chips">${r.regions.slice(0, 6).map(id =>
          fbPill(`${id} (${r.regionCounts[id]})`, "neutral")).join("")}</div>
      </div></div>`).join("");

  const early = new Map();
  if (wp && wp.available) {
    (wp.earliest.events || []).forEach(e =>
      early.set(`${e.eventDate}|${e.eventType}`, e));
  }
  const list = (ev.events || [])
    .filter(e => f.kind === "all" || e.event_type === f.kind)
    .slice(0, 60)
    .sort((a, b) => (a.date < b.date ? 1 : -1));
  const evTable = list.length ? `
    <table class="fb-tbl"><thead><tr>
      <th>Date</th><th>Event type</th><th>Position</th><th>Intensity</th>
      <th>Earliest warning</th><th>Region</th><th></th></tr></thead>
    <tbody>${list.map(e => {
      const w = early.get(`${e.date}|${e.event_type}`);
      const region = (imp.rows.find(r => r.eventType === e.event_type) || {})
        .regions;
      return `<tr>
        <td>${fbEsc(e.date)}</td>
        <td>${fbEsc(String(e.event_type).replace(/_/g, " "))}</td>
        <td>${e.lat.toFixed(1)}° ${e.lon.toFixed(1)}°</td>
        <td>${e.intensity_hpa}</td>
        <td>${w ? (w.issued
          ? `${fbPill(`D${w.earliestWarningLead}`, "up")} <span class="fb-note">${w.hoursBeforeValid} h ahead</span>`
          : fbPill("not issued", "down")) : `<span class="fb-note">not sampled</span>`}</td>
        <td>${region && region.length ? fbEsc(region[0]) : `<span class="fb-note">outside all regions</span>`}</td>
        <td><button class="fb-btn ghost" onclick="fbOpenCase('${fbEsc(e.date)}',${e.lat},${e.lon})">Case study</button></td>
      </tr>`;}).join("")}</tbody></table>` :
    fbEmpty("No events", "The catalogue returned no rows for this filter.");

  const warnBlock = wp && wp.available ? `
    <div class="fb-card">
      <div class="fb-card-h">Earliest warning · ${wp.earliest.summary.eventsEvaluated} events sampled</div>
      <div class="fb-card-b">
        <div class="fb-kpis" style="grid-template-columns:repeat(4,1fr)">
          <div class="fb-kpi"><div class="fb-kpi-k">Detected</div>
            <div class="fb-kpi-v">${wp.earliest.summary.eventsWithWarning}
              <small> / ${wp.earliest.summary.eventsEvaluated}</small></div></div>
          <div class="fb-kpi"><div class="fb-kpi-k">Detection rate</div>
            <div class="fb-kpi-v">${wp.earliest.summary.detectionRate === null ? "—"
              : Math.round(wp.earliest.summary.detectionRate * 100) + "%"}</div></div>
          <div class="fb-kpi"><div class="fb-kpi-k">Median lead</div>
            <div class="fb-kpi-v">${wp.earliest.summary.medianEarliestLead ?? "—"}
              <small> d</small></div></div>
          <div class="fb-kpi"><div class="fb-kpi-k">Best lead</div>
            <div class="fb-kpi-v">${wp.earliest.summary.bestLead ?? "—"}<small> d</small></div></div>
        </div>
        <div class="fb-note">Rule: ${fbEsc(wp.rule)}. ${fbEsc(wp.caveat)}</div>
      </div></div>` : fbEmpty("Earliest warning unavailable",
        wp ? fbEsc(wp.message || "Unavailable in current data mode.")
           : "Warning performance unavailable.");

  el.innerHTML = `
    ${head}
    <div class="fb-kpis">
      <div class="fb-kpi"><div class="fb-kpi-k">Catalogue events</div>
        <div class="fb-kpi-v">${total}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Event types</div>
        <div class="fb-kpi-v">${rows.length}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Regions affected</div>
        <div class="fb-kpi-v">${regionsCovered}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Cycles sampled</div>
        <div class="fb-kpi-v">${wp ? wp.cyclesSampled.length : "—"}</div></div>
    </div>
    <div class="fb-grid2">${typeCards}</div>
    ${warnBlock}
    <div class="fb-card">
      <div class="fb-card-h">Event catalogue · ${list.length} most recent shown</div>
      <div class="fb-card-b">${evTable}</div>
      <div class="fb-card-f"><div class="fb-note">${fbEsc(imp.method || "")}</div></div>
    </div>`;
}

/* ═════════════════════════ TAB: PERFORMANCE ═════════════════════════ */
async function fbOpsPerformance(el) {
  el.innerHTML = fbLoading("Loading lead scorecard…");
  let wp;
  try { wp = await fbpJSON("/api/warning-performance?sample=12&limit=60"); }
  catch (e) { el.innerHTML = fbErr("Performance unavailable", e); return; }
  if (!wp.available) {
    el.innerHTML = fbEmpty("Performance unavailable", fbEsc(wp.message || wp.reason || "Unavailable in current data mode."));
    return;
  }
  const o = wp.scorecard.overall;
  const pct = v => (v === null || v === undefined) ? "—" : Math.round(v * 100) + "%";
  const byLead = wp.scorecard.byLead;
  const chart = fbLineChart([
    {label: "Recall (hit rate)", color: "var(--accent)", points: byLead.map(r => ({y: r.recall}))},
    {label: "Precision", color: "var(--green)", points: byLead.map(r => ({y: r.precision}))},
    {label: "False alarm ratio", color: "var(--red)", dashed: true,
     points: byLead.map(r => ({y: r.falseAlarmRatio}))}
  ], {xs: byLead.map(r => "D" + r.lead), min: 0, max: 1,
      fmtY: v => Math.round(v * 100) + "%", aria: "skill by forecast lead"});

  el.innerHTML = `
    <div class="fb-kpis">
      <div class="fb-kpi"><div class="fb-kpi-k">Accuracy</div>
        <div class="fb-kpi-v">${pct(o.accuracy)}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Precision</div>
        <div class="fb-kpi-v">${pct(o.precision)}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Recall</div>
        <div class="fb-kpi-v">${pct(o.recall)}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">False alarm ratio</div>
        <div class="fb-kpi-v">${pct(o.falseAlarmRatio)}</div></div>
    </div>
    <div class="fb-grid2">
      <div class="fb-card">
        <div class="fb-card-h">Skill by forecast lead</div>
        <div class="fb-card-b">${chart}</div>
        <div class="fb-card-f"><div class="fb-note">Rule: ${fbEsc(wp.rule)}.
          ${byLead.length} leads, ${o.total} grid-lead samples.</div></div>
      </div>
      <div class="fb-card">
        <div class="fb-card-h">By event type</div>
        <div class="fb-card-b">
          <table class="fb-tbl"><thead><tr><th>Type</th><th>Events</th>
            <th>Hit</th><th>Miss</th><th>FA</th><th>Recall</th></tr></thead>
          <tbody>${wp.scorecard.byEventType.map(r => `<tr>
            <td>${fbEsc(r.eventType.replace(/_/g, " "))}</td>
            <td>${r.events}</td><td>${r.hits}</td><td>${r.misses}</td>
            <td>${r.falseAlarms}</td><td>${pct(r.recall)}</td></tr>`).join("")}
          </tbody></table>
        </div>
      </div>
    </div>
    <div class="fb-card">
      <div class="fb-card-h">Lead scorecard</div>
      <div class="fb-card-b">
        <table class="fb-tbl"><thead><tr><th>Lead</th><th>Events</th><th>Hits</th>
          <th>Misses</th><th>False alarms</th><th>Correct neg.</th>
          <th>Accuracy</th><th>Precision</th><th>Recall</th></tr></thead>
        <tbody>${byLead.map(r => `<tr>
          <td>D${r.lead}</td><td>${r.events}</td><td>${r.hits}</td>
          <td>${r.misses}</td><td>${r.falseAlarms}</td><td>${r.correctNegatives}</td>
          <td>${pct(r.accuracy)}</td><td>${pct(r.precision)}</td>
          <td>${pct(r.recall)}</td></tr>`).join("")}</tbody></table>
      </div>
      <div class="fb-card-f"><div class="fb-note">${fbEsc(wp.caveat)}</div></div>
    </div>`;
}

/* ═══════════════════════════ TAB: TRUST ═══════════════════════════ */
async function fbOpsTrust(el) {
  el.innerHTML = fbLoading("Loading provenance…");
  let t;
  try { t = await fbpJSON("/api/trust"); }
  catch (e) { el.innerHTML = fbErr("Trust panel unavailable", e); return; }
  const table = (title, rows, cols) => `
    <div class="fb-card">
      <div class="fb-card-h">${fbEsc(title)}</div>
      <div class="fb-card-b"><table class="fb-tbl">
        <thead><tr>${cols.map(c => `<th>${fbEsc(c)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map(r => `<tr>${cols.map((c, i) =>
          `<td>${fbEsc(String(r[i] ?? "—"))}</td>`).join("")}</tr>`).join("")}
        </tbody></table></div></div>`;
  el.innerHTML = `
    <div class="fb-grid2">
      ${table("Provenance", t.provenance.map(p => [p.label, p.value, p.detail]),
              ["What", "Source", "Detail"])}
      ${table("Thresholds in force", t.thresholds.map(x => [x.name, x.value, x.source]),
              ["Threshold", "Value", "Where it is defined"])}
    </div>
    <div class="fb-grid2">
      <div class="fb-card">
        <div class="fb-card-h">Model artefacts</div>
        <div class="fb-card-b">${t.artefacts.length ? `<table class="fb-tbl">
          <thead><tr><th>File</th><th>Size</th><th>Modified</th></tr></thead>
          <tbody>${t.artefacts.map(a => `<tr><td>${fbEsc(a.name)}</td>
            <td>${(a.bytes / 1024).toFixed(1)} KB</td>
            <td>${fbEsc(a.modified)}</td></tr>`).join("")}</tbody></table>`
          : fbEmpty("No artefacts found", "Run `make train` to build the models.")}</div>
        <div class="fb-card-f"><div class="fb-note">Generated ${fbEsc(t.generatedAt)}</div></div>
      </div>
      <div class="fb-card">
        <div class="fb-card-h">Ensemble members</div>
        <div class="fb-card-b">
          <div class="fb-kpi"><div class="fb-kpi-k">Declared members</div>
            <div class="fb-kpi-v">${t.members.declared || "—"}</div></div>
          <div class="fb-note">${fbEsc(t.members.note || "")}</div>
        </div>
      </div>
    </div>
    <div class="fb-card">
      <div class="fb-card-h">What this system does not do</div>
      <div class="fb-card-b"><ul style="margin:0;padding-left:18px;color:var(--t2);font-size:12.5px;line-height:1.7">
        ${t.limitations.map(l => `<li>${fbEsc(l)}</li>`).join("")}</ul></div>
    </div>`;
}

/* ═══════════════════════════ TAB: BRIEF ═══════════════════════════ */
async function fbOpsBrief(el) {
  el.innerHTML = fbLoading("Assembling brief…");
  let b;
  try {
    b = await fbpJSON(`/api/forecast-brief?cycle=${fbpCycle() || ""}&lead=${fbpLead()}`);
  } catch (e) { el.innerHTML = fbErr("Brief unavailable", e); return; }
  if (!b.available) {
    el.innerHTML = fbEmpty("Brief unavailable", fbEsc(b.reason || "Unavailable in current data mode."));
    return;
  }
  const text = [b.headline, ""].concat(
    b.sections.map(s => s.heading + "\n" + s.lines.map(l => "  - " + l).join("\n")),
    ["", "Caveats"].concat(b.caveats.map(c => "  - " + c))).join("\n");
  el.innerHTML = `
    <div class="fb-kpis">
      <div class="fb-kpi"><div class="fb-kpi-k">Cycle</div>
        <div class="fb-kpi-v">${fbEsc(b.cycle)}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Lead</div>
        <div class="fb-kpi-v">D${b.lead}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Generated</div>
        <div class="fb-kpi-v" style="font-size:13px">${fbEsc(b.generatedAt)}</div></div>
      <div class="fb-kpi"><div class="fb-kpi-k">Sections</div>
        <div class="fb-kpi-v">${b.sections.length}</div></div>
    </div>
    <div class="fb-card">
      <div class="fb-card-h">Forecast brief</div>
      <div class="fb-card-b">
        <div class="fb-note" style="font-size:13.5px;color:var(--t1);margin-bottom:10px">
          ${fbEsc(b.headline)}</div>
        ${b.sections.map(s => `
          <div style="margin-top:12px">
            <div class="fb-card-h" style="border:0;padding:0 0 6px">${fbEsc(s.heading)}</div>
            <ul style="margin:0;padding-left:18px;color:var(--t2);font-size:12.5px;line-height:1.7">
              ${s.lines.map(l => `<li>${fbEsc(l)}</li>`).join("")}</ul>
          </div>`).join("")}
      </div>
      <div class="fb-card-f">
        <button class="fb-btn" id="fbCopyBrief">Copy as text</button>
        <span class="fb-note">${b.caveats.map(fbEsc).join(" · ")}</span>
      </div>
    </div>`;
  const btn = document.getElementById("fbCopyBrief");
  if (btn) btn.onclick = () => {
    if (navigator.clipboard) navigator.clipboard.writeText(text);
    btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = "Copy as text"; }, 1600);
  };
}

/* ═══════════════════════════ GLOBAL SEARCH ═══════════════════════════ */
let _fbSearchTok = 0;
async function fbSearch(q) {
  const pop = document.getElementById("fbSearchPop");
  if (!pop) return;
  const tok = ++_fbSearchTok;
  if (!q || !q.trim()) {
    pop.hidden = false;
    pop.innerHTML = `<div class="fb-search-none">Type to search regions,
      alert rules, event types and sections.</div>`;
    return;
  }
  let d;
  try { d = await fbpJSON(`/api/search?q=${encodeURIComponent(q)}&lead=${fbpLead()}`); }
  catch (e) {
    if (tok !== _fbSearchTok) return;
    pop.hidden = false;
    pop.innerHTML = `<div class="fb-search-none">Search unavailable.</div>`;
    return;
  }
  if (tok !== _fbSearchTok) return;
  pop.hidden = false;
  if (!d.hits.length) {
    pop.innerHTML = `<div class="fb-search-none">No matches for “${fbEsc(q)}”.</div>`;
    return;
  }
  const grouped = {};
  d.hits.forEach(h => { (grouped[h.type] = grouped[h.type] || []).push(h); });
  pop.innerHTML = Object.keys(grouped).map(ns =>
    `<div class="fb-search-ns">${fbEsc(ns)}</div>` +
    grouped[ns].slice(0, 6).map(h => `<div class="fb-search-hit"
      onmousedown="fbSearchGo('${fbEsc(h.view || "")}','${fbEsc(h.tab || "")}')">
      <span>${fbEsc(h.label)}</span><small>${fbEsc(h.detail || "")}</small></div>`).join("")
  ).join("");
}
function fbSearchGo(view, tab) {
  const pop = document.getElementById("fbSearchPop");
  if (pop) pop.hidden = true;
  if (view) {
    if (typeof show === "function") show(view);
    if (view === "ops" && tab) { FBP.opsTab = tab; fbpRenderOps(); }
    if (view === "intel" && tab) { FBP.intelTab = tab; fbpRenderIntel(); }
  }
}
function fbSearchClose() {
  const pop = document.getElementById("fbSearchPop");
  if (pop) pop.hidden = true;
}
