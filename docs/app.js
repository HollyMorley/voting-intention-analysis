// ============================================================
// app.js — the behaviour of the dashboard.
//
// All modelling is done in Python (analysis/export_dashboard.py) and frozen
// into docs/data/*.json. This file only reads those numbers and draws them.
// Nothing here trains a model or does heavy maths.
//
// The two value axes are `econ` (economic left -> right) and `natint`
// (nationalism-internationalism: internationalist/open -> nationalist/closed).
//
// Layout:
//   - load the four JSON files
//   - a global "policy" (all parties / main parties only) filters everything
//   - one draw function per section, re-run when a control changes
// ============================================================

const FILES = ["respondents", "importance", "model_eval", "meta", "persona_model"];
const PLOT_CFG = { displayModeBar: false, responsive: true };
const FONT = { family: "-apple-system, Segoe UI, Roboto, sans-serif", size: 12 };

// Fixed orderings for the ordinal demographics (everything else sorts by size).
const ORDERS = {
  age: ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
  political_interest: ["None at all", "Not very much", "Some", "Quite a lot", "A great deal"],
  income: [
    "Up to £7,000", "£7,001 to £14,000", "£14,001 to £21,000", "£21,001 to £28,000",
    "£28,001 to £34,000", "£34,001 to £41,000", "£41,001 to £48,000", "£48,001 to £55,000",
    "£55,001 to £62,000", "£62,001 to £69,000", "£69,001 to £76,000", "£76,001 to £83,000",
    "£83,001 or more", "Prefer not to answer",
  ],
};

// One-line description of each feature shown above the chart.
const DEMO_DESC = {
  age:               "Respondent's age group (18–24 through 65+).",
  region:            "Region of the UK where the respondent lives.",
  eu_referendum:     "How the respondent voted in the 2016 Brexit referendum (Remain, Leave, or did not vote).",
  political_interest: "Self-reported level of interest in politics, from 'None at all' to 'A great deal'.",
  work_organisation: "Type of organisation the respondent works or worked for - public sector, private sector, self-employed, etc.",
  goal:              "The respondent's top priority from a list of global challenges (e.g. climate change, poverty, gender equality).",
  working_class:     "Illustrative socioeconomic characteristic - whether the respondent agrees they consider themselves working class.",
  care_disabled_child: "Illustrative vulnerability characteristic - whether the respondent cares for a child with a serious health issue or disability.",
  public_transport:  "Illustrative cosmopolitan characteristic - whether the respondent regularly uses public transport.",
  opinion_spread:    "Illustrative response style characteristic - how strongly opinionated the respondent is across all attitude items, split into quartiles.",
};

const state = { policy: "all" };
let DATA = {};

// "Another party" is excluded from the "main parties only" view in addition to
// the three flagged as small in the JSON (SNP, Plaid Cymru, Rather not say).
const ALSO_SMALL = ["Another party"];

// ---------- small helpers ----------
const pct = (x) => `${Math.round(x * 100)}%`;
const colourOf = (party) => DATA.meta.party_colours[party] || "#9e9e9e";

function isSmall(p) {
  return DATA.meta.small.includes(p) || ALSO_SMALL.includes(p);
}

// Respondents to show under the current policy ("main" hides the tiny groups).
function respondents() {
  return state.policy === "all"
    ? DATA.respondents
    : DATA.respondents.filter((r) => !isSmall(r.vote));
}
// Parties/classes to show, in display order, under the current policy.
function classes() {
  const order = DATA.meta.class_order;
  return state.policy === "all" ? order : order.filter((c) => !isSmall(c));
}
function decidedParties() {
  return DATA.meta.decided.filter((p) => state.policy === "all" || !isSmall(p));
}

// Order a set of category labels: fixed order if we have one, else by frequency.
function orderCategories(cats, key, counts) {
  if (ORDERS[key]) return ORDERS[key].filter((c) => cats.includes(c));
  return [...cats].sort((a, b) => (counts[b] || 0) - (counts[a] || 0));
}

// Linear-interpolation percentile on a pre-sorted array.
function quantile(sorted, p) {
  const i = p * (sorted.length - 1);
  const lo = Math.floor(i), hi = Math.ceil(i);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

// ============================================================
// SECTION 1 - the values map
// ============================================================

// Fixed axis ranges derived once from the full dataset so the axes never shift
// between "all parties" / "main parties only" or individuals on/off.
function compassAxisRanges() {
  const pad = 0.05;
  const natints = DATA.respondents.map((r) => r.natint).filter((v) => v != null);
  const econs   = DATA.respondents.map((r) => r.econ).filter((v) => v != null);
  return {
    x: [Math.min(...natints) - pad, Math.max(...natints) + pad],
    y: [Math.min(...econs)   - pad, Math.max(...econs)   + pad],
  };
}

function drawCompass() {
  const rows = respondents();
  const showUncommitted = document.getElementById("showUndecided").checked;
  const showIndividuals = document.getElementById("showIndividuals").checked;
  const axRange = compassAxisRanges();
  const traces = [];

  // Scatter traces are always added to keep the legend stable (same entries, same
  // width). When individuals are hidden we use 'legendonly' so the dots disappear
  // but the legend entry - and its pixel width - stays identical.
  for (const party of classes()) {
    const isUncommitted = DATA.meta.undecided.includes(party);
    if (isUncommitted && !showUncommitted) continue;
    const pts = rows.filter((r) => r.vote === party);
    if (!pts.length) continue;
    traces.push({
      x: pts.map((r) => r.natint),
      y: pts.map((r) => r.econ),
      mode: "markers",
      type: "scatter",
      name: `${party} (${pts.length})`,
      visible: showIndividuals ? true : "legendonly",
      marker: { size: 6, color: colourOf(party), opacity: 0.5, line: { width: 0 } },
      hovertemplate: `${party}<br>nat-int %{x:.2f}<br>economic %{y:.2f}<extra></extra>`,
    });
  }

  // Average positions always shown (including uncommitted when toggled), with
  // IQR (25th–75th percentile) error bars showing the spread of voter positions.
  // Scatter traces own the legend so means stay off it.
  for (const party of classes()) {
    const isUncommitted = DATA.meta.undecided.includes(party);
    if (isUncommitted && !showUncommitted) continue;
    const pts = rows.filter((r) => r.vote === party);
    if (pts.length < 5) continue;
    const n = pts.length;
    const mEcon   = pts.reduce((s, r) => s + r.econ,   0) / n;
    const mNatint = pts.reduce((s, r) => s + r.natint, 0) / n;
    const econSorted   = pts.map((r) => r.econ).sort((a, b) => a - b);
    const natintSorted = pts.map((r) => r.natint).sort((a, b) => a - b);
    const q1Econ = quantile(econSorted, 0.25),   q3Econ = quantile(econSorted, 0.75);
    const q1Natint = quantile(natintSorted, 0.25), q3Natint = quantile(natintSorted, 0.75);
    traces.push({
      x: [mNatint], y: [mEcon], mode: "markers", type: "scatter",
      name: party, showlegend: false,
      error_x: { type: "data", symmetric: false, array: [q3Natint - mNatint], arrayminus: [mNatint - q1Natint], visible: true, color: colourOf(party), thickness: 2, width: 6 },
      error_y: { type: "data", symmetric: false, array: [q3Econ - mEcon],     arrayminus: [mEcon - q1Econ],     visible: true, color: colourOf(party), thickness: 2, width: 6 },
      marker: { size: 10, color: colourOf(party), symbol: "diamond", line: { color: "#fff", width: 2 } },
      hovertemplate: `${party} mean<br>nat-int %{x:.2f} (IQR ${q1Natint.toFixed(2)}–${q3Natint.toFixed(2)})<br>economic %{y:.2f} (IQR ${q1Econ.toFixed(2)}–${q3Econ.toFixed(2)})<extra></extra>`,
    });
  }

  const layout = {
    margin: { t: 10, r: 10, b: 50, l: 60 }, font: FONT, hovermode: "closest",
    xaxis: { title: "← internationalist / open      NAT–INT      nationalist / closed →", zeroline: true, zerolinecolor: "#cbd3dc", range: axRange.x },
    yaxis: { title: "← economic left      ECONOMIC      economic right →", zeroline: true, zerolinecolor: "#cbd3dc", range: axRange.y },
    legend: { font: { size: 10 }, itemsizing: "constant", bgcolor: "rgba(255,255,255,.6)" },
  };
  Plotly.react("compassChart", traces, layout, PLOT_CFG);
}

// ============================================================
// SECTION 4 - vote intention by demographic group
// ============================================================
function drawDemo() {
  const rows = respondents();
  const key = document.getElementById("demoSelect").value;
  const mode = document.querySelector("#demoModeToggle .active").dataset.mode;
  const cls = classes();

  // counts[category][party]
  const counts = {}, totals = {};
  for (const r of rows) {
    const cat = r[key] == null ? "No answer" : String(r[key]);
    (counts[cat] ||= {});
    counts[cat][r.vote] = (counts[cat][r.vote] || 0) + 1;
    totals[cat] = (totals[cat] || 0) + 1;
  }
  const cats = orderCategories(Object.keys(counts), key, totals).reverse(); // reversed: first reads at top

  const traces = cls.map((party) => ({
    type: "bar", orientation: "h", name: party,
    y: cats,
    x: cats.map((c) => {
      const n = counts[c][party] || 0;
      return mode === "share" ? (totals[c] ? (100 * n) / totals[c] : 0) : n;
    }),
    marker: { color: colourOf(party) },
    hovertemplate: `${party}: %{x:.0f}${mode === "share" ? "%" : ""}<extra>%{y}</extra>`,
  }));

  const layout = {
    barmode: "stack", margin: { t: 10, r: 10, b: 100, l: 150 }, font: FONT,
    xaxis: { title: mode === "share" ? "Share of group (%)" : "Number of respondents", ticksuffix: mode === "share" ? "%" : "" },
    yaxis: { automargin: true },
    legend: { font: { size: 10 }, orientation: "h", y: -0.28 },
  };
  Plotly.react("demoChart", traces, layout, PLOT_CFG);
  document.getElementById("demoDesc").textContent = DEMO_DESC[key] || "";
}

// ============================================================
// SECTION 2 - the model
// ============================================================
function updateModelStats() {
  const e = DATA.model_eval[state.policy];
  document.getElementById("statAcc").textContent = pct(e.cv_balanced_accuracy);
  document.getElementById("statBaseline").textContent = pct(e.baselines["always Labour (most frequent)"]);
  const g = e.gap_vs_baseline;
  document.getElementById("statGap").textContent =
    `+${Math.round(g.mean * 100)}%`;
  document.getElementById("statGap").title =
    `95% range: +${Math.round(g.lo * 100)} to +${Math.round(g.hi * 100)} points`;
}

function drawImportance() {
  const metric = document.querySelector("#impToggle .active").dataset.metric;
  const imp = DATA.importance[state.policy];
  const blocks = [...imp.blocks].sort(
    (a, b) => a[`${metric}_mean`] - b[`${metric}_mean`]
  ); // ascending so the biggest sits at top of a horizontal bar
  const labels = blocks.map((b) => b.label);
  const vals = blocks.map((b) => b[`${metric}_mean`]);
  const lo = blocks.map((b) => b[`${metric}_mean`] - b[`${metric}_lo`]);
  const hi = blocks.map((b) => b[`${metric}_hi`] - b[`${metric}_mean`]);
  const sig = blocks.map((b) => b[`${metric}_sig`]);

  const trace = {
    type: "bar", orientation: "h", x: vals, y: labels,
    error_x: { type: "data", symmetric: false, array: hi, arrayminus: lo, color: "#94a3b8", thickness: 1 },
    marker: { color: "#2f5d8a" },
    text: sig.map((s) => (s === "n.s." ? "" : s)),
    textposition: "outside", cliponaxis: false,
    hovertemplate: `%{y}<br>${metric === "perm" ? "model importance" : "balanced accuracy"}: %{x:.3f}<extra></extra>`,
  };
  const refX = metric === "perm" ? 0 : imp.chance;
  const layout = {
    margin: { t: 10, r: 30, b: 44, l: 140 }, font: FONT,
    xaxis: {
      title: metric === "perm"
        ? "Drop in balanced accuracy when this group is scrambled (bigger = more important)"
        : "Balanced accuracy using only this group of answers",
      zeroline: false,
    },
    yaxis: { automargin: true },
    shapes: [{ type: "line", x0: refX, x1: refX, yref: "paper", y0: 0, y1: 1, line: { color: "#cbd3dc", dash: "dot" } }],
    annotations: metric === "uni"
      ? [{ x: imp.chance, y: 1, yref: "paper", text: `chance (${pct(imp.chance)})`, showarrow: false, font: { size: 10, color: "#94a3b8" }, xanchor: "left", yanchor: "bottom" }]
      : [],
  };
  Plotly.react("importanceChart", [trace], layout, PLOT_CFG);
}

function drawConfusion() {
  const cm = DATA.model_eval[state.policy].confusion;
  const labels = cm.labels;
  // y reversed so the diagonal runs top-left -> bottom-right
  const z = [...cm.matrix].reverse();
  const yLabels = [...labels].reverse();
  const text = z.map((row) => row.map((v) => (v ? v.toFixed(2) : "")));
  const trace = {
    type: "heatmap", z, x: labels, y: yLabels, text, texttemplate: "%{text}",
    textfont: { size: 10 }, colorscale: "Blues", zmin: 0, zmax: 1, xgap: 1, ygap: 1,
    colorbar: { title: "share of row", thickness: 12, len: 0.8 },
    hovertemplate: "actual %{y}<br>predicted %{x}<br>%{z:.0%}<extra></extra>",
  };
  const layout = {
    margin: { t: 10, r: 10, b: 90, l: 130 }, font: FONT,
    xaxis: { title: "Model's prediction", tickangle: -40, automargin: true },
    yaxis: { title: "Actually intends to vote", automargin: true },
  };
  Plotly.react("confusionChart", [trace], layout, PLOT_CFG);
}

// ============================================================
// SECTION 3 - persona prediction (client-side RF inference)
// ============================================================

function rfPredictTree(tree, x) {
  let node = 0;
  while (tree.feature[node] !== -2) {
    node = x[tree.feature[node]] <= tree.threshold[node]
      ? tree.left[node] : tree.right[node];
  }
  return tree.value[node];
}

function rfPredict(x) {
  const pm = DATA.persona_model;
  const n = pm.classes.length;
  const acc = new Array(n).fill(0);
  for (const tree of pm.trees) {
    const v = rfPredictTree(tree, x);
    for (let i = 0; i < n; i++) acc[i] += v[i];
  }
  const total = acc.reduce((a, b) => a + b, 0);
  return pm.classes.map((c, i) => [c, acc[i] / total]);
}

// Build a full feature vector from a median base, overriding the 4 political features.
// EU vote is now a demographic dimension (baked into medians), not an override.
function buildFeatureVector(medians, polInt, econ, natint, goalStr) {
  const ctrl = DATA.persona_model.control;
  const x = [...medians];
  x[ctrl.political_interest] = polInt;
  x[ctrl.econ]               = econ;
  x[ctrl.natint]             = natint;
  for (const idx of Object.values(ctrl.goal)) x[idx] = 0;
  if (ctrl.goal[goalStr] != null) x[ctrl.goal[goalStr]] = 1;
  return x;
}

function demoCombinedKey() {
  const eu = document.querySelector("#personaEuToggle button.active")?.dataset.eu ?? "All";
  return ["personaRegion", "personaAge", "personaGender", "personaWorkOrg"]
    .map((id) => document.getElementById(id).value).concat(eu).join("|");
}

function describePolInt(v) {
  return DATA.persona_model.pol_int_labels[Math.round(v)] ?? String(Math.round(v));
}
function describeEcon(v)   { return v > 0.15 ? "right" : v < -0.15 ? "left" : "centre"; }
function describeNatint(v) { return v > 0.15 ? "nationalist" : v < -0.15 ? "internationalist" : "centre"; }

function updateVirtLabels() {
  document.getElementById("virtPolIntLabel").textContent =
    describePolInt(+document.getElementById("virtPolInt").value);
  document.getElementById("virtEconLabel").textContent =
    describeEcon(+document.getElementById("virtEcon").value);
  document.getElementById("virtNatintLabel").textContent =
    describeNatint(+document.getElementById("virtNatint").value);
}

function getVirtValues() {
  return {
    polInt: +document.getElementById("virtPolInt").value,
    econ:   +document.getElementById("virtEcon").value,
    natint: +document.getElementById("virtNatint").value,
    goal:   document.getElementById("virtGoal").value,
  };
}

// Shared state: current demographic medians + reference values.
const _persona = { medians: null, ref: null };

function onDemoChange() {
  const key = demoCombinedKey();
  const entry = DATA.persona_model.demographics[key];
  const info  = document.getElementById("personaDemoInfo");

  if (!entry) {
    info.style.display = "";
    info.innerHTML = "<b>No respondents</b> exactly match this combination &mdash; try adjusting your selection.";
    _persona.medians = null;
    _persona.ref = null;
    Plotly.purge("personaChart");
    return;
  }

  info.style.display = "";
  info.innerHTML = entry.n < 10
    ? `<b>Thin data:</b> only ${entry.n} respondent${entry.n === 1 ? "" : "s"} match this profile &mdash; treat predictions with caution.`
    : `${entry.n} respondents match this demographic profile.`;

  _persona.medians = entry.medians;
  _persona.ref = entry.ref;

  // Update reference display.
  document.getElementById("refPolInt").textContent  = describePolInt(entry.ref.pol_int);
  document.getElementById("refEcon").textContent    = describeEcon(entry.ref.econ);
  document.getElementById("refNatint").textContent  = describeNatint(entry.ref.natint);
  document.getElementById("refGoal").textContent    = entry.ref.goal;

  // Reset virtual panel to demographic reference values.
  document.getElementById("virtPolInt").value = Math.round(entry.ref.pol_int);
  document.getElementById("virtEcon").value   = entry.ref.econ.toFixed(2);
  document.getElementById("virtNatint").value = entry.ref.natint.toFixed(2);
  document.getElementById("virtGoal").value   = entry.ref.goal;

  updateVirtLabels();
  drawPersonaBars();
}

function onVirtChange() {
  updateVirtLabels();
  drawPersonaBars();
}

function drawPersonaBars() {
  if (!_persona.medians) return;

  const ref  = _persona.ref;
  const virt = getVirtValues();

  const refVec  = buildFeatureVector(_persona.medians, ref.pol_int,  ref.econ,    ref.natint,    ref.goal);
  const virtVec = buildFeatureVector(_persona.medians, virt.polInt, virt.econ, virt.natint, virt.goal);

  const refMap  = Object.fromEntries(rfPredict(refVec));
  const virtMap = Object.fromEntries(rfPredict(virtVec));

  const parties = classes().filter((c) => refMap[c] != null || virtMap[c] != null);
  const colors  = parties.map(colourOf);
  const refY    = parties.map((c) => Math.round((refMap[c]  ?? 0) * 100));
  const virtY   = parties.map((c) => Math.round((virtMap[c] ?? 0) * 100));

  Plotly.react("personaChart", [
    {
      type: "bar", name: "Reference (demographic avg)",
      x: parties, y: refY,
      marker: { color: colors },
      hovertemplate: "%{x}: %{y}%<extra>Reference</extra>",
    },
    {
      type: "bar", name: "Virtual person",
      x: parties, y: virtY,
      marker: { color: "rgba(0,0,0,0)", line: { color: colors, width: 2.5 } },
      hovertemplate: "%{x}: %{y}%<extra>Virtual</extra>",
    },
  ], {
    barmode: "group",
    margin: { t: 24, r: 10, b: 90, l: 44 },
    font: FONT,
    yaxis: { title: "Predicted probability (%)", ticksuffix: "%", range: [0, 100] },
    xaxis: { tickangle: -35, automargin: true },
    legend: { orientation: "h", y: -0.3, font: { size: 11 } },
    plot_bgcolor: "rgba(0,0,0,0)",
    paper_bgcolor: "rgba(0,0,0,0)",
  }, PLOT_CFG);
}

// ============================================================
// SECTION 5 - swayable voters
// ============================================================
function swayCandidates(party, mode) {
  const rows = respondents();
  const dec = decidedParties();
  // rank a respondent's decided parties by the model's probability
  const decidedRank = (r) => dec.map((p) => [p, r.proba[p] ?? 0]).sort((a, b) => b[1] - a[1]);

  let pool;
  if (mode === "undecided") {
    // undecided people whose strongest party-lean is this party
    pool = rows.filter((r) => DATA.meta.undecided.includes(r.vote) && decidedRank(r)[0][0] === party);
  } else {
    // committed elsewhere: model predicted them for this party (incorrect prediction)
    // OR this party is their closest alternative (second-highest probability)
    pool = rows.filter((r) => {
      if (DATA.meta.undecided.includes(r.vote) || r.vote === party) return false;
      const rank = decidedRank(r);
      return (rank[0] && rank[0][0] === party) || (rank[1] && rank[1][0] === party);
    });
  }
  return pool.sort((a, b) => (b.proba[party] ?? 0) - (a.proba[party] ?? 0));
}

function drawSway() {
  const party = document.getElementById("swayParty").value;
  const mode = document.querySelector("#swayModeToggle .active").dataset.mode;
  const demoKey = document.getElementById("swayDemo").value;
  const K = +document.getElementById("swayK").value;
  document.getElementById("swayKVal").textContent = K;

  const pool = swayCandidates(party, mode);
  const group = pool.slice(0, K);
  const summary = document.getElementById("swaySummary");

  if (!group.length) {
    summary.textContent = "No one in the survey fits this combination - try the other pool or another party.";
    Plotly.purge("swayCompass");
    Plotly.purge("swayDemoChart");
    return;
  }

  const mean = (arr, f) => arr.reduce((s, r) => s + f(r), 0) / arr.length;
  const gEcon = mean(group, (r) => r.econ);
  const gNatint = mean(group, (r) => r.natint);

  // committed voters for the same party, as a reference point
  const committed = respondents().filter((r) => r.vote === party);
  const cEcon = committed.length ? mean(committed, (r) => r.econ) : null;
  const cNatint = committed.length ? mean(committed, (r) => r.natint) : null;

  // headline sentence
  const econWord = gEcon > 0.03 ? "lean economically right" : gEcon < -0.03 ? "lean economically left" : "sit in the economic centre";
  const natWord = gNatint > 0.03 ? "more nationalist / closed" : gNatint < -0.03 ? "more internationalist / open" : "mixed on nationalism";
  const topCat = (() => {
    const c = {};
    for (const r of group) { const v = r[demoKey] ?? "No answer"; c[v] = (c[v] || 0) + 1; }
    return Object.entries(c).sort((a, b) => b[1] - a[1])[0];
  })();
  const poolWord = mode === "undecided" ? "uncommitted people leaning" : "voters who could switch";
  summary.innerHTML =
    `These are the <b>${group.length}</b> ${poolWord} toward <b>${party}</b>. ` +
    `On average they ${econWord} and are ${natWord}. ` +
    `The most common ${(DATA.meta.demographics[demoKey] || demoKey).toLowerCase()}: <b>${topCat[0]}</b> ` +
    `(${Math.round((100 * topCat[1]) / group.length)}% of them).`;

  // left chart: average values vs committed voters
  const compassTraces = [
    { x: [gNatint], y: [gEcon], mode: "markers+text", type: "scatter", name: "persuadable",
      text: ["persuadable"], textposition: "top center", textfont: { size: 10 },
      marker: { size: 18, color: colourOf(party), symbol: "star", line: { color: "#fff", width: 1.5 } },
      hovertemplate: "persuadable group<br>nat-int %{x:.2f}<br>economic %{y:.2f}<extra></extra>" },
  ];
  if (cEcon != null) {
    compassTraces.push({
      x: [cNatint], y: [cEcon], mode: "markers+text", type: "scatter", name: "committed",
      text: ["committed"], textposition: "bottom center", textfont: { size: 10 },
      marker: { size: 16, color: colourOf(party), symbol: "circle", opacity: 0.5, line: { color: "#fff", width: 1.5 } },
      hovertemplate: `${party} voters<br>nat-int %{x:.2f}<br>economic %{y:.2f}<extra></extra>`,
    });
  }
  const bound = 0.6;
  Plotly.react("swayCompass", compassTraces, {
    margin: { t: 10, r: 10, b: 40, l: 44 }, font: FONT, showlegend: false,
    xaxis: { title: "← open   nat-int   closed →", range: [-bound, bound], zeroline: true, zerolinecolor: "#cbd3dc" },
    yaxis: { title: "← left   econ   right →", range: [-bound, bound], zeroline: true, zerolinecolor: "#cbd3dc" },
  }, PLOT_CFG);

  // right chart: this group's make-up vs everyone, on the chosen demographic
  const all = respondents();
  const share = (rows) => {
    const c = {}; for (const r of rows) { const v = r[demoKey] ?? "No answer"; c[v] = (c[v] || 0) + 1; }
    const out = {}; for (const k in c) out[k] = c[k] / rows.length; return out;
  };
  const gShare = share(group), aShare = share(all);
  const totals = {}; for (const r of all) { const v = r[demoKey] ?? "No answer"; totals[v] = (totals[v] || 0) + 1; }
  const cats = orderCategories(Object.keys(totals), demoKey, totals);

  Plotly.react("swayDemoChart", [
    { type: "bar", name: "persuadable group", x: cats, y: cats.map((c) => 100 * (gShare[c] || 0)), marker: { color: colourOf(party) }, hovertemplate: "%{x}: %{y:.0f}%<extra>persuadable</extra>" },
    { type: "bar", name: "everyone", x: cats, y: cats.map((c) => 100 * (aShare[c] || 0)), marker: { color: "#cbd3dc" }, hovertemplate: "%{x}: %{y:.0f}%<extra>everyone</extra>" },
  ], {
    barmode: "group", margin: { t: 36, r: 10, b: 80, l: 40 }, font: FONT,
    xaxis: { tickangle: -35, automargin: true },
    yaxis: { title: "% of group", ticksuffix: "%" },
    legend: { font: { size: 10 }, orientation: "h", x: 0.5, y: 1.08, xanchor: "center", yanchor: "bottom" },
  }, PLOT_CFG);
}

// ============================================================
// Wiring
// ============================================================
function redrawAll() {
  drawCompass();
  drawDemo();
  updateModelStats();
  drawImportance();
  drawConfusion();
  refreshSwayParties();
  drawSway();
  drawPersonaBars();
}

// keep the target-party dropdown in sync with the policy (hide tiny parties)
function refreshSwayParties() {
  const sel = document.getElementById("swayParty");
  const current = sel.value;
  const parties = decidedParties();
  sel.innerHTML = "";
  for (const p of parties) {
    const o = document.createElement("option");
    o.value = p; o.textContent = p; sel.appendChild(o);
  }
  sel.value = parties.includes(current) ? current : "Labour";
}

function wireToggle(id, key, onChange) {
  document.querySelectorAll(`#${id} button`).forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(`#${id} button`).forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      onChange(btn.dataset[key]);
    });
  });
}

async function main() {
  // Prefer the inlined bundle (works when the page is opened directly from disk, where
  // fetch() of local files is blocked). Fall back to fetching the JSON when served over
  // http(s) without data.js present.
  if (window.DASHBOARD_DATA) {
    DATA = window.DASHBOARD_DATA;
  } else {
    const loaded = await Promise.all(
      FILES.map((f) => fetch(`data/${f}.json`).then((r) => r.json()))
    );
    FILES.forEach((f, i) => (DATA[f] = loaded[i]));
  }

  // populate the two demographic dropdowns
  const demos = DATA.meta.demographics;
  for (const sel of [document.getElementById("demoSelect"), document.getElementById("swayDemo")]) {
    for (const [k, label] of Object.entries(demos)) {
      const o = document.createElement("option");
      o.value = k; o.textContent = label; sel.appendChild(o);
    }
  }
  document.getElementById("swayDemo").value = "age";

  // global policy toggle
  wireToggle("policyToggle", "policy", (p) => { state.policy = p; redrawAll(); });

  // allow deep-linking a starting view, e.g. index.html?policy=main
  if (new URLSearchParams(location.search).get("policy") === "main") {
    state.policy = "main";
    document.querySelectorAll("#policyToggle button").forEach((b) =>
      b.classList.toggle("active", b.dataset.policy === "main"));
  }

  // per-section controls
  document.getElementById("showIndividuals").addEventListener("change", drawCompass);
  document.getElementById("showUndecided").addEventListener("change", drawCompass);
  document.getElementById("demoSelect").addEventListener("change", drawDemo);
  wireToggle("demoModeToggle", "mode", drawDemo);
  wireToggle("impToggle", "metric", drawImportance);
  document.getElementById("swayParty").addEventListener("change", drawSway);
  document.getElementById("swayDemo").addEventListener("change", drawSway);
  document.getElementById("swayK").addEventListener("input", drawSway);
  wireToggle("swayModeToggle", "mode", drawSway);

  // ---- persona section setup ----
  const pm = DATA.persona_model;
  const opts = pm.demo_options;

  function fillSelect(id, values, defaultVal) {
    const sel = document.getElementById(id);
    values.forEach((v) => {
      const o = document.createElement("option");
      o.value = v; o.textContent = v; sel.appendChild(o);
    });
    if (defaultVal != null && values.includes(defaultVal)) sel.value = defaultVal;
  }

  fillSelect("personaRegion",  opts.region,   "South East");
  fillSelect("personaAge",     opts.age,      "35-44");
  fillSelect("personaGender",  opts.gender,   "Male");
  fillSelect("personaWorkOrg", opts.work_org,
    opts.work_org.find((w) => w.startsWith("Private")) ?? opts.work_org[0]);
  fillSelect("virtGoal", Object.keys(pm.control.goal).sort(), null);

  ["personaRegion", "personaAge", "personaGender", "personaWorkOrg"].forEach((id) =>
    document.getElementById(id).addEventListener("change", onDemoChange));
  wireToggle("personaEuToggle", "eu", onDemoChange);
  ["virtPolInt", "virtEcon", "virtNatint"].forEach((id) =>
    document.getElementById(id).addEventListener("input", onVirtChange));
  document.getElementById("virtGoal").addEventListener("change", onVirtChange);

  refreshSwayParties();
  redrawAll();
  onDemoChange();
}

main();