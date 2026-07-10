/* Builds every Plotly figure of the gemma3 report from window.REPORT_DATA. */

const DATA = window.REPORT_DATA;
const MODEL_ORDER = DATA.model_order;

const COLORS = {
  gemma3_1b: "#1f77b4",
  gemma3_4b: "#ff7f0e",
  gemma3_12b: "#2ca02c",
  gemma3_27b: "#d62728",
};

const SHORT = {
  gemma3_1b: "1B",
  gemma3_4b: "4B",
  gemma3_12b: "12B",
  gemma3_27b: "27B",
};

const BASE_LAYOUT = {
  font: { family: "inherit", size: 13 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  margin: { l: 60, r: 20, t: 40, b: 50 },
};

const CONFIG = { responsive: true, displaylogo: false };

function layout(extra) {
  return Object.assign({}, BASE_LAYOUT, extra);
}

/** Models that have data for the given step key, in canonical order. */
function withStep(key) {
  return MODEL_ORDER.filter((m) => DATA.models[m] && DATA.models[m][key]);
}

function el(tag, attrs, parent) {
  const node = document.createElement(tag);
  Object.assign(node, attrs || {});
  if (parent) parent.appendChild(node);
  return node;
}

/** Adds a "Models shown: ..." note right under the section's <h2>. */
function coverageNote(sectionId, models) {
  const section = document.getElementById(sectionId);
  const note = el("p", { className: "coverage" });
  note.textContent =
    models.length > 0
      ? "Models shown: " + models.map((m) => SHORT[m]).join(", ")
      : "No data available for any model.";
  section.querySelector("h2").after(note);
  if (models.length === 0) {
    section
      .querySelectorAll(".plot, [id^='plots-']")
      .forEach((d) => (d.innerHTML = "<p class='no-data'>No data.</p>"));
  }
  return models.length > 0;
}

/* ------------------------------------------------------------------ */
/* Overview                                                            */
/* ------------------------------------------------------------------ */

function buildOverview() {
  const rows = MODEL_ORDER.map((m) => {
    const d = DATA.models[m];
    const meta = d && d.meta;
    const steps = d && d.steps_present.length ? d.steps_present.join(", ") : "—";
    return `<tr>
      <td><span style="color:${COLORS[m]}">●</span> ${SHORT[m]}</td>
      <td><code>${meta ? meta.model : "google/" + m.replace("gemma3_", "gemma-3-") + "-pt"}</code></td>
      <td>${meta ? meta.n_layers : "—"}</td>
      <td>${meta ? meta.n_heads : "—"}</td>
      <td>${steps}</td>
    </tr>`;
  }).join("");
  document.getElementById("model-table").innerHTML = `
    <table class="data">
      <thead><tr><th>Model</th><th>Checkpoint</th><th>Layers</th><th>Q heads</th><th>Steps completed</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  const missing = MODEL_ORDER.filter(
    (m) => !DATA.models[m] || DATA.models[m].steps_present.length === 0
  );
  if (missing.length) {
    document.getElementById("missing-banner").innerHTML = `
      <div class="banner">⚠ ${missing.map((m) => SHORT[m]).join(", ")}:
      scripts are in place but no experiments have been run yet — this model is
      omitted from every plot below.</div>`;
  }
}

/* ------------------------------------------------------------------ */
/* 01 — baseline                                                       */
/* ------------------------------------------------------------------ */

function buildS01() {
  const models = withStep("s01");
  if (!coverageNote("s01", models)) return;

  const templates = ["RR", "NV", "NP_S", "NP_Z", "COORD"];
  Plotly.newPlot(
    "plot-s01-template",
    models.map((m) => ({
      type: "bar",
      name: SHORT[m],
      x: templates,
      y: templates.map((t) => DATA.models[m].s01.per_template[t]),
      marker: { color: COLORS[m] },
    })),
    layout({
      title: { text: "Mean Δ-surprisal at the critical word, by template" },
      barmode: "group",
      yaxis: { title: { text: "Δ-surprisal (nats)" } },
      height: 380,
    }),
    CONFIG
  );

  Plotly.newPlot(
    "plot-s01-pairs",
    models.map((m) => ({
      type: "box",
      name: SHORT[m],
      y: DATA.models[m].s01.pairs.map((p) => p.delta),
      text: DATA.models[m].s01.pairs.map((p) => `#${p.id} ${p.template}`),
      boxpoints: "all",
      jitter: 0.5,
      pointpos: 0,
      marker: { color: COLORS[m], size: 4 },
      line: { color: COLORS[m] },
    })),
    layout({
      title: { text: "Per-pair Δ-surprisal distribution (50 pairs)" },
      yaxis: { title: { text: "Δ-surprisal (nats)" }, zeroline: true },
      showlegend: false,
      height: 380,
    }),
    CONFIG
  );
}

/* ------------------------------------------------------------------ */
/* 02 / 05 — layer sweeps                                              */
/* ------------------------------------------------------------------ */

function layerSweepTraces(stepKey, mainField, mainLabel) {
  const models = withStep(stepKey);
  const traces = [];
  for (const m of models) {
    const d = DATA.models[m];
    const n = d.meta.n_layers;
    const x = d[stepKey].per_layer.map((l) => l.layer / n);
    traces.push({
      type: "scatter",
      mode: "lines",
      name: `${SHORT[m]} ${mainLabel}`,
      legendgroup: m,
      x,
      y: d[stepKey].per_layer.map((l) => l[mainField]),
      line: { color: COLORS[m], width: 2 },
      hovertemplate: `${SHORT[m]} L%{customdata}: %{y:.3f}<extra>${mainLabel}</extra>`,
      customdata: d[stepKey].per_layer.map((l) => l.layer),
    });
    traces.push({
      type: "scatter",
      mode: "lines",
      name: `${SHORT[m]} control`,
      legendgroup: m,
      visible: "legendonly",
      x,
      y: d[stepKey].per_layer.map((l) => l.control_inflation),
      line: { color: COLORS[m], width: 1.5, dash: "dot" },
      hovertemplate: `${SHORT[m]} L%{customdata}: %{y:.3f}<extra>control</extra>`,
      customdata: d[stepKey].per_layer.map((l) => l.layer),
    });
  }
  return { models, traces };
}

function buildS02() {
  const { models, traces } = layerSweepTraces("s02", "effect_drop", "effect drop");
  if (!coverageNote("s02", models)) return;
  Plotly.newPlot(
    "plot-s02",
    traces,
    layout({
      title: { text: "Δ-surprisal effect drop per zero-ablated layer (control traces toggle in legend)" },
      xaxis: { title: { text: "normalised depth (layer / n_layers)" }, range: [0, 1] },
      yaxis: { title: { text: "effect drop (nats)" }, zeroline: true },
      height: 440,
    }),
    CONFIG
  );
}

function buildS05() {
  const { models, traces } = layerSweepTraces("s05", "vc_delta_inflation", "vc_delta inflation");
  if (!coverageNote("s05", models)) return;
  Plotly.newPlot(
    "plot-s05",
    traces,
    layout({
      title: { text: "Comprehension-cost (vc_delta) inflation per zero-ablated layer" },
      xaxis: { title: { text: "normalised depth (layer / n_layers)" }, range: [0, 1] },
      yaxis: { title: { text: "vc_delta inflation (nats)" }, zeroline: true },
      height: 440,
    }),
    CONFIG
  );
}

/* ------------------------------------------------------------------ */
/* 03 — head sweeps                                                    */
/* ------------------------------------------------------------------ */

function buildS03() {
  const models = withStep("s03");
  if (!coverageNote("s03", models)) return;
  const container = document.getElementById("plots-s03");

  for (const m of models) {
    const d = DATA.models[m];
    el("h3", { textContent: `${SHORT[m]} — layers ${d.s03.map((s) => "L" + s.layer).join(", ")}` }, container);
    const div = el("div", { className: "plot" }, container);
    const nHeads = d.meta.n_heads;
    Plotly.newPlot(
      div,
      d.s03.map((sweep, i) => ({
        type: "bar",
        name: `L${sweep.layer}`,
        x: sweep.per_head.map((h) => h.head),
        y: sweep.per_head.map((h) => h.effect_drop),
        marker: { color: COLORS[m], opacity: 0.35 + (0.65 * (i + 1)) / d.s03.length },
      })),
      layout({
        barmode: "group",
        xaxis: { title: { text: "head" }, dtick: 1, range: [-0.5, nHeads - 0.5] },
        yaxis: { title: { text: "effect drop (nats)" }, zeroline: true },
        height: 300,
        margin: { l: 60, r: 20, t: 10, b: 45 },
      }),
      CONFIG
    );
  }
}

/* ------------------------------------------------------------------ */
/* 04 — candidate heads vs comprehension                               */
/* ------------------------------------------------------------------ */

function buildS04() {
  const models = withStep("s04");
  if (!coverageNote("s04", models)) return;
  const container = document.getElementById("plot-s04");
  container.classList.remove("plot");

  for (const m of models) {
    const conds = DATA.models[m].s04.conditions;
    const div = el("div", { className: "plot" }, container);
    Plotly.newPlot(
      div,
      [
        {
          type: "bar",
          x: conds.map((c) => c.label),
          y: conds.map((c) => c.mean_vc_delta),
          marker: {
            color: conds.map((c) => (c.label === "baseline" ? "#888" : COLORS[m])),
          },
        },
      ],
      layout({
        title: { text: `${SHORT[m]} — mean vc_delta under head ablation` },
        yaxis: { title: { text: "vc_delta (nats)" }, zeroline: true },
        showlegend: false,
        height: 320,
      }),
      CONFIG
    );
  }
}

/* ------------------------------------------------------------------ */
/* 06 — head sweeps on the comprehension metric                        */
/* ------------------------------------------------------------------ */

/** step-14 max-of-K p95 threshold for the K heads swept in step 06, or null. */
function vcNullThreshold(m) {
  const d = DATA.models[m];
  if (!d.s14 || !d.s06) return null;
  const k = String(d.s06.length * d.meta.n_heads);
  const bucket = d.s14.max_of_k[k] || d.s14.max_of_k[String(d.meta.n_heads)];
  return bucket ? bucket.p95 : null;
}

function buildS06() {
  const models = withStep("s06");
  if (!coverageNote("s06", models)) return;
  const container = document.getElementById("plots-s06");

  for (const m of models) {
    const d = DATA.models[m];
    el("h3", { textContent: `${SHORT[m]} — layers ${d.s06.map((s) => "L" + s.layer).join(", ")}` }, container);
    const div = el("div", { className: "plot" }, container);
    const nHeads = d.meta.n_heads;
    const thr = vcNullThreshold(m);
    Plotly.newPlot(
      div,
      d.s06.map((sweep, i) => ({
        type: "bar",
        name: `L${sweep.layer}`,
        x: sweep.per_head.map((h) => h.head),
        y: sweep.per_head.map((h) => h.vc_gps_inflation),
        marker: { color: COLORS[m], opacity: 0.35 + (0.65 * (i + 1)) / d.s06.length },
      })),
      layout({
        barmode: "group",
        xaxis: { title: { text: "head" }, dtick: 1, range: [-0.5, nHeads - 0.5] },
        yaxis: { title: { text: "vc_gps inflation (nats)" }, zeroline: true },
        height: 300,
        margin: { l: 60, r: 20, t: 10, b: 45 },
        shapes: thr === null ? [] : [{
          type: "line",
          x0: 0, x1: 1, xref: "paper", y0: thr, y1: thr,
          line: { color: "#555", width: 1.5, dash: "dash" },
        }],
        annotations: thr === null ? [] : [{
          x: 1, xref: "paper", xanchor: "right", y: thr, yanchor: "bottom",
          text: "null max-of-K p95 (step 14)", showarrow: false, font: { size: 11, color: "#555" },
        }],
      }),
      CONFIG
    );
  }
}

/* ------------------------------------------------------------------ */
/* 07/08 — lexical floor                                               */
/* ------------------------------------------------------------------ */

function buildS0708() {
  const models = MODEL_ORDER.filter(
    (m) => DATA.models[m] && (DATA.models[m].s07 || DATA.models[m].s08)
  );
  if (!coverageNote("s0708", models)) return;
  const container = document.getElementById("plot-s0708");
  container.classList.remove("plot");

  const prefixes = [
    ["mean_s_full", "full", 0.9],
    ["mean_s_stripped", "stripped", 0.6],
    ["mean_s_anchor", "anchor", 0.3],
  ];
  const cats = [];
  for (const m of models) {
    if (DATA.models[m].s07) cats.push([m, "s07", `${SHORT[m]} canonical`]);
    if (DATA.models[m].s08) cats.push([m, "s08", `${SHORT[m]} novel`]);
  }

  const barsDiv = el("div", { className: "plot" }, container);
  Plotly.newPlot(
    barsDiv,
    prefixes.map(([field, name, opacity]) => ({
      type: "bar",
      name,
      x: cats.map((c) => c[2]),
      y: cats.map(([m, key]) => DATA.models[m][key].summary[field]),
      marker: { color: cats.map(([m]) => COLORS[m]), opacity },
    })),
    layout({
      title: { text: "Surprisal of the verb continuation by prefix type" },
      barmode: "group",
      yaxis: { title: { text: "surprisal (nats)" } },
      height: 400,
    }),
    CONFIG
  );

  const ciDiv = el("div", { className: "plot" }, container);
  Plotly.newPlot(
    ciDiv,
    [
      {
        type: "scatter",
        mode: "markers",
        x: cats.map((c) => c[2]),
        y: cats.map(([m, key]) => {
          const s = DATA.models[m][key].summary;
          return s.mean_s_full - s.mean_s_stripped;
        }),
        error_y: {
          type: "data",
          symmetric: false,
          array: cats.map(([m, key]) => {
            const d = DATA.models[m][key];
            const mean = d.summary.mean_s_full - d.summary.mean_s_stripped;
            return d.ci95_full_minus_stripped[1] - mean;
          }),
          arrayminus: cats.map(([m, key]) => {
            const d = DATA.models[m][key];
            const mean = d.summary.mean_s_full - d.summary.mean_s_stripped;
            return mean - d.ci95_full_minus_stripped[0];
          }),
        },
        marker: { size: 9, color: cats.map(([m]) => COLORS[m]) },
      },
    ],
    layout({
      title: { text: "full − stripped with bootstrap 95% CI (memorisation would predict ≪ 0 on canonical only)" },
      yaxis: { title: { text: "full − stripped (nats)" }, zeroline: true },
      showlegend: false,
      height: 340,
    }),
    CONFIG
  );
}

/* ------------------------------------------------------------------ */
/* 09 — vc_delta with CIs                                              */
/* ------------------------------------------------------------------ */

function buildS09() {
  const models = withStep("s09");
  if (!coverageNote("s09", models)) return;

  const traces = ["canonical", "novel"].map((cond, i) => ({
    type: "scatter",
    mode: "markers",
    name: cond,
    x: models.map((m) => SHORT[m]),
    y: models.map((m) => DATA.models[m].s09[cond].summary.mean_vc_delta),
    error_y: {
      type: "data",
      symmetric: false,
      array: models.map((m) => {
        const d = DATA.models[m].s09[cond];
        return d.ci95.vc_delta[1] - d.summary.mean_vc_delta;
      }),
      arrayminus: models.map((m) => {
        const d = DATA.models[m].s09[cond];
        return d.summary.mean_vc_delta - d.ci95.vc_delta[0];
      }),
    },
    marker: { size: 10, symbol: i === 0 ? "circle" : "diamond" },
  }));

  Plotly.newPlot(
    "plot-s09",
    traces,
    layout({
      title: { text: "Residual comprehension cost (vc_delta) with bootstrap 95% CI" },
      yaxis: { title: { text: "mean vc_delta (nats)" }, zeroline: true },
      height: 380,
    }),
    CONFIG
  );
}

/* ------------------------------------------------------------------ */
/* 10 — attention patterns                                             */
/* ------------------------------------------------------------------ */

function buildS10() {
  const models = withStep("s10");
  if (!coverageNote("s10", models)) return;
  const container = document.getElementById("plots-s10");

  for (const m of models) {
    const d = DATA.models[m];
    el("h3", { textContent: `${SHORT[m]} — attention(disambiguator → ambiguous word), GPS − normal` }, container);
    const div = el("div", { className: "plot" }, container);
    Plotly.newPlot(
      div,
      [
        {
          type: "heatmap",
          z: d.s10.mean_attn_diff,
          x: d.s10.mean_attn_diff[0].map((_, h) => h),
          y: d.s10.mean_attn_diff.map((_, l) => l),
          colorscale: "RdBu",
          reversescale: true,
          zmid: 0,
          colorbar: { title: { text: "Δ attn" } },
          hovertemplate: "L%{y}.H%{x}: %{z:.4f}<extra></extra>",
        },
      ],
      layout({
        xaxis: { title: { text: "head" }, dtick: d.meta.n_heads > 8 ? 2 : 1 },
        yaxis: { title: { text: "layer" }, autorange: "reversed" },
        height: Math.max(320, d.meta.n_layers * 9),
        margin: { l: 60, r: 20, t: 10, b: 45 },
      }),
      CONFIG
    );

    const rows = d.s10.top_heads_by_diff
      .map(
        (h) => `<tr>
          <td>L${h.layer}.H${h.head}</td>
          <td>${h.mean_attn_gps.toFixed(3)}</td>
          <td>${h.mean_attn_diff.toFixed(3)}</td>
          <td>${h.spearman_vc_delta.toFixed(3)}</td>
          <td>${h.spearman_p.toFixed(3)}</td>
        </tr>`
      )
      .join("");
    el(
      "div",
      {
        innerHTML: `<table class="data">
          <thead><tr><th>head</th><th>attn (GPS)</th><th>attn diff (GPS−normal)</th>
          <th>Spearman vs vc_delta</th><th>p</th></tr></thead>
          <tbody>${rows}</tbody></table>`,
      },
      container
    );
  }
}

/* ------------------------------------------------------------------ */
/* 11 — activation patching                                            */
/* ------------------------------------------------------------------ */

function buildS11() {
  const models = withStep("s11");
  const ok = coverageNote("s11", models);

  const ran = MODEL_ORDER.filter(
    (m) => DATA.models[m] && DATA.models[m].steps_present.length > 0
  );
  const skipped = ran.filter((m) => !models.includes(m));
  if (skipped.length) {
    document.getElementById("note-s11").textContent =
      skipped.map((m) => SHORT[m]).join(", ") +
      ": step 11 was started but did not complete (no JSON output), so it is missing here.";
  }
  if (!ok) return;

  const variants = [
    ["recovery_v_amb", "v_amb", "solid"],
    ["recovery_critical", "critical", "dot"],
    ["recovery_both", "both", "dash"],
  ];
  const traces = [];
  for (const m of models) {
    const d = DATA.models[m];
    const n = d.meta.n_layers;
    for (const [field, name, dash] of variants) {
      traces.push({
        type: "scatter",
        mode: "lines",
        name: `${SHORT[m]} ${name}`,
        legendgroup: m,
        x: d.s11.per_layer.map((l) => l.layer / n),
        y: d.s11.per_layer.map((l) => l[field]),
        line: { color: COLORS[m], dash, width: 2 },
        customdata: d.s11.per_layer.map((l) => l.layer),
        hovertemplate: `${SHORT[m]} L%{customdata}: %{y:.3f}<extra>${name}</extra>`,
      });
    }
  }

  Plotly.newPlot(
    "plot-s11",
    traces,
    layout({
      title: { text: "Recovery of the comprehension gap when patching NORMAL activations into the GPS run" },
      xaxis: { title: { text: "normalised depth (layer / n_layers)" }, range: [0, 1] },
      yaxis: { title: { text: "recovery (1 = fully restored)" }, zeroline: true },
      height: 460,
    }),
    CONFIG
  );
}

/* ------------------------------------------------------------------ */
/* 12 — zero vs mean ablation                                          */
/* ------------------------------------------------------------------ */

function buildS12() {
  const models = withStep("s12");
  if (!coverageNote("s12", models)) return;

  const entries = [];
  for (const m of models) {
    const d = DATA.models[m].s12;
    for (const e of [...d.head_sets, ...d.layers]) {
      entries.push({ model: m, label: `${SHORT[m]} ${e.label}`, zero: e.zero, mean: e.mean });
    }
  }

  const mkPlot = (divId, field, title, ytitle) =>
    Plotly.newPlot(
      divId,
      [
        {
          type: "bar",
          name: "zero ablation",
          x: entries.map((e) => e.label),
          y: entries.map((e) => e.zero[field]),
          marker: { color: entries.map((e) => COLORS[e.model]), opacity: 0.9 },
        },
        {
          type: "bar",
          name: "mean ablation",
          x: entries.map((e) => e.label),
          y: entries.map((e) => e.mean[field]),
          marker: { color: entries.map((e) => COLORS[e.model]), opacity: 0.4 },
        },
      ],
      layout({
        title: { text: title },
        barmode: "group",
        yaxis: { title: { text: ytitle }, zeroline: true },
        xaxis: { tickangle: -30 },
        height: 420,
        margin: { l: 60, r: 20, t: 40, b: 120 },
      }),
      CONFIG
    );

  mkPlot(
    "plot-s12-effect",
    "effect_drop",
    "Δ-surprisal effect drop: zero vs mean ablation",
    "effect drop (nats)"
  );
  mkPlot(
    "plot-s12-vc",
    "vc_gps_inflation",
    "GPS comprehension inflation: zero vs mean ablation",
    "vc_gps inflation (nats)"
  );
}

/* ------------------------------------------------------------------ */
/* 13 — null distribution                                              */
/* ------------------------------------------------------------------ */

function buildS13() {
  const models = withStep("s13");
  if (!coverageNote("s13", models)) return;
  const container = document.getElementById("plots-s13");

  for (const m of models) {
    const d = DATA.models[m].s13;
    const p = d.percentiles.nv_effect_drop;
    el("h3", { textContent: `${SHORT[m]} — ${d.n_samples} random single-head ablations` }, container);
    const div = el("div", { className: "plot" }, container);

    const vline = (x, color, dash) => ({
      type: "line",
      x0: x, x1: x, y0: 0, y1: 1, yref: "paper",
      line: { color, width: 1.5, dash },
    });

    Plotly.newPlot(
      div,
      [
        {
          type: "histogram",
          x: d.samples.map((s) => s.nv_effect_drop),
          nbinsx: 30,
          marker: { color: COLORS[m], opacity: 0.75 },
        },
      ],
      layout({
        xaxis: { title: { text: "effect drop (nats)" } },
        yaxis: { title: { text: "count" } },
        showlegend: false,
        height: 300,
        margin: { l: 60, r: 20, t: 30, b: 45 },
        shapes: [
          vline(p.p5, "#555", "dash"),
          vline(p.p95, "#555", "dash"),
          vline(-p.abs_max, "#c33", "dot"),
          vline(p.abs_max, "#c33", "dot"),
        ],
        annotations: [
          { x: p.p5, y: 1.04, yref: "paper", text: "p5", showarrow: false, font: { size: 11 } },
          { x: p.p95, y: 1.04, yref: "paper", text: "p95", showarrow: false, font: { size: 11 } },
          { x: p.abs_max, y: 1.04, yref: "paper", text: "|max|", showarrow: false, font: { size: 11, color: "#c33" } },
        ],
      }),
      CONFIG
    );
  }
}

/* ------------------------------------------------------------------ */
/* 14 — null distribution, comprehension metric                        */
/* ------------------------------------------------------------------ */

function buildS14() {
  const models = withStep("s14");
  if (!coverageNote("s14", models)) return;
  const container = document.getElementById("plots-s14");

  for (const m of models) {
    const d = DATA.models[m].s14;
    const p = d.percentiles.vc_gps_inflation;
    const thr = vcNullThreshold(m);
    const best = DATA.models[m].s06
      ? Math.max(...DATA.models[m].s06.flatMap((s) => s.per_head.map((h) => h.vc_gps_inflation)))
      : null;
    el("h3", { textContent: `${SHORT[m]} — ${d.n_samples} random single-head ablations (vc metric)` }, container);
    const div = el("div", { className: "plot" }, container);

    const vline = (x, color, dash) => ({
      type: "line",
      x0: x, x1: x, y0: 0, y1: 1, yref: "paper",
      line: { color, width: 1.5, dash },
    });
    const shapes = [vline(p.p5, "#555", "dash"), vline(p.p95, "#555", "dash")];
    const annotations = [
      { x: p.p5, y: 1.04, yref: "paper", text: "p5", showarrow: false, font: { size: 11 } },
      { x: p.p95, y: 1.04, yref: "paper", text: "p95", showarrow: false, font: { size: 11 } },
    ];
    if (thr !== null) {
      shapes.push(vline(thr, "#555", "dot"));
      annotations.push({ x: thr, y: 1.12, yref: "paper", text: "max-of-K p95", showarrow: false, font: { size: 11 } });
    }
    if (best !== null) {
      shapes.push(vline(best, COLORS[m], "solid"));
      annotations.push({
        x: best, y: 1.04, yref: "paper", text: "best head (step 06)",
        showarrow: false, font: { size: 11, color: COLORS[m] },
      });
    }

    Plotly.newPlot(
      div,
      [
        {
          type: "histogram",
          x: d.vc_gps_inflations,
          nbinsx: 40,
          marker: { color: COLORS[m], opacity: 0.75 },
        },
      ],
      layout({
        xaxis: { title: { text: "vc_gps inflation (nats)" } },
        yaxis: { title: { text: "count" } },
        showlegend: false,
        height: 300,
        margin: { l: 60, r: 20, t: 40, b: 45 },
        shapes,
        annotations,
      }),
      CONFIG
    );
  }
}

/* ------------------------------------------------------------------ */

buildOverview();
buildS01();
buildS02();
buildS03();
buildS04();
buildS05();
buildS06();
buildS0708();
buildS09();
buildS10();
buildS11();
buildS12();
buildS13();
buildS14();
