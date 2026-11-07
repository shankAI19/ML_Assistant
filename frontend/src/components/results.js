import { artifactUrl, codeUrl } from "../api.js";
import { element } from "../state.js";

let plotly;
let renderGeneration = 0;
const loadPlotly = () =>
  (plotly ??= import("plotly.js-dist-min").then((module) => module.default));

function format(value) {
  return value == null
    ? "Unavailable"
    : Number.isInteger(value)
      ? String(value)
      : Number(value).toFixed(3);
}

export async function renderResults(runs) {
  const generation = ++renderGeneration;
  const panel = document.querySelector("#results");
  document.querySelector("#run-count").textContent =
    `${runs.length} run${runs.length === 1 ? "" : "s"}`;
  if (!runs.length) return;
  const Plotly = await loadPlotly();
  if (generation !== renderGeneration) return;
  panel.querySelectorAll(".chart").forEach((chart) => Plotly.purge(chart));
  panel.replaceChildren();
  for (const run of [...runs].reverse()) {
    if (generation !== renderGeneration) return;
    const card = element("article", "result-card");
    const partition = run.result.evaluation_partition;
    card.append(
      element(
        "div",
        "eyebrow",
        `${run.task.toUpperCase()}${partition ? ` / ${partition.toUpperCase()}` : ""}`,
      ),
      element("h3", "", run.explanation),
    );
    const metrics =
      run.result.metrics ||
      run.result.validation_metrics ||
      run.result.test_metrics;
    if (metrics && typeof metrics === "object") {
      const grid = element("div", "metrics");
      const metricOrder = [
        ...new Set([
          ...(run.result.plan?.focus_metrics || []),
          "average_precision",
          "precision",
          "recall",
          "f1",
          "roc_auc",
          "specificity",
          "accuracy",
          "balanced_accuracy",
          "alert_rate",
        ]),
      ];
      for (const key of metricOrder) {
        const tile = element("div", "metric");
        if (key === "alert_rate")
          tile.title =
            "Fraction of rows predicted positive at the selected threshold; this is not a measure of correctness.";
        tile.append(
          element("span", "", key.replaceAll("_", " ")),
          element("strong", "", format(metrics[key])),
        );
        grid.append(tile);
      }
      card.append(
        grid,
        element(
          "p",
          "small muted",
          `Evaluated rows: ${metrics.rows} · positives: ${metrics.positives} · threshold: ${metrics.threshold}`,
        ),
      );
    }
    if (run.task === "profile") {
      card.append(
        element(
          "p",
          "muted",
          `${run.result.rows.toLocaleString()} rows · ${run.result.columns.length} columns · ${run.result.duplicates} exact duplicate rows`,
        ),
      );
      const scroll = element("div", "table-scroll");
      const table = element("table");
      const header = element("tr");
      for (const name of ["Column", "Type", "Missing", "Unique", "Notes"])
        header.append(element("th", "", name));
      table.append(header);
      for (const col of run.result.columns) {
        const row = element("tr");
        for (const value of [
          col.name,
          col.dtype,
          col.missing,
          col.unique,
          [
            col.binary_candidate ? `Binary: ${col.values.join(", ")}` : "",
            col.possible_id ? "Possible ID" : "",
          ]
            .filter(Boolean)
            .join(" · "),
        ])
          row.append(element("td", "", String(value)));
        table.append(row);
      }
      scroll.append(table);
      card.append(scroll);
    }
    if (run.result.note)
      card.append(element("p", "muted small", run.result.note));
    for (const warning of run.result.warnings || [])
      card.append(element("p", "small warning", warning));
    const links = element("div", "downloads");
    for (const filename of run.artifacts.filter((name) =>
      /\.(docx|md|csv|joblib)$/.test(name),
    )) {
      const link = element("a", "button secondary small", `↓ ${filename}`);
      link.href = artifactUrl(run.id, filename);
      links.append(link);
    }
    const code = element("a", "text-button", "Download executed code ↗");
    code.href = codeUrl(run.id);
    links.append(code);
    card.append(links);
    const receipt = element("details", "receipt");
    receipt.append(
      element(
        "summary",
        "",
        `${run.receipt.elapsed_seconds}s · network disabled · ${run.receipt.memory_mb} MB cap · sandbox destroyed`,
      ),
      element("pre", "code", JSON.stringify(run.receipt, null, 2)),
    );
    card.append(receipt);
    panel.append(card);
    for (const filename of run.result.charts || []) {
      const chart = element("div", "chart");
      card.append(chart);
      try {
        const response = await fetch(artifactUrl(run.id, filename), {
          credentials: "same-origin",
          cache: "no-store",
        });
        if (!response.ok) throw new Error("Chart unavailable");
        const figure = await response.json();
        if (!chart.isConnected) break;
        await Plotly.react(
          chart,
          figure.data,
          {
            ...figure.layout,
            autosize: true,
            height: 400,
            margin: { l: 65, r: 20, t: 55, b: 65 },
            font: { family: "system-ui", color: "#314745", size: 14 },
            paper_bgcolor: "#ffffff",
            plot_bgcolor: "#ffffff",
            colorway: ["#14685e", "#bc9c65", "#738fa2", "#b57066"],
            legend: { orientation: "h", y: -0.28 },
          },
          { responsive: true, displaylogo: false },
        );
      } catch {
        if (chart.isConnected)
          chart.textContent =
            "Chart unavailable. The session may have expired.";
      }
    }
  }
}

export function clearResults() {
  renderGeneration += 1;
  const panel = document.querySelector("#results");
  const charts = [...panel.querySelectorAll(".chart")];
  if (plotly)
    plotly.then((Plotly) => charts.forEach((chart) => Plotly.purge(chart)));
  panel.replaceChildren();
  document.querySelector("#run-count").textContent = "0 runs";
}
