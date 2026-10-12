import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

function formatPct(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function MetricCard({ label, value, detail }) {
  return (
    <article className="metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </article>
  );
}

function App() {
  const [summary, setSummary] = useState(null);
  const [sample, setSample] = useState([]);
  const [selected, setSelected] = useState(0);

  useEffect(() => {
    Promise.all([
      fetch("/model-summary.json").then((response) => response.json()),
      fetch("/holdout-sample.json").then((response) => response.json()),
    ]).then(([modelSummary, holdoutSample]) => {
      setSummary(modelSummary);
      setSample(holdoutSample);
    });
  }, []);

  const selectedRow = sample[selected] ?? {};
  const logistic = useMemo(
    () => summary?.metrics?.find((row) => row.model.includes("logistic")),
    [summary]
  );

  if (!summary) {
    return <main className="loading">Loading model bench…</main>;
  }

  return (
    <main>
      <section className="hero">
        <div className="hero-copy">
          <p className="kicker">Business classifier benchmark</p>
          <h1>Attrition risk, audited like a model room.</h1>
          <p className="lede">
            A portfolio build that compares a parsimonious classical classifier with Jev and Laya
            typed-decision adapters on the same frozen employee holdout.
          </p>
          <div className="hero-metrics">
            <MetricCard label="Rows" value={summary.rows} detail="IBM HR Attrition" />
            <MetricCard
              label="Holdout PR-AUC"
              value={logistic?.pr_auc?.toFixed(3)}
              detail="classical model"
            />
            <MetricCard
              label="CV PR-AUC"
              value={summary.cv.mean.toFixed(3)}
              detail={`${summary.cv.folds} folds`}
            />
          </div>
        </div>

        <aside className="decision-card" aria-label="Current employee score">
          <span>Frozen holdout row</span>
          <strong>{formatPct(selectedRow.classical_score ?? 0)}</strong>
          <p>Classical attrition risk score for the selected employee profile.</p>
          <input
            type="range"
            min="0"
            max={Math.max(sample.length - 1, 0)}
            value={selected}
            onChange={(event) => setSelected(Number(event.target.value))}
            aria-label="Select holdout employee"
          />
        </aside>
      </section>

      <section className="bench">
        {summary.metrics.map((item) => (
          <article key={item.model}>
            <h2>{item.model}</h2>
            <p className="metric">{item.pr_auc.toFixed(3)}</p>
            <p>PR-AUC · ROC-AUC {item.roc_auc.toFixed(3)} · F1 {item.f1.toFixed(3)}</p>
          </article>
        ))}
        {summary.typed_decision_models.map((item) => (
          <article key={item.model} className="pending">
            <h2>{item.model}</h2>
            <p className="metric">Ready</p>
            <p>{item.status.replaceAll("_", " ")}</p>
          </article>
        ))}
      </section>

      <section className="split">
        <article>
          <h2>What the classical model learned</h2>
          <div className="bars">
            {summary.top_coefficients.slice(0, 8).map((row) => (
              <div className="bar-row" key={row.feature}>
                <span>{row.feature.replace("cat__", "").replace("num__", "")}</span>
                <div>
                  <i style={{ width: `${Math.min(row.abs_coefficient * 34, 100)}%` }} />
                </div>
                <b>{row.coefficient.toFixed(2)}</b>
              </div>
            ))}
          </div>
        </article>

        <article>
          <h2>EDA controls</h2>
          <ul className="checks">
            <li>Attrition rate: {formatPct(summary.eda.positive_rate)}</li>
            <li>PCA components for 80% variance: {summary.pca.components_for_80pct_variance}</li>
            <li>Train/test split: {summary.split.train}/{summary.split.test}</li>
            <li>Threshold rule: {summary.threshold.rule}</li>
          </ul>
        </article>
      </section>

      <section className="state">
        <h2>The exact state sent to typed-decision models</h2>
        <pre>{selectedRow.typed_state}</pre>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
