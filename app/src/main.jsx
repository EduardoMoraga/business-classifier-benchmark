import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const NAV = [
  { id: "command", label: "Sala de control", en: "Command center" },
  { id: "simulator", label: "Simulador", en: "What-if lab" },
  { id: "decision", label: "Decisión", en: "Business threshold" },
  { id: "benchmark", label: "Comparativa", en: "Benchmark arena" },
  { id: "evidence", label: "Trazabilidad", en: "Evidence trail" },
];

const MODEL_COPY = {
  classical: {
    title: "ML clásico",
    en: "Classical ML",
    subtitle: "Regresión logística parsimoniosa",
    status: "ejecutado · live",
    voice: "Holdout congelado, regularización L1, VIF, PCA y odds ratios. La vieja escuela, pero auditable.",
  },
  jev: {
    title: "Jev / TypeSafe",
    en: "System One hosted API",
    subtitle: "Preguntas tipadas sobre el mismo estado",
    status: "no ejecutado · adapter ready",
    voice: "Recibe exactamente el mismo estado del empleado. Queda pendiente correrlo con credenciales frescas y registrar costo/latencia.",
  },
  laya: {
    title: "Laya",
    en: "Open decision model",
    subtitle: "Typed decisions local/API",
    status: "no ejecutado · adapter ready",
    voice: "Mismo holdout, misma pregunta, salida probabilística. Puede correr local o por API según checkpoint.",
  },
};

const FEATURE_MAP = {
  overtimeYes: "cat__OverTime_Yes",
  overtimeNo: "cat__OverTime_No",
  jobSatisfaction: "num__JobSatisfaction",
  environmentSatisfaction: "num__EnvironmentSatisfaction",
  distance: "num__DistanceFromHome",
  promotion: "num__YearsSinceLastPromotion",
};

function pct(value, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}
function num(value, digits = 3) {
  return Number(value || 0).toFixed(digits);
}
function cleanFeature(feature) {
  return feature.replace("cat__", "").replace("num__", "").replaceAll("_", " · ");
}
function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}
function toLogit(p) {
  const safe = clamp(p, 0.001, 0.999);
  return Math.log(safe / (1 - safe));
}
function fromLogit(x) {
  return 1 / (1 + Math.exp(-x));
}
function riskBand(score) {
  if (score >= 0.66) return { label: "riesgo alto", en: "high risk", tone: "hot" };
  if (score >= 0.38) return { label: "observar", en: "watch", tone: "warm" };
  return { label: "estable", en: "stable", tone: "cool" };
}
function coefficient(summary, key) {
  const feature = FEATURE_MAP[key];
  return summary.top_coefficients.find((row) => row.feature === feature)?.coefficient ?? 0;
}

function useData() {
  const [state, setState] = useState({ summary: null, sample: [], error: null });
  useEffect(() => {
    const base = import.meta.env.BASE_URL || "/";
    Promise.all([
      fetch(`${base}model-summary.json`).then((r) => {
        if (!r.ok) throw new Error(`model-summary ${r.status}`);
        return r.json();
      }),
      fetch(`${base}holdout-sample.json`).then((r) => {
        if (!r.ok) throw new Error(`holdout-sample ${r.status}`);
        return r.json();
      }),
    ])
      .then(([summary, sample]) => setState({ summary, sample, error: null }))
      .catch((error) => setState({ summary: null, sample: [], error }));
  }, []);
  return state;
}

function TopNav({ active, setActive }) {
  function select(id) {
    setActive(id);
    window.location.hash = id;
  }
  return (
    <header className="topbar">
      <a className="brand" href="https://eduardomoraga.github.io/" target="_blank" rel="noreferrer">
        <span>Eduardo Moraga</span>
        <b>Employee Attrition Lab</b>
      </a>
      <nav aria-label="Secciones del proyecto" role="tablist">
        {NAV.map((item) => (
          <button
            key={item.id}
            role="tab"
            aria-selected={active === item.id}
            className={active === item.id ? "active" : ""}
            onClick={() => select(item.id)}
          >
            {item.label}
            <small>{item.en}</small>
          </button>
        ))}
      </nav>
    </header>
  );
}

function Hero({ summary, setActive }) {
  const logistic = summary.metrics.find((row) => row.model.includes("logistic"));
  const dummy = summary.metrics[0];
  const lift = logistic.pr_auc / dummy.pr_auc - 1;
  return (
    <section className="hero-grid">
      <div className="hero-copy">
        <p className="section-code">demo viva · ciencia de datos · decisiones tipadas</p>
        <h1>No te lo cuento: el clasificador queda abierto.</h1>
        <p className="lede">
          Un laboratorio de attrition para comparar ciencia de datos clásica contra Jev y Laya: el
          mismo holdout, la misma pregunta de negocio y una ruta trazable desde el dato crudo hasta
          la decisión.
        </p>
        <div className="hero-actions">
          <button onClick={() => setActive("simulator")}>Probar empleado</button>
          <button className="ghost" onClick={() => setActive("decision")}>Mover umbral de negocio</button>
        </div>
      </div>

      <aside className="hero-console">
        <div className="console-head">
          <span className="dot" />
          <span>edu · model-room LIVE · {summary.run?.id ?? "local-run"}</span>
        </div>
        <div className="console-metrics">
          <Metric label="PR-AUC holdout" value={num(logistic.pr_auc)} delta={`+${pct(lift, 0)} vs dummy`} />
          <Metric label="ROC-AUC" value={num(logistic.roc_auc)} delta="holdout congelado" />
          <Metric label="CV PR-AUC" value={num(summary.cv.mean)} delta={`${summary.cv.folds} folds`} />
          <Metric label="Attrition base" value={pct(summary.eda.positive_rate)} delta="clase minoritaria" />
        </div>
        <div className="pipeline-strip" aria-label="Pipeline del modelo">
          {["raw HR data", "EDA", "VIF / PCA", "L1 model", "Jev / Laya", "decisión"].map((step) => (
            <span key={step}>{step}</span>
          ))}
        </div>
      </aside>
    </section>
  );
}

function Metric({ label, value, delta }) {
  return (
    <article className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{delta}</small>
    </article>
  );
}

function CommandCenter({ summary, sample, selected, setSelected, setActive }) {
  const row = sample[selected] ?? sample[0];
  const band = riskBand(row.classical_score);
  const logistic = summary.metrics.find((item) => item.model.includes("logistic"));
  const events = [
    `run ${summary.run?.id ?? "local"} · hash ${summary.run?.dataset_hash ?? "n/a"}`,
    `split congelado: ${summary.split.train}/${summary.split.test}`,
    `VIF crítico: ${summary.eda.highest_vif[0].feature} (${num(summary.eda.highest_vif[0].vif, 1)})`,
    `PCA: ${summary.pca.components_for_80pct_variance} componentes explican 80%`,
    "Jev/Laya: no ejecutados todavía; adaptadores listos para benchmark honesto",
  ];

  return (
    <section className="panel command-layout" role="tabpanel">
      <div className="section-title">
        <p>Sala de control · Command center</p>
        <h2>Un caso de RR.HH. tratado como producto de decisión.</h2>
      </div>
      <div className="live-card risk-card">
        <p>Empleado seleccionado · holdout</p>
        <strong className={band.tone}>{pct(row.classical_score)}</strong>
        <span className={`pill ${band.tone}`}>{band.label} · {band.en}</span>
        <input type="range" min="0" max={sample.length - 1} value={selected} onChange={(e) => setSelected(Number(e.target.value))} />
        <button onClick={() => setActive("simulator")}>Abrir simulador</button>
      </div>
      <div className="live-card map-card">
        <p>Mapa de modelos</p>
        <ModelRow name="Dummy prior" score={summary.metrics[0].pr_auc} status="baseline" />
        <ModelRow name="ML clásico" score={logistic.pr_auc} status="live" />
        <ModelRow name="Jev" score={null} status="no ejecutado" />
        <ModelRow name="Laya" score={null} status="no ejecutado" />
      </div>
      <div className="live-card stream-card">
        <p>Event stream</p>
        {events.map((event, index) => <span key={event}>0{index + 1} · {event}</span>)}
      </div>
    </section>
  );
}

function ModelRow({ name, score, status }) {
  return (
    <div className={`model-row ${score ? "" : "pending-row"}`}>
      <span>{name}</span>
      <div>{score ? <i style={{ width: `${score * 100}%` }} /> : <em />}</div>
      <b>{score ? num(score) : status}</b>
    </div>
  );
}

function Simulator({ sample, summary, selected, setSelected }) {
  const base = sample[selected] ?? sample[0];
  const [scenario, setScenario] = useState({ overtime: "base", jobSatisfaction: 0, environment: 0, promotion: 0 });
  const contributions = useMemo(() => {
    const parts = [];
    if (scenario.overtime !== "base" && scenario.overtime !== base.OverTime) {
      const next = scenario.overtime === "Yes" ? coefficient(summary, "overtimeYes") : coefficient(summary, "overtimeNo");
      const current = base.OverTime === "Yes" ? coefficient(summary, "overtimeYes") : coefficient(summary, "overtimeNo");
      parts.push({ label: `OverTime ${base.OverTime} → ${scenario.overtime}`, value: next - current });
    }
    if (scenario.jobSatisfaction) parts.push({ label: `Job satisfaction +${scenario.jobSatisfaction}`, value: coefficient(summary, "jobSatisfaction") * scenario.jobSatisfaction });
    if (scenario.environment) parts.push({ label: `Environment +${scenario.environment}`, value: coefficient(summary, "environmentSatisfaction") * scenario.environment });
    if (scenario.promotion) parts.push({ label: `Years since promotion +${scenario.promotion}`, value: coefficient(summary, "promotion") * scenario.promotion });
    return parts;
  }, [base, scenario, summary]);
  const logit = toLogit(base.classical_score) + contributions.reduce((acc, item) => acc + item.value, 0);
  const simulated = fromLogit(logit);
  const band = riskBand(simulated);

  return (
    <section className="panel simulator-grid" role="tabpanel">
      <div className="section-title wide">
        <p>Simulador · What-if lab</p>
        <h2>La predicción se toca. El supuesto queda visible.</h2>
      </div>
      <article className="employee-card">
        <div className="employee-head"><span>employee #{selected + 1}</span><b>{base.JobRole}</b></div>
        <select value={selected} onChange={(e) => setSelected(Number(e.target.value))}>
          {sample.map((row, index) => <option key={`${row.Age}-${index}`} value={index}>{index + 1}. {row.JobRole} · {row.Department} · {pct(row.classical_score)}</option>)}
        </select>
        <dl>
          <div><dt>Edad</dt><dd>{base.Age}</dd></div><div><dt>Área</dt><dd>{base.Department}</dd></div>
          <div><dt>OverTime</dt><dd>{base.OverTime}</dd></div><div><dt>Satisfacción</dt><dd>{base.JobSatisfaction}/4</dd></div>
          <div><dt>Distancia</dt><dd>{base.DistanceFromHome} km</dd></div><div><dt>Años compañía</dt><dd>{base.YearsAtCompany}</dd></div>
        </dl>
      </article>
      <article className="controls-card">
        <Control label="OverTime" hint="Usa diferencia real entre coeficientes OverTime_Yes y OverTime_No.">
          <div className="segmented">{["base", "Yes", "No"].map((v) => <button key={v} className={scenario.overtime === v ? "active" : ""} onClick={() => setScenario((s) => ({ ...s, overtime: v }))}>{v}</button>)}</div>
        </Control>
        <Control label="Subir job satisfaction" hint="Coeficiente real aplicado sobre el logit como aproximación transparente."><input type="range" min="0" max="3" value={scenario.jobSatisfaction} onChange={(e) => setScenario((s) => ({ ...s, jobSatisfaction: Number(e.target.value) }))} /></Control>
        <Control label="Subir environment satisfaction" hint="Simula intervención sobre clima/equipo."><input type="range" min="0" max="3" value={scenario.environment} onChange={(e) => setScenario((s) => ({ ...s, environment: Number(e.target.value) }))} /></Control>
        <Control label="Más años sin promoción" hint="Coeficiente positivo: aumenta riesgo estimado."><input type="range" min="0" max="5" value={scenario.promotion} onChange={(e) => setScenario((s) => ({ ...s, promotion: Number(e.target.value) }))} /></Control>
      </article>
      <article className="result-card">
        <p>riesgo simulado</p>
        <strong className={band.tone}>{pct(simulated)}</strong>
        <span className={`pill ${band.tone}`}>{band.label} · {pct(simulated - base.classical_score)}</span>
        <div className="formula"><b>logit nuevo = logit base + Σ coeficientes</b>{contributions.length ? contributions.map((c) => <small key={c.label}>{c.label}: {c.value > 0 ? "+" : ""}{num(c.value, 2)}</small>) : <small>Sin intervención: score original del holdout.</small>}</div>
      </article>
    </section>
  );
}

function Control({ label, hint, children }) {
  return <label className="control"><span>{label}</span>{children}<small>{hint}</small></label>;
}

function Decision({ summary }) {
  const [fnCost, setFnCost] = useState(10000);
  const [fpCost, setFpCost] = useState(1000);
  const [[tn, fp], [fn, tp]] = summary.confusion_matrix;
  const expected = fn * fnCost + fp * fpCost;
  const caught = tp / (tp + fn);
  const workload = (tp + fp) / summary.split.test;
  return (
    <section className="panel decision-grid" role="tabpanel">
      <div className="section-title wide"><p>Decisión · Business threshold</p><h2>El modelo importa cuando cambia una decisión de negocio.</h2></div>
      <article className="matrix-card"><h3>Matriz de confusión · threshold {num(summary.threshold.selected)}</h3><div className="confusion"><span>TN<b>{tn}</b></span><span>FP<b>{fp}</b></span><span>FN<b>{fn}</b></span><span>TP<b>{tp}</b></span></div></article>
      <article className="cost-card"><h3>Costo editable</h3><label>Falso negativo: ${fnCost.toLocaleString()}<input type="range" min="1000" max="50000" step="1000" value={fnCost} onChange={(e) => setFnCost(Number(e.target.value))} /></label><label>Falso positivo: ${fpCost.toLocaleString()}<input type="range" min="100" max="10000" step="100" value={fpCost} onChange={(e) => setFpCost(Number(e.target.value))} /></label></article>
      <article className="recommendation"><p>costo esperado</p><strong>${expected.toLocaleString()}</strong><span>Captura {pct(caught)} de las salidas con una carga de intervención de {pct(workload)} del holdout.</span><b>Recomendación: automatizar casos estables, revisar humanos en zona amarilla, accionar retención en riesgo alto.</b></article>
    </section>
  );
}

function Benchmark({ summary }) {
  const logistic = summary.metrics.find((item) => item.model.includes("logistic"));
  const rows = [
    { key: "classical", pr: logistic.pr_auc, roc: logistic.roc_auc, f1: logistic.f1, cost: "local", latency: "ms", reproducibility: "alta" },
    { key: "jev", pr: null, roc: null, f1: null, cost: "API", latency: "red", reproducibility: "logs + run id" },
    { key: "laya", pr: null, roc: null, f1: null, cost: "local/API", latency: "ms/API", reproducibility: "checkpoint" },
  ];
  return (
    <section className="panel benchmark-panel" role="tabpanel">
      <div className="section-title wide"><p>Comparativa · Benchmark arena</p><h2>La comparación no se dibuja: se corre. Lo pendiente queda marcado.</h2></div>
      <div className="model-cards">{rows.map((row) => <article key={row.key} className="bench-card"><span>{MODEL_COPY[row.key].status}</span><h3>{MODEL_COPY[row.key].title}</h3><small>{MODEL_COPY[row.key].en}</small><p>{MODEL_COPY[row.key].voice}</p><div className="score-line"><b>{row.pr ? num(row.pr) : "not run"}</b><small>PR-AUC</small></div></article>)}</div>
      <div className="comparison-table" role="table" aria-label="Comparación de modelos"><div role="row" className="thead"><span>Modelo</span><span>PR-AUC</span><span>ROC</span><span>F1</span><span>Costo</span><span>Latencia</span><span>Reproducibilidad</span></div>{rows.map((row) => <div role="row" key={row.key}><span data-label="Modelo">{MODEL_COPY[row.key].title}</span><span data-label="PR-AUC">{row.pr ? num(row.pr) : "no ejecutado"}</span><span data-label="ROC">{row.roc ? num(row.roc) : "—"}</span><span data-label="F1">{row.f1 ? num(row.f1) : "—"}</span><span data-label="Costo">{row.cost}</span><span data-label="Latencia">{row.latency}</span><span data-label="Reproducibilidad">{row.reproducibility}</span></div>)}</div>
    </section>
  );
}

function Evidence({ summary, sample }) {
  const maxPca = Math.max(...summary.pca.first_components.map((row) => row.explained_variance));
  return (
    <section className="panel evidence-grid" role="tabpanel">
      <div className="section-title wide"><p>Trazabilidad · Evidence trail</p><h2>Cada cifra vuelve al dato, al coeficiente o al estado enviado.</h2></div>
      <article className="evidence-card"><h3>Coeficientes que empujan la decisión</h3><div className="coefs">{summary.top_coefficients.slice(0, 12).map((row) => <div className="coef" key={row.feature}><span>{cleanFeature(row.feature)}</span><div className={row.coefficient > 0 ? "positive" : "negative"}><i style={{ width: `${Math.min(row.abs_coefficient * 70, 100)}%` }} /></div><b>{row.coefficient > 0 ? "+" : ""}{num(row.coefficient, 2)}</b></div>)}</div></article>
      <article className="evidence-card"><h3>Multicolinealidad bajo control</h3><div className="vif-list">{summary.eda.highest_vif.slice(0, 8).map((row) => <span key={row.feature}><b>{row.feature}</b>{num(row.vif, 1)} VIF</span>)}</div><h3>PCA como lente, no como excusa</h3><div className="pca-bars">{summary.pca.first_components.map((row) => <span key={row.component} style={{ height: `${Math.max(12, (row.explained_variance / maxPca) * 100)}%` }} title={`PC${row.component}: ${pct(row.explained_variance)}`} />)}</div><p className="note">{summary.pca.components_for_80pct_variance} componentes explican 80% de la varianza numérica.</p></article>
      <article className="evidence-card wide-card"><h3>Estado exacto para Jev y Laya</h3><pre>{sample[0]?.typed_state}</pre></article>
    </section>
  );
}

function Footer() {
  return <footer><p>Construido como laboratorio de producto, no como notebook. Built for real decisions.</p><div><a href="https://eduardomoraga.github.io/" target="_blank" rel="noreferrer">Portfolio</a><a href="https://github.com/EduardoMoraga" target="_blank" rel="noreferrer">GitHub</a><span>Rugol / Moragent DNA</span><span>Dataset público · licencia por verificar antes de publicar raw data</span></div></footer>;
}

function App() {
  const { summary, sample, error } = useData();
  const initial = NAV.some((item) => item.id === window.location.hash.slice(1)) ? window.location.hash.slice(1) : "command";
  const [active, setActive] = useState(initial);
  const [selected, setSelected] = useState(0);
  if (error) return <main className="loading error">No pude cargar los artefactos del modelo: {error.message}</main>;
  if (!summary || sample.length === 0) return <main className="loading">edu · cargando model-room…</main>;
  return <><TopNav active={active} setActive={setActive} /><main><Hero summary={summary} setActive={setActive} />{active === "command" && <CommandCenter summary={summary} sample={sample} selected={selected} setSelected={setSelected} setActive={setActive} />}{active === "simulator" && <Simulator summary={summary} sample={sample} selected={selected} setSelected={setSelected} />}{active === "decision" && <Decision summary={summary} />}{active === "benchmark" && <Benchmark summary={summary} />}{active === "evidence" && <Evidence summary={summary} sample={sample} />}<Footer /></main></>;
}

createRoot(document.getElementById("root")).render(<App />);
