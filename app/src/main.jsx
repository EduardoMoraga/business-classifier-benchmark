import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

/* ---------- formatting (es-CL: decimal comma, dot thousands) ---------- */

const fmt = (d) => new Intl.NumberFormat("es-CL", { minimumFractionDigits: d, maximumFractionDigits: d });
const num = (v, d = 3) => fmt(d).format(Number(v || 0));
const int = (v) => fmt(0).format(Number(v || 0));
const pct = (v, d = 1) => `${fmt(d).format(Number(v || 0) * 100)}%`;
const money = (v) => `$${fmt(0).format(v)}`;
const clean = (f) => f.replace(/^(cat|num)__/, "").replaceAll("_", " ");
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const logit = (p) => Math.log(p / (1 - p));
const day = (iso) => (iso ? new Date(iso).toLocaleDateString("es-CL", { day: "2-digit", month: "2-digit", year: "numeric" }).replaceAll("/", "-") : "—");

const TABS = [
  ["summary", "Resumen"],
  ["bench", "Comparación de modelos"],
  ["lab", "Torneo e híbridos"],
  ["eda", "EDA"],
  ["corr", "Correlación · VIF"],
  ["pca", "PCA"],
  ["model", "Modelo"],
  ["decision", "Umbral y costo"],
  ["sim", "Simulador"],
  ["method", "Metodología"],
];
const SAMPLE_TABS = new Set(["summary", "eda", "sim"]);
const WATCH_MARGIN = 0.25;

/* ---------- data ---------- */

function useArtifacts() {
  const [state, setState] = useState({ summary: null, sample: [], typed: null, tournament: null, error: null });
  useEffect(() => {
    const base = import.meta.env.BASE_URL || "/";
    const get = (file) => fetch(`${base}${file}`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${file}: HTTP ${r.status}`))));
    const optional = (file) => get(file).catch(() => null);
    Promise.all([
      get("model-summary.json"),
      optional("holdout.json").then((rows) => rows || get("holdout-sample.json")),
      optional("typed-benchmark.json"),
      optional("tournament.json"),
    ])
      .then(([summary, sample, typed, tournament]) => setState({ summary, sample, typed, tournament, error: null }))
      .catch((error) => setState({ summary: null, sample: [], typed: null, tournament: null, error }));
  }, []);
  return state;
}

function usePaged(rows, step) {
  const [limit, setLimit] = useState(step);
  useEffect(() => setLimit(step), [rows, step]);
  const more = rows.length > limit && (
    <button className="more" onClick={() => setLimit((l) => l + step)}>Mostrar {Math.min(step, rows.length - limit)} más · quedan {rows.length - limit}</button>
  );
  return [rows.slice(0, limit), more];
}

const PROVIDERS = [["jev", "Jev"], ["laya", "Laya"]];
const typedRun = (typed, id) => {
  const run = typed?.providers?.[id];
  return run && run.rows_scored > 0 ? run : null;
};
const DEPT_SHORT = { "Research & Development": "R&D", "Human Resources": "HR" };

function derive(summary) {
  const [[tn, fp], [fn, tp]] = summary.confusion_matrix;
  const logistic = summary.metrics.find((m) => m.model.toLowerCase().includes("logistic"));
  const dummy = summary.metrics.find((m) => m.model.toLowerCase().includes("dummy"));
  const thr = summary.threshold.selected;
  return { tn, fp, fn, tp, positives: tp + fn, negatives: tn + fp, logistic, dummy, thr };
}

function band(score, thr) {
  if (score >= thr) return { label: "Alto", tone: "bad" };
  if (score >= thr - WATCH_MARGIN) return { label: "Vigilar", tone: "warn" };
  return { label: "Bajo", tone: "good" };
}

function outcome(row, thr) {
  const pred = row.classical_score >= thr;
  if (pred && row.y_true) return "TP";
  if (pred) return "FP";
  if (row.y_true) return "FN";
  return "TN";
}

/* ---------- app ---------- */

function App() {
  const { summary, sample, typed, tournament, error } = useArtifacts();
  const [tab, setTab] = useState("summary");
  const [filters, setFilters] = useState({ dept: "Todos", risk: "Todos", overtime: false, query: "" });
  const [selected, setSelected] = useState(0);

  const d = useMemo(() => (summary ? derive(summary) : null), [summary]);
  const rows = useMemo(() => {
    if (!d) return [];
    const q = filters.query.trim().toLowerCase();
    return sample
      .map((row, idx) => ({ ...row, idx }))
      .filter((r) => filters.dept === "Todos" || r.Department === filters.dept)
      .filter((r) => filters.risk === "Todos" || band(r.classical_score, d.thr).label === filters.risk)
      .filter((r) => !filters.overtime || r.OverTime === "Yes")
      .filter((r) => !q || `${r.JobRole} ${r.Department} ${r.EducationField}`.toLowerCase().includes(q));
  }, [sample, filters, d]);

  if (error) return <main className="state">No se pudieron cargar los artefactos: {error.message}</main>;
  if (!summary) return <main className="state">Cargando Attrition Intelligence…</main>;

  const depts = ["Todos", ...new Set(sample.map((r) => r.Department))];
  const ctx = { summary, sample, typed, tournament, rows, d, setTab, selected, setSelected };
  const pending = PROVIDERS.filter(([id]) => !typedRun(typed, id)).length;

  return (
    <div className="app">
      <Masthead summary={summary} />
      <nav className="tabs" aria-label="Secciones">
        {TABS.map(([id, label], i) => (
          <React.Fragment key={id}>
            {id === "decision" && <span className="tabs-sep">AVANZADO</span>}
            <button className={tab === id ? "on" : ""} onClick={() => setTab(id)}>
              {label}
              {id === "bench" && pending > 0 && <sup className="badge" title="Modelos sin correr">{pending}</sup>}
            </button>
          </React.Fragment>
        ))}
      </nav>
      {SAMPLE_TABS.has(tab) ? (
        <FilterBar filters={filters} setFilters={setFilters} depts={depts} count={rows.length} total={sample.length} holdout={summary.split.test} />
      ) : (
        <div className="scope">
          <span className="scope-label">ALCANCE</span>
          {tab === "lab" ? "Train para ajustar y elegir · holdout intacto para medir · mismas filas para todos" : tab === "bench" ? "Holdout completo · mismas filas, mismas métricas, mismo state string" : tab === "method" ? `Run ${summary.run?.id} · semilla ${summary.split.random_state} · hash ${summary.run?.dataset_hash}` : `Dataset completo · ${int(summary.rows)} filas · ${summary.eda.features} variables · train ${int(summary.split.train)} / test ${int(summary.split.test)}`}
        </div>
      )}
      <main className="page">
        {tab === "summary" && <Summary {...ctx} />}
        {tab === "eda" && <Eda {...ctx} />}
        {tab === "corr" && <Correlation {...ctx} />}
        {tab === "pca" && <Pca {...ctx} />}
        {tab === "model" && <Model {...ctx} />}
        {tab === "decision" && <Decision {...ctx} />}
        {tab === "bench" && <Benchmark {...ctx} />}
        {tab === "lab" && <Lab {...ctx} />}
        {tab === "sim" && <Simulator {...ctx} />}
        {tab === "method" && <Methodology {...ctx} />}
      </main>
      <footer className="foot">
        <span>Run {summary.run?.id} · dataset hash {summary.run?.dataset_hash} · fuente: <a href={summary.source} target="_blank" rel="noreferrer">IBM HR Analytics (Kaggle)</a></span>
        <a href="https://eduardomoraga.github.io/" target="_blank" rel="noreferrer">Eduardo Moraga · Economist & Data Scientist ↗</a>
      </footer>
    </div>
  );
}

function Masthead({ summary }) {
  return (
    <header className="mast">
      <a className="wordmark" href="https://eduardomoraga.github.io/" target="_blank" rel="noreferrer"><span className="sq" />Eduardo Moraga</a>
      <div className="mast-title">
        <h1>Attrition Intelligence<em>.</em></h1>
        <p>{summary.dataset} · Corte {day(summary.run?.created_at)} · Holdout n={summary.split.test} · Logística L1</p>
      </div>
      <div className="mast-right">
        <span>People Analytics · Classifier Benchmark</span>
        <a className="mast-pill" href="https://eduardomoraga.github.io/" target="_blank" rel="noreferrer">Ver portafolio</a>
      </div>
    </header>
  );
}

function FilterBar({ filters, setFilters, depts, count, total, holdout }) {
  const set = (k, v) => setFilters((f) => ({ ...f, [k]: v }));
  return (
    <section className="filters">
      <div className="filters-row">
        <button className={`toggle ${filters.overtime ? "on" : ""}`} onClick={() => set("overtime", !filters.overtime)} aria-pressed={filters.overtime}>
          <i /> Solo horas extra
        </button>
        <span className="flabel">DEPARTAMENTO</span>
        <div className="seg">
          {depts.map((x) => (
            <button key={x} className={filters.dept === x ? "on" : ""} onClick={() => set("dept", x)}>{DEPT_SHORT[x] || x}</button>
          ))}
        </div>
        <span className="flabel">RIESGO</span>
        {["Todos", "Alto", "Vigilar", "Bajo"].map((x) => (
          <button key={x} className={`chip ${filters.risk === x ? "on" : ""}`} onClick={() => set("risk", x)}>{x}</button>
        ))}
        <label className="search">
          <svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.5" /><path d="M13 13l4 4" /></svg>
          <input value={filters.query} onChange={(e) => set("query", e.target.value)} placeholder="Buscar cargo, área o formación…" />
        </label>
      </div>
      <p className="fcount">{count} de {total} empleados del holdout{total < holdout ? ` (muestra de ${holdout})` : ""}</p>
    </section>
  );
}

/* ---------- primitives ---------- */

function SectionHead({ kicker, meta, children }) {
  return (
    <div className="shead">
      <div className="shead-top"><span className="kicker"><span className="sq" />{kicker}</span>{meta && <span className="smeta">{meta}</span>}</div>
      <h2>{children}</h2>
    </div>
  );
}

function KeyMessage({ items }) {
  return (
    <section className="keymsg">
      <span className="kicker dim">KEY MESSAGE</span>
      <ol>{items.map((it, i) => <li key={i}><span className="dot">{i + 1}</span><p>{it}</p></li>)}</ol>
    </section>
  );
}

function Exhibit({ n, title, meta, children, className = "" }) {
  return (
    <section className={`exhibit ${className}`}>
      <header><span className="exn">EXHIBIT {n}</span><h3>{title}</h3>{meta && <span className="exmeta">{meta}</span>}</header>
      {children}
    </section>
  );
}

function Delta({ v, unit = "", good = v >= 0, digits = 1, suffix }) {
  if (Math.abs(v) < 0.5 * 10 ** -digits) return <span className="delta flat">■ 0{unit}{suffix && <em> {suffix}</em>}</span>;
  const up = v >= 0;
  return (
    <span className={`delta ${good ? "pos" : "neg"}`}>
      {up ? "▲" : "▼"} {up ? "+" : "−"}{fmt(digits).format(Math.abs(v))}{unit}{suffix && <em> {suffix}</em>}
    </span>
  );
}

function Kpi({ label, value, delta, sub, chart }) {
  return (
    <article className="kpi">
      <span className="kpi-label">{label}</span>
      <div className="kpi-main"><strong>{value}</strong>{delta}</div>
      <p className="kpi-sub">{sub}</p>
      <div className="kpi-chart">{chart}</div>
    </article>
  );
}

/* Micro chart: two-or-more labelled horizontal bars on a shared 0–1 scale. */
function MicroBars({ items, max = 1 }) {
  return (
    <div className="microbars">
      {items.map(([label, v, strong]) => (
        <div key={label} className="mb-row">
          <span>{label}</span>
          <i><b className={strong ? "strong" : ""} style={{ width: `${(v / max) * 100}%` }} /></i>
        </div>
      ))}
    </div>
  );
}

/* Micro chart: stacked composition bar. */
function Stack({ parts }) {
  const total = parts.reduce((s, p) => s + p[1], 0);
  return (
    <div className="stack">
      <div className="stack-bar">{parts.map(([label, v, tone]) => <b key={label} className={tone} style={{ width: `${(v / total) * 100}%` }} />)}</div>
      <div className="stack-legend">{parts.map(([label, v, tone]) => <span key={label}><i className={tone} />{label} {int(v)}</span>)}</div>
    </div>
  );
}

/* Micro chart: mean ± sd band with a point marker. */
function RangeMark({ lo, hi, mean, mark, markLabel }) {
  const x = (v) => `${v * 100}%`;
  return (
    <div className="range">
      <div className="range-track">
        <span className="range-band" style={{ left: x(lo), width: x(hi - lo) }} />
        <span className="range-mean" style={{ left: x(mean) }} />
        <span className="range-mark" style={{ left: x(mark) }} title={markLabel} />
      </div>
      <div className="range-axis"><span>0</span><span>±1σ CV · ◆ {markLabel}</span><span>1</span></div>
    </div>
  );
}

function Bar({ v, max = 1, tone = "" }) {
  return <i className={`hbar ${tone}`}><b style={{ width: `${Math.min(1, v / max) * 100}%` }} /></i>;
}

function Pill({ tone, children }) {
  return <span className={`pill ${tone}`}>{children}</span>;
}

/* ---------- Resumen ---------- */

function Summary({ summary, typed, rows, d, setTab, setSelected }) {
  const { logistic, dummy, tp, fn, fp, positives, thr } = d;
  const lift = logistic.pr_auc / dummy.pr_auc;
  const imp = summary.permutation_importance;
  const top = imp[0];
  const pair = summary.eda.high_correlation_pairs[0];
  const cv = summary.cv;

  return (
    <>
      <SectionHead kicker="RESUMEN" meta={`Holdout estratificado 20% · random_state ${summary.split.random_state} · umbral ${num(thr, 2)}`}>
        Holdout de {summary.split.test} empleados: el modelo captura {tp} de {positives} salidas reales con PR-AUC {num(logistic.pr_auc)}, {num(lift, 1)}× el baseline
      </SectionHead>

      <section className="kpis">
        <Kpi label="PR-AUC" value={num(logistic.pr_auc)} delta={<Delta v={logistic.pr_auc - dummy.pr_auc} digits={3} suffix="vs dummy" />}
          sub={<>Dummy {num(dummy.pr_auc)} · Lift <b>{num(lift, 1)}×</b></>}
          chart={<MicroBars items={[["Dummy", dummy.pr_auc], ["Modelo", logistic.pr_auc, true]]} />} />
        <Kpi label="ROC-AUC" value={num(logistic.roc_auc)} delta={<Delta v={logistic.roc_auc - dummy.roc_auc} digits={3} suffix="vs azar" />}
          sub={<>Azar {num(dummy.roc_auc)} · separación holdout</>}
          chart={<MicroBars items={[["Azar", dummy.roc_auc], ["Modelo", logistic.roc_auc, true]]} />} />
        <Kpi label="Recall @ umbral" value={pct(summary.threshold.recall)} delta={<Pill tone="neutral">{num(thr, 2)}</Pill>}
          sub={<>TP <b>{tp}</b> · FN <b className="neg">{fn}</b> no capturadas</>}
          chart={<Stack parts={[["Capturadas", tp, "s-blue"], ["No capturadas", fn, "s-red"]]} />} />
        <Kpi label="Precisión @ umbral" value={pct(summary.threshold.precision)} delta={<Pill tone="neutral">F1 {pct(summary.threshold.f1)}</Pill>}
          sub={<>TP <b>{tp}</b> · FP <b>{fp}</b> alertas sin salida</>}
          chart={<Stack parts={[["Acierto", tp, "s-blue"], ["Falsa alarma", fp, "s-amber"]]} />} />
        <Kpi label="Tasa de fuga" value={pct(summary.eda.positive_rate)} delta={<Pill tone="neutral">clase minoritaria</Pill>}
          sub={<>{int(summary.eda.target_balance.yes)} de {int(summary.eda.rows)} empleados en train</>}
          chart={<Stack parts={[["Se va", summary.eda.target_balance.yes, "s-blue"], ["Se queda", summary.eda.target_balance.no, "s-light"]]} />} />
        <Kpi label="CV PR-AUC" value={num(cv.mean)} delta={<span className="delta flat">± {num(cv.std)}</span>}
          sub={<>{cv.folds} folds · 5×5 repetido · holdout {Math.abs(logistic.pr_auc - cv.mean) <= cv.std ? "dentro" : "fuera"} de ±1σ</>}
          chart={<RangeMark lo={cv.mean - cv.std} hi={cv.mean + cv.std} mean={cv.mean} mark={logistic.pr_auc} markLabel="holdout" />} />
      </section>

      <KeyMessage items={[
        <>Señal dominante: <b>{top.feature}</b> explica una caída de <b>{num(top.importance_mean, 3)}</b> en PR-AUC al permutarse, <b>{num(top.importance_mean / imp[1].importance_mean, 1)}×</b> la segunda variable ({imp[1].feature}).</>,
        <>Costo operativo del umbral {num(thr, 2)}: por cada salida capturada hay <b>{num(fp / tp, 2)}</b> falsas alarmas; quedan <b>{fn} salidas</b> sin detectar ({pct(fn / positives)} de las reales).</>,
        <>Redundancia controlada: <b>{pair.feature_a}</b> y <b>{pair.feature_b}</b> correlacionan <b>{num(pair.abs_corr, 2)}</b> (VIF &gt; 10); la regularización L1 evita duplicar la señal.</>,
      ]} />

      <div className="grid-2">
        <Exhibit n={1} title="Motores de fuga" meta="Permutation importance · caída de PR-AUC">
          <table className="dense">
            <thead><tr><th>Variable</th><th className="w-bar">Importancia relativa</th><th className="r">Δ PR-AUC</th></tr></thead>
            <tbody>{imp.map((r, i) => (
              <tr key={r.feature}><td><b>{r.feature}</b></td><td><Bar v={r.importance_mean} max={top.importance_mean} tone={i === 0 ? "dark" : ""} /></td><td className="r num">{num(r.importance_mean, 3)}</td></tr>
            ))}</tbody>
          </table>
        </Exhibit>
        <Exhibit n={2} title="Scorecard de modelos" meta="mismo holdout · n=294">
          <BenchTable summary={summary} typed={typed} d={d} compact />
          <button className="link" onClick={() => setTab("bench")}>Ver benchmark completo →</button>
        </Exhibit>
      </div>

      <Exhibit n={3} title="Score de riesgo por empleado" meta={`holdout · umbral ${num(thr, 2)} · Vigilar = a menos de ${int(WATCH_MARGIN * 100)} pp del umbral`}>
        <EmployeeTable rows={rows} thr={thr} onPick={(idx) => { setSelected(idx); setTab("sim"); }} />
      </Exhibit>
    </>
  );
}

function EmployeeTable({ rows: all, thr, onPick }) {
  const sorted = useMemo(() => [...all].sort((a, b) => b.classical_score - a.classical_score), [all]);
  const [rows, more] = usePaged(sorted, 20);
  if (!all.length) return <p className="empty">Sin empleados para los filtros seleccionados.</p>;
  const groups = [...new Set(rows.map((r) => r.Department))];
  return (
    <div className="table-wrap">
      <table className="dense">
        <thead><tr>
          <th>#</th><th>Cargo</th><th className="r">Edad</th><th className="r">Ingreso mensual</th><th>Horas extra</th><th>Viaje</th>
          <th className="r">Años s/ promoción</th><th className="r">Sat. laboral</th><th className="w-bar">Score de riesgo</th><th>Banda</th><th>Real</th><th>Resultado</th>
        </tr></thead>
        <tbody>
          {groups.map((g) => (
            <React.Fragment key={g}>
              <tr className="group"><td colSpan={12}>{g}</td></tr>
              {rows.filter((r) => r.Department === g).map((r) => {
                const b = band(r.classical_score, thr);
                const o = outcome(r, thr);
                return (
                  <tr key={r.idx} className={onPick ? "pick" : ""} onClick={onPick ? () => onPick(r.idx) : undefined}>
                    <td className="muted">{r.idx + 1}</td>
                    <td><b>{r.JobRole}</b></td>
                    <td className="r num">{r.Age}</td>
                    <td className="r num">{money(r.MonthlyIncome)}</td>
                    <td>{r.OverTime === "Yes" ? <Pill tone="bad">Sí</Pill> : <span className="muted">No</span>}</td>
                    <td className="muted">{r.BusinessTravel.replace("Travel_", "").replace("-", " ")}</td>
                    <td className="r num">{r.YearsSinceLastPromotion}</td>
                    <td className="r"><Dots v={r.JobSatisfaction} /></td>
                    <td><ScoreBar v={r.classical_score} thr={thr} /></td>
                    <td><Pill tone={b.tone}>{b.label}</Pill></td>
                    <td>{r.y_true ? <b>Se fue</b> : <span className="muted">Sigue</span>}</td>
                    <td><span className={`outcome o-${o}`}>{o}</span></td>
                  </tr>
                );
              })}
            </React.Fragment>
          ))}
        </tbody>
      </table>
      {more}
    </div>
  );
}

function ScoreBar({ v, thr }) {
  return (
    <span className="scorebar">
      <i><b className={v >= thr ? "hot" : ""} style={{ width: `${v * 100}%` }} /><em style={{ left: `${thr * 100}%` }} /></i>
      <span className="num">{pct(v)}</span>
    </span>
  );
}

function Dots({ v, max = 4 }) {
  return <span className="dots" title={`${v} de ${max}`}>{Array.from({ length: max }, (_, i) => <i key={i} className={i < v ? "on" : ""} />)}</span>;
}

/* ---------- EDA ---------- */

function Eda({ summary, rows, d }) {
  const { eda, split } = summary;
  const base = eda.positive_rate;
  const byDept = [...new Set(rows.map((r) => r.Department))].map((dep) => {
    const rs = rows.filter((r) => r.Department === dep);
    return { dep, n: rs.length, mean: rs.reduce((s, r) => s + r.classical_score, 0) / rs.length, ot: rs.filter((r) => r.OverTime === "Yes").length, left: rs.filter((r) => r.y_true).length };
  });
  const segments = summary.segments || [];
  const segNames = [...new Set(segments.map((r) => r.segment))];
  const segMax = Math.max(...segments.map((r) => r.rate), base);
  const top = [...segments].filter((r) => r.n >= 30).sort((a, b) => b.rate - a.rate)[0];
  return (
    <>
      <SectionHead kicker="EDA" meta={`solo train · ${int(eda.rows)} filas · el holdout no participa`}>
        {int(summary.rows)} empleados, {eda.features} variables y una clase minoritaria de {pct(base)}: accuracy no sirve, se evalúa con PR-AUC
      </SectionHead>
      <section className="kpis k4">
        <Kpi label="FILAS" value={int(summary.rows)} delta={<Pill tone="neutral">{summary.columns} col.</Pill>} sub={<>{eda.features} features tras limpieza</>} chart={<Stack parts={[["Train", split.train, "s-blue"], ["Test", split.test, "s-light"]]} />} />
        <Kpi label="BALANCE EN TRAIN" value={`${int(eda.target_balance.yes)} / ${int(eda.target_balance.no)}`} delta={<Pill tone="warn">desbalance</Pill>} sub={<>Sí / No · ratio 1:{num(eda.target_balance.no / eda.target_balance.yes, 1)}</>} chart={<Stack parts={[["Sí", eda.target_balance.yes, "s-blue"], ["No", eda.target_balance.no, "s-light"]]} />} />
        <Kpi label="BASELINE PR-AUC" value={num(d.dummy.pr_auc)} delta={<Pill tone="neutral">≈ prevalencia</Pill>} sub="Lo que logra un modelo sin información" chart={<MicroBars items={[["Prevalencia", base], ["Dummy", d.dummy.pr_auc, true]]} />} />
        {top ? (
          <Kpi label="SEGMENTO MÁS EXPUESTO" value={pct(top.rate)} delta={<Pill tone="bad">{num(top.rate / base, 1)}× base</Pill>} sub={<>{top.segment} = <b>{top.level}</b> · n {top.n}</>} chart={<MicroBars items={[["Base", base], [top.level.slice(0, 10), top.rate, true]]} max={segMax} />} />
        ) : (
          <Kpi label="PARES |r| ≥ 0,75" value={eda.high_correlation_pairs.length} delta={<Pill tone="bad">VIF máx {num(eda.highest_vif[0].vif, 1)}</Pill>} sub="Redundancia a tratar antes de modelar" chart={null} />
        )}
      </section>
      {segments.length > 0 && (
        <Exhibit n={1} title="Tasa de fuga por segmento" meta={`train · línea = tasa base ${pct(base)} · segmentos con n < 30 en gris`}>
          <div className="seg-grid">
            {segNames.map((name) => (
              <table key={name} className="dense">
                <thead><tr><th>{name}</th><th className="r">n</th><th className="w-bar">Tasa de fuga</th></tr></thead>
                <tbody>{segments.filter((r) => r.segment === name).sort((a, b) => b.rate - a.rate).map((r) => (
                  <tr key={r.level} className={r.n < 30 ? "pending" : ""}>
                    <td><b>{r.level.replaceAll("_", " ")}</b></td>
                    <td className="r num">{int(r.n)}</td>
                    <td><RateBar v={r.rate} base={base} max={segMax} faint={r.n < 30} /></td>
                  </tr>
                ))}</tbody>
              </table>
            ))}
          </div>
        </Exhibit>
      )}
      <div className="grid-2">
        <Exhibit n={2} title="Perfil del holdout por departamento" meta="filtros aplicados">
          <table className="dense">
            <thead><tr><th>Departamento</th><th className="r">Empleados</th><th className="w-bar">Score medio</th><th className="r">Horas extra</th><th className="r">Salidas reales</th></tr></thead>
            <tbody>{byDept.length ? byDept.map((r) => (
              <tr key={r.dep}><td><b>{r.dep}</b></td><td className="r num">{r.n}</td><td><ScoreBar v={r.mean} thr={d.thr} /></td><td className="r num">{r.ot}</td><td className="r num">{r.left}</td></tr>
            )) : <tr><td colSpan={5} className="empty">Sin filas para los filtros.</td></tr>}</tbody>
          </table>
          <p className="note">Perfil descriptivo del holdout ({rows.length} filas con los filtros actuales). No se usa para decisiones de modelado.</p>
        </Exhibit>
        <Exhibit n={3} title="Diseño de validación" meta="reproducible">
          <table className="dense">
            <tbody>
              <tr><td>Partición</td><td className="r"><b>{int(split.train)} train · {int(split.test)} test</b></td></tr>
              <tr><td>Estratificación</td><td className="r">Sí, por Attrition</td></tr>
              <tr><td>EDA, correlación, VIF, PCA</td><td className="r">{eda.scope === "train_only" ? "Solo train" : "Dataset completo"}</td></tr>
              <tr><td>Validación cruzada</td><td className="r">RepeatedStratifiedKFold 5×5 ({summary.cv.folds} folds)</td></tr>
              <tr><td>Métrica primaria</td><td className="r">Average precision (PR-AUC)</td></tr>
              <tr><td>Semilla</td><td className="r num">{split.random_state}</td></tr>
              <tr><td>Hash del dataset</td><td className="r mono">{summary.run?.dataset_hash}</td></tr>
              <tr><td>Fuente</td><td className="r"><a href={summary.source} target="_blank" rel="noreferrer">Kaggle · IBM HR ↗</a></td></tr>
            </tbody>
          </table>
        </Exhibit>
      </div>
    </>
  );
}

function RateBar({ v, base, max, faint }) {
  return (
    <span className="scorebar">
      <i><b className={!faint && v > base ? "hot" : ""} style={{ width: `${(v / max) * 100}%` }} /><em style={{ left: `${(base / max) * 100}%` }} /></i>
      <span className="num">{pct(v)}</span>
    </span>
  );
}

/* ---------- Correlación · VIF ---------- */

function Correlation({ summary }) {
  const pairs = summary.eda.high_correlation_pairs;
  const vif = summary.eda.highest_vif;
  const feats = [...new Set(pairs.flatMap((p) => [p.feature_a, p.feature_b]))];
  const lookup = new Map(pairs.flatMap((p) => [[`${p.feature_a}|${p.feature_b}`, p.abs_corr], [`${p.feature_b}|${p.feature_a}`, p.abs_corr]]));
  const severe = vif.filter((v) => v.vif >= 10);
  const vmax = Math.ceil(vif[0].vif / 2) * 2;
  const CAREER = ["JobLevel", "MonthlyIncome", "TotalWorkingYears"];
  const career = pairs.filter((p) => CAREER.includes(p.feature_a) || CAREER.includes(p.feature_b));
  const tenure = pairs.filter((p) => p.feature_a.startsWith("Years") && p.feature_b.startsWith("Years"));

  return (
    <>
      <SectionHead kicker="CORRELACIÓN · VIF" meta="solo train · umbral |r| ≥ 0,75 · VIF sobre variables numéricas">
        {pairs[0].feature_a} y {pairs[0].feature_b} comparten {pct(pairs[0].abs_corr ** 2, 0)} de su varianza; {severe.length} variables superan VIF 10
      </SectionHead>
      <KeyMessage items={[
        <>El par más redundante es <b>{pairs[0].feature_a} – {pairs[0].feature_b}</b> con |r| = <b>{num(pairs[0].abs_corr, 2)}</b>: el cargo prácticamente fija el sueldo.</>,
        <><b>{career.length} de {pairs.length}</b> pares involucran el eje de carrera (JobLevel, MonthlyIncome, TotalWorkingYears){tenure.length > 0 && <>; {tenure.length === 1 ? "1 par pertenece" : `${tenure.length} pares pertenecen`} al bloque de <b>antigüedad</b> ({[...new Set(tenure.flatMap((p) => [p.feature_a, p.feature_b]))].join(", ")})</>}.</>,
        <>Con VIF de <b>{num(vif[0].vif, 1)}</b> y <b>{num(vif[1].vif, 1)}</b>, los coeficientes no penalizados serían inestables; por eso el modelo final usa L1 (C = 0,25).</>,
      ]} />
      <div className="grid-2">
        <Exhibit n={1} title="Mapa de correlación de pares críticos" meta="|r| · celdas vacías < 0,75">
          <div className="table-wrap">
            <table className="heat">
              <thead><tr><th />{feats.map((f) => <th key={f}><span>{f}</span></th>)}</tr></thead>
              <tbody>{feats.map((a) => (
                <tr key={a}><th>{a}</th>{feats.map((b) => {
                  const v = a === b ? 1 : lookup.get(`${a}|${b}`);
                  return <td key={b} className={v ? "" : "nil"} style={v ? { "--a": (v - 0.7) / 0.3 } : undefined}>{v ? (a === b ? "1" : num(v, 2)) : ""}</td>;
                })}</tr>
              ))}</tbody>
            </table>
          </div>
        </Exhibit>
        <Exhibit n={2} title="Factor de inflación de varianza" meta="VIF · referencias 5 y 10">
          <div className="vif">
            <div className="vif-guides"><span style={{ left: `${(5 / vmax) * 100}%` }}>5</span><span className="hard" style={{ left: `${(10 / vmax) * 100}%` }}>10</span></div>
            {vif.map((v) => (
              <div key={v.feature} className="vif-row">
                <span>{v.feature}</span>
                <i><b className={v.vif >= 10 ? "bad" : v.vif >= 5 ? "warn" : ""} style={{ width: `${(v.vif / vmax) * 100}%` }} /></i>
                <em className="num">{num(v.vif, 1)}</em>
              </div>
            ))}
          </div>
        </Exhibit>
      </div>
      <Exhibit n={3} title="Pares con correlación alta" meta={`${pairs.length} pares`}>
        <table className="dense">
          <thead><tr><th>Variable A</th><th>Variable B</th><th className="r">|r|</th><th className="r">R² compartido</th><th className="w-bar">Intensidad</th><th>Lectura</th></tr></thead>
          <tbody>{pairs.map((p) => (
            <tr key={p.feature_a + p.feature_b}>
              <td><b>{p.feature_a}</b></td><td><b>{p.feature_b}</b></td>
              <td className="r num">{num(p.abs_corr, 3)}</td><td className="r num">{pct(p.abs_corr ** 2, 0)}</td>
              <td><Bar v={p.abs_corr - 0.7} max={0.3} tone={p.abs_corr >= 0.9 ? "red" : ""} /></td>
              <td>{p.abs_corr >= 0.9 ? <Pill tone="bad">Redundante</Pill> : <Pill tone="warn">Solapada</Pill>}</td>
            </tr>
          ))}</tbody>
        </table>
      </Exhibit>
    </>
  );
}

/* ---------- PCA ---------- */

function Pca({ summary }) {
  const { pca } = summary;
  const comps = pca.first_components;
  let acc = 0;
  const pts = comps.map((c) => ({ ...c, cum: (acc += c.explained_variance) }));
  const W = 640, H = 260, P = { l: 40, r: 40, t: 24, b: 34 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const bw = iw / comps.length;
  const y = (v) => P.t + ih - v * ih;
  const line = pts.map((p, i) => `${P.l + bw * i + bw / 2},${y(p.cum)}`).join(" ");

  return (
    <>
      <SectionHead kicker="PCA" meta={`${pca.numeric_features} variables numéricas estandarizadas`}>
        PC1 explica {pct(comps[0].explained_variance)} y se necesitan {pca.components_for_80pct_variance} de {pca.numeric_features} componentes para llegar a 80%: la información está repartida
      </SectionHead>
      <section className="kpis k4">
        <Kpi label="PC1" value={pct(comps[0].explained_variance)} delta={<Pill tone="neutral">eje de carrera</Pill>} sub="Antigüedad, nivel e ingreso" chart={<MicroBars items={[["PC1", comps[0].explained_variance, true], ["PC2", comps[1].explained_variance]]} max={0.25} />} />
        <Kpi label="PC1–PC3" value={pct(pts[2].cum)} delta={<Pill tone="neutral">acumulado</Pill>} sub="Tres ejes principales" chart={<Stack parts={[["PC1–3", pts[2].cum * 100, "s-blue"], ["Resto", (1 - pts[2].cum) * 100, "s-light"]]} />} />
        <Kpi label={`PC1–PC${comps.length}`} value={pct(pts.at(-1).cum)} delta={<Pill tone="neutral">acumulado</Pill>} sub={`${comps.length} componentes reportados`} chart={<Stack parts={[[`PC1–${comps.length}`, pts.at(-1).cum * 100, "s-blue"], ["Resto", (1 - pts.at(-1).cum) * 100, "s-light"]]} />} />
        <Kpi label="COMPONENTES PARA 80%" value={pca.components_for_80pct_variance} delta={<Pill tone="warn">de {pca.numeric_features}</Pill>} sub="Reducción posible pero con pérdida interpretativa" chart={<Stack parts={[["Necesarios", pca.components_for_80pct_variance, "s-blue"], ["Prescindibles", pca.numeric_features - pca.components_for_80pct_variance, "s-light"]]} />} />
      </section>
      <div className="grid-2 wide-left">
        <Exhibit n={1} title="Varianza explicada por componente" meta="barra = individual · línea = acumulada">
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Varianza explicada por componente principal">
            {[0, 0.2, 0.4, 0.6, 0.8].map((t) => (
              <g key={t}><line x1={P.l} x2={W - P.r} y1={y(t)} y2={y(t)} className={t === 0.8 ? "goal" : "grid"} /><text x={P.l - 8} y={y(t) + 4} className="axis" textAnchor="end">{int(t * 100)}%</text></g>
            ))}
            <text x={W - P.r} y={y(0.8) - 6} className="axis goal-label" textAnchor="end">objetivo 80% → PC{pca.components_for_80pct_variance}</text>
            {pts.map((p, i) => (
              <g key={p.component}>
                <rect x={P.l + bw * i + bw * 0.2} y={y(p.explained_variance)} width={bw * 0.6} height={ih - (y(p.explained_variance) - P.t)} className={i === 0 ? "bar-strong" : "bar"} />
                <text x={P.l + bw * i + bw / 2} y={y(p.explained_variance) - 6} className="val" textAnchor="middle">{num(p.explained_variance * 100, 1)}</text>
                <text x={P.l + bw * i + bw / 2} y={H - 12} className={`axis ${i === 0 ? "b" : ""}`} textAnchor="middle">PC{p.component}</text>
              </g>
            ))}
            <polyline points={line} className="cum" />
            {pts.map((p, i) => <circle key={p.component} cx={P.l + bw * i + bw / 2} cy={y(p.cum)} r="3.5" className="cum-dot" />)}
          </svg>
        </Exhibit>
        <Exhibit n={2} title="Tabla de componentes" meta="varianza individual y acumulada">
          <table className="dense">
            <thead><tr><th>Componente</th><th className="r">Individual</th><th className="r">Acumulada</th><th className="w-bar">Hacia 80%</th></tr></thead>
            <tbody>{pts.map((p) => (
              <tr key={p.component}><td><b>PC{p.component}</b></td><td className="r num">{pct(p.explained_variance)}</td><td className="r num">{pct(p.cum)}</td><td><Bar v={p.cum} max={0.8} /></td></tr>
            ))}</tbody>
          </table>
          <p className="note">Decisión: no se reemplazan variables por componentes; la pérdida de interpretabilidad no compensa en un modelo que RR.HH. debe poder auditar.</p>
        </Exhibit>
      </div>
    </>
  );
}

/* ---------- XY chart (curves, calibration, cost) ---------- */

function XYChart({ series, diag, refY, marker, xLabel, yLabel, yMax = 1, xFmt = (v) => num(v, 2), yFmt = (v) => num(v, 2), label }) {
  const W = 420, H = 300, P = { l: 52, r: 14, t: 14, b: 42 };
  const iw = W - P.l - P.r, ih = H - P.t - P.b;
  const x = (v) => P.l + v * iw;
  const y = (v) => P.t + ih - (v / yMax) * ih;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const path = (pts) => pts.map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join("");
  return (
    <svg className="chart xy" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={P.l} x2={W - P.r} y1={y(t * yMax)} y2={y(t * yMax)} className="grid" />
          <text x={P.l - 8} y={y(t * yMax) + 4} className="axis" textAnchor="end">{yFmt(t * yMax)}</text>
          <text x={x(t)} y={H - 22} className="axis" textAnchor="middle">{xFmt(t)}</text>
        </g>
      ))}
      <text x={P.l + iw / 2} y={H - 4} className="axis b" textAnchor="middle">{xLabel}</text>
      <text x={12} y={P.t + ih / 2} className="axis b" textAnchor="middle" transform={`rotate(-90 12 ${P.t + ih / 2})`}>{yLabel}</text>
      {diag && <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(yMax)} className="diag" />}
      {refY != null && (
        <g><line x1={P.l} x2={W - P.r} y1={y(refY.v)} y2={y(refY.v)} className="diag" /><text x={P.l + 6} y={y(refY.v) - 6} className="axis" textAnchor="start">{refY.label}</text></g>
      )}
      {series.map((sr) => (
        <g key={sr.key}>
          <path d={path(sr.points)} className={`curve ${sr.cls || ""}`} />
          {sr.dots && sr.points.map(([a, b], i) => <circle key={i} cx={x(a)} cy={y(b)} r="4" className={`cdot ${sr.cls || ""}`} />)}
        </g>
      ))}
      {marker && (
        <g>
          <rect x={x(marker.x) - 5} y={y(marker.y) - 5} width="10" height="10" className="marker" />
          <text x={x(marker.x) + (marker.x > 0.7 ? -10 : 10)} y={y(marker.y) - 10} className="val" textAnchor={marker.x > 0.7 ? "end" : "start"}>{marker.label}</text>
        </g>
      )}
    </svg>
  );
}

/* ---------- Modelo ---------- */

function Model({ summary, d }) {
  const coefs = summary.top_coefficients;
  const max = Math.max(...coefs.map((c) => c.abs_coefficient));
  const up = coefs.filter((c) => c.coefficient > 0);
  const down = coefs.filter((c) => c.coefficient < 0);
  return (
    <>
      <SectionHead kicker="MODELO" meta="LogisticRegression · L1 · liblinear · C = 0,25 · class_weight balanced">
        Regresión logística parsimoniosa: {up.length} factores suben el riesgo y {down.length} lo bajan; el mayor efecto es {clean(coefs[0].feature)} (odds × {num(coefs[0].odds_ratio, 2)})
      </SectionHead>
      <KeyMessage items={[
        <><b>{clean(up[0].feature)}</b> multiplica las odds de salida por <b>{num(up[0].odds_ratio, 2)}</b> frente a la categoría base.</>,
        <>Sin horas extra (<b>{clean(down[0].feature)}</b>) las odds caen a <b>{num(down[0].odds_ratio, 2)}×</b>: es la palanca de gestión más directa.</>,
        <>Variables numéricas están estandarizadas: cada coeficiente es el efecto de <b>+1 desviación estándar</b> sobre el log-odds.</>,
      ]} />
      <div className="grid-2 wide-left">
        <Exhibit n={1} title="Coeficientes y odds ratio" meta="top 15 por |coeficiente|">
          <table className="dense">
            <thead><tr><th>Variable</th><th className="w-div">← baja riesgo · sube riesgo →</th><th className="r">Coef.</th><th className="r">Odds ratio</th></tr></thead>
            <tbody>{coefs.map((c) => (
              <tr key={c.feature}>
                <td><b>{clean(c.feature)}</b> <span className="muted tag-src">{c.feature.startsWith("num__") ? "num" : "cat"}</span></td>
                <td><span className="div-bar"><i className={c.coefficient > 0 ? "up" : "down"} style={{ width: `${(c.abs_coefficient / max) * 50}%` }} /></span></td>
                <td className={`r num ${c.coefficient > 0 ? "neg" : "pos"}`}>{c.coefficient > 0 ? "+" : "−"}{num(Math.abs(c.coefficient), 2)}</td>
                <td className="r num"><b>{num(c.odds_ratio, 2)}×</b></td>
              </tr>
            ))}</tbody>
          </table>
        </Exhibit>
        <Exhibit n={2} title="Matriz de confusión" meta={`holdout · umbral ${num(d.thr, 3)}`}>
          <Confusion d={d} />
          <table className="dense mt">
            <tbody>
              <tr><td>Precisión</td><td className="r num"><b>{pct(summary.threshold.precision)}</b></td></tr>
              <tr><td>Recall</td><td className="r num"><b>{pct(summary.threshold.recall)}</b></td></tr>
              <tr><td>F1</td><td className="r num"><b>{pct(summary.threshold.f1)}</b></td></tr>
              <tr><td>Especificidad</td><td className="r num"><b>{pct(d.tn / d.negatives)}</b></td></tr>
              <tr><td>Brier score</td><td className="r num">{num(d.logistic.brier, 3)} <span className="muted">(dummy {num(d.dummy.brier, 3)})</span></td></tr>
            </tbody>
          </table>
          <p className="note">El umbral maximiza F1 sobre el holdout (regla <span className="mono">{summary.threshold.rule}</span>): sirve para la demo, no como umbral productivo.</p>
        </Exhibit>
      </div>
      {summary.curves && <Curves summary={summary} d={d} />}
    </>
  );
}

function Curves({ summary, d }) {
  const cal = summary.calibration || [];
  const hi = cal.at(-1);
  const prev = d.positives / (d.positives + d.negatives);
  const opPoint = { recall: d.tp / d.positives, precision: d.tp / (d.tp + d.fp), fpr: d.fp / d.negatives };
  return (
    <>
      <KeyMessage items={[
        <>La curva PR se mantiene sobre la línea base de <b>{pct(prev)}</b> en todo el rango de recall: el ordenamiento es útil, no sólo en el umbral elegido.</>,
        hi && <>Las probabilidades están <b>infladas</b>: en el tramo más alto el modelo predice <b>{pct(hi.predicted)}</b> y la fuga observada es <b>{pct(hi.observed)}</b>. Es el costo de <b>class_weight balanced</b>.</>,
        <>Lectura de negocio: usar el score para <b>priorizar</b> casos, no como probabilidad literal, hasta calibrar (Platt o isotónica).</>,
      ].filter(Boolean)} />
      <div className="grid-3">
        <Exhibit n={3} title="Curva precision–recall" meta={`AP ${num(d.logistic.pr_auc)} · ■ umbral ${num(d.thr, 2)}`}>
          <XYChart label="Curva precision-recall" xLabel="Recall" yLabel="Precisión" xFmt={(v) => pct(v, 0)} yFmt={(v) => pct(v, 0)}
            series={[{ key: "pr", points: summary.curves.pr }]} refY={{ v: prev, label: `base ${pct(prev)}` }}
            marker={{ x: opPoint.recall, y: opPoint.precision, label: `${pct(opPoint.recall, 0)} · ${pct(opPoint.precision, 0)}` }} />
        </Exhibit>
        <Exhibit n={4} title="Curva ROC" meta={`AUC ${num(d.logistic.roc_auc)} · diagonal = azar`}>
          <XYChart label="Curva ROC" xLabel="Tasa de falsos positivos" yLabel="Tasa de verdaderos positivos" xFmt={(v) => pct(v, 0)} yFmt={(v) => pct(v, 0)}
            series={[{ key: "roc", points: summary.curves.roc }]} diag
            marker={{ x: opPoint.fpr, y: opPoint.recall, label: `FPR ${pct(opPoint.fpr, 0)}` }} />
        </Exhibit>
        <Exhibit n={5} title="Calibración" meta={`${cal.length} tramos por cuantil · diagonal = perfecta`}>
          <XYChart label="Curva de calibración" xLabel="Probabilidad predicha" yLabel="Fuga observada" xFmt={(v) => pct(v, 0)} yFmt={(v) => pct(v, 0)}
            series={[{ key: "cal", points: cal.map((c) => [c.predicted, c.observed]), dots: true, cls: "accent" }]} diag />
        </Exhibit>
      </div>
    </>
  );
}

function Confusion({ d }) {
  const cells = [
    ["TN", d.tn, "Se queda · predicho queda", "c-ok"],
    ["FP", d.fp, "Se queda · predicho sale", "c-warn"],
    ["FN", d.fn, "Sale · predicho queda", "c-bad"],
    ["TP", d.tp, "Sale · predicho sale", "c-hit"],
  ];
  return (
    <div className="conf">
      <span className="conf-y">Real</span>
      <span className="conf-x">Predicción → Queda · Sale</span>
      {cells.map(([k, v, lbl, cls]) => <div key={k} className={cls}><span>{k}</span><strong>{v}</strong><em>{lbl}</em></div>)}
    </div>
  );
}

/* ---------- Umbral y costo ---------- */

function Decision({ summary, d }) {
  const [costFn, setCostFn] = useState(10000);
  const [costFp, setCostFp] = useState(1000);
  const sweep = summary.threshold_sweep || [];
  const [t, setT] = useState(() => Math.round(d.thr * 100) / 100);
  const cost = (r) => r.fn * costFn + r.fp * costFp;
  const at = sweep.find((r) => Math.abs(r.threshold - t) < 1e-6) || { threshold: d.thr, tp: d.tp, fp: d.fp, fn: d.fn, tn: d.tn };
  const optimum = sweep.length ? sweep.reduce((a, b) => (cost(b) < cost(a) ? b : a)) : at;

  const strategies = [
    { name: "No intervenir a nadie", fn: d.positives, fp: 0, flagged: 0 },
    { name: "Intervenir a todos", fn: 0, fp: d.negatives, flagged: d.positives + d.negatives },
    { name: `Modelo @ ${num(at.threshold, 2)}`, fn: at.fn, fp: at.fp, flagged: at.tp + at.fp, model: true },
  ].map((s) => ({ ...s, cost: s.fn * costFn + s.fp * costFp }));
  const best = strategies.reduce((a, b) => (b.cost < a.cost ? b : a));
  const alt = Math.min(strategies[0].cost, strategies[1].cost);
  const model = strategies[2];
  const max = Math.max(...strategies.map((s) => s.cost));
  const costMax = Math.max(...sweep.map(cost), 1);
  const precision = at.tp + at.fp ? at.tp / (at.tp + at.fp) : 0;
  const recall = at.tp / d.positives;

  return (
    <>
      <SectionHead kicker="UMBRAL Y COSTO" meta={`supuestos editables · ${d.positives + d.negatives} filas del holdout`}>
        Con umbral {num(at.threshold, 2)} el modelo {model.cost <= alt ? "ahorra" : "cuesta"} {money(Math.abs(alt - model.cost))} frente a la mejor alternativa sin modelo; el umbral de menor costo es {num(optimum.threshold, 2)}
      </SectionHead>
      <section className="assumptions">
        <label><span>Costo de una salida no detectada (FN)</span><input type="number" min="0" step="500" value={costFn} onChange={(e) => setCostFn(Math.max(0, Number(e.target.value)))} /></label>
        <label><span>Costo de una intervención innecesaria (FP)</span><input type="number" min="0" step="100" value={costFp} onChange={(e) => setCostFp(Math.max(0, Number(e.target.value)))} /></label>
        {sweep.length > 0 && (
          <label className="slider"><span>Umbral de intervención · {num(at.threshold, 2)}</span>
            <input type="range" min={sweep[0].threshold} max={sweep.at(-1).threshold} step="0.01" value={t} onChange={(e) => setT(Number(e.target.value))} />
          </label>
        )}
        {sweep.length > 0 && <button className="chip" onClick={() => setT(optimum.threshold)}>Usar óptimo ({num(optimum.threshold, 2)})</button>}
        <p>Valores por defecto de <span className="mono">src/evaluation/business_cost.py</span>. Ratio FN:FP = <b>{costFp ? num(costFn / costFp, 1) : "∞"}:1</b>.</p>
      </section>
      <section className="kpis k4">
        <Kpi label="INTERVENIDOS" value={int(at.tp + at.fp)} delta={<Pill tone="neutral">{pct((at.tp + at.fp) / (d.positives + d.negatives), 0)} del holdout</Pill>} sub={<>TP <b>{at.tp}</b> · FP <b>{at.fp}</b></>} chart={<Stack parts={[["Acierto", at.tp, "s-blue"], ["Falsa alarma", at.fp, "s-amber"]]} />} />
        <Kpi label="RECALL" value={pct(recall)} delta={<Pill tone="neutral">{at.tp} de {d.positives}</Pill>} sub={<>FN <b className="neg">{at.fn}</b> salidas no detectadas</>} chart={<Stack parts={[["Capturadas", at.tp, "s-blue"], ["No capturadas", at.fn, "s-red"]]} />} />
        <Kpi label="PRECISIÓN" value={pct(precision)} delta={<Pill tone="neutral">{num(at.tp + at.fp ? at.fp / Math.max(at.tp, 1) : 0, 2)} FP por TP</Pill>} sub="Acierto de cada intervención" chart={<MicroBars items={[["Base", d.positives / (d.positives + d.negatives)], ["Modelo", precision, true]]} />} />
        <Kpi label="COSTO TOTAL" value={money(model.cost)} delta={<Delta v={(model.cost - alt) / 1000} unit=" k" good={model.cost <= alt} digits={0} suffix="vs alternativa" />} sub={<>Óptimo {money(cost(optimum))} @ {num(optimum.threshold, 2)}</>} chart={<MicroBars items={[["Alternativa", alt], ["Modelo", model.cost, true]]} max={Math.max(alt, model.cost, 1)} />} />
      </section>
      <div className="grid-2 wide-left">
        <Exhibit n={1} title="Costo esperado por estrategia" meta="FN × costo FN + FP × costo FP">
          <table className="dense">
            <thead><tr><th>Estrategia</th><th className="r">Intervenidos</th><th className="r">FN</th><th className="r">FP</th><th className="w-bar">Costo total</th><th className="r">$</th><th /></tr></thead>
            <tbody>{strategies.map((s) => (
              <tr key={s.name} className={s.model ? "hl" : ""}>
                <td><b>{s.name}</b></td><td className="r num">{int(s.flagged)}</td><td className="r num">{s.fn}</td><td className="r num">{s.fp}</td>
                <td><Bar v={s.cost} max={max || 1} tone={s === best ? "dark" : ""} /></td><td className="r num"><b>{money(s.cost)}</b></td>
                <td>{s === best && <Pill tone="good">Menor costo</Pill>}</td>
              </tr>
            ))}</tbody>
          </table>
        </Exhibit>
        {sweep.length > 0 && (
          <Exhibit n={2} title="Costo según umbral" meta="■ umbral actual">
            <XYChart label="Costo total por umbral" xLabel="Umbral" yLabel="Costo total" yMax={costMax} yFmt={(v) => `${fmt(0).format(v / 1000)}k`}
              series={[{ key: "cost", points: sweep.map((r) => [r.threshold, cost(r)]), cls: "accent" }]}
              marker={{ x: at.threshold, y: cost(at), label: money(cost(at)) }} />
          </Exhibit>
        )}
      </div>
      <p className="note">El umbral óptimo se calcula sobre el mismo holdout que se reporta: es optimista. En producción se fija con validación cruzada sobre train.</p>
    </>
  );
}

/* ---------- Benchmark ---------- */

function BenchTable({ summary, typed, d, compact }) {
  const n = summary.split.test;
  const adapters = Object.fromEntries((summary.typed_decision_models || []).map((m) => [m.model, m.status]));
  const pendingStatus = { Jev: "Adapter listo · requiere TYPESAFE_API_KEY", Laya: "Adapter listo · requiere LAYA_API_KEY" };
  const rows = [
    { name: "Dummy prior", kind: "Baseline", input: "Prevalencia", m: d.dummy, rows: `${n}/${n}`, latency: "local" },
    { name: "Logística L1", kind: "ML clásico", input: "Tabular estandarizado", m: d.logistic, rows: `${n}/${n}`, latency: "local" },
    ...PROVIDERS.map(([id, name]) => {
      const run = typedRun(typed, id);
      return {
        name, kind: "Typed decision", input: "State string",
        m: run?.metrics?.pr_auc != null ? run.metrics : null,
        rows: run ? `${run.rows_scored}/${run.rows_requested}` : "—",
        latency: run?.latency_ms_p50 != null ? `${fmt(0).format(run.latency_ms_p50)} ms` : "—",
        status: run ? (run.metrics?.pr_auc == null ? "Corrido · sin ambas clases" : null) : adapters[name] ? pendingStatus[name] : "Sin adapter",
      };
    }),
  ];
  return (
    <div className="table-wrap">
      <table className="dense">
        <thead><tr><th>Modelo</th>{!compact && <th>Tipo</th>}{!compact && <th>Entrada</th>}<th className="w-bar">PR-AUC</th><th className="r">ROC-AUC</th>{!compact && <th className="r">Brier</th>}{!compact && <th className="r">F1 @ 0,5</th>}{!compact && <th className="r">Filas</th>}{!compact && <th className="r">Latencia p50</th>}<th>Estado</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.name} className={r.m ? "" : "pending"}>
            <td><b>{r.name}</b></td>
            {!compact && <td className="muted">{r.kind}</td>}
            {!compact && <td className="muted">{r.input}</td>}
            <td>{r.m ? <span className="scorebar"><i><b className={r.name.startsWith("Log") ? "hot-blue" : ""} style={{ width: `${r.m.pr_auc * 100}%` }} /></i><span className="num">{num(r.m.pr_auc)}</span></span> : <span className="ghost">sin correr</span>}</td>
            <td className="r num">{r.m ? num(r.m.roc_auc) : "—"}</td>
            {!compact && <td className="r num">{r.m ? num(r.m.brier) : "—"}</td>}
            {!compact && <td className="r num">{r.m ? num(r.m.f1) : "—"}</td>}
            {!compact && <td className="r num">{r.rows}</td>}
            {!compact && <td className="r num muted">{r.latency}</td>}
            <td>{r.m ? <Pill tone="good">Ejecutado</Pill> : <Pill tone="warn">{compact ? "Pendiente" : r.status}</Pill>}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

const CMP_THR = 0.5;

function compareRows(summary, typed, d) {
  const at = (summary.threshold_sweep || []).find((r) => Math.abs(r.threshold - CMP_THR) < 1e-6);
  const rows = [
    { name: "Sin modelo (dummy)", note: "Asigna a todos la misma probabilidad (16%): nunca alerta", m: d.dummy, c: { tp: 0, fp: 0, fn: d.positives, tn: d.negatives } },
    { name: "Regresión logística", note: "Modelo estadístico clásico sobre la tabla de datos", m: d.logistic, c: at },
  ];
  PROVIDERS.forEach(([id, name]) => {
    const run = typedRun(typed, id);
    rows.push(run
      ? { name, note: `Lee la ficha del empleado en texto · ${run.rows_scored} de ${summary.split.test} empleados`, m: run.metrics, c: run.confusion, partial: run.rows_scored < summary.split.test, run }
      : { name, note: "Aún no corre: falta la API key", m: null, c: null });
  });
  return rows;
}

function PlainCompare({ summary, typed, d }) {
  const rows = compareRows(summary, typed, d);
  const scored = rows.filter((r) => r.m?.pr_auc != null && !r.partial);
  const best = scored.reduce((a, b) => (b.m.pr_auc > a.m.pr_auc ? b : a), scored[0]);
  const prev = d.positives / (d.positives + d.negatives);
  return (
    <Exhibit n={1} title="¿Qué tan bien detecta cada modelo a quienes se van?" meta={`mismos empleados · alerta si el riesgo ≥ ${num(CMP_THR, 1)}`}>
      <div className="cmp-grid">
        {rows.map((r) => {
          const c = r.c;
          const pos = c ? c.tp + c.fn : 0;
          const alerts = c ? c.tp + c.fp : 0;
          const quality = r.m?.pr_auc != null ? Math.max(0, (r.m.pr_auc - prev) / (1 - prev)) : null;
          return (
            <article key={r.name} className={`cmp ${r === best ? "best" : ""} ${c ? "" : "off"} ${r.partial ? "partial" : ""}`}>
              <header>
                <h4>{r.name}{r === best && <Pill tone="good">Mejor lista de riesgo</Pill>}{r.partial && <Pill tone="warn">Parcial · no comparable</Pill>}</h4>
                <p>{r.note}</p>
              </header>
              {c ? (
                <>
                  <p className="cmp-head">Detecta <b>{c.tp} de {pos}</b> personas que se fueron{alerts ? <> · {pct(c.tp / alerts, 0)} de sus alertas son correctas</> : " · no emite alertas"}</p>
                  <div className="mini-conf">
                    <div className="c-hit"><strong>{c.tp}</strong><span>Alertas correctas</span><em>se iban y el modelo avisó</em></div>
                    <div className="c-warn"><strong>{c.fp}</strong><span>Falsas alarmas</span><em>avisó, pero se quedaron</em></div>
                    <div className="c-bad"><strong>{c.fn}</strong><span>Se le escaparon</span><em>se fueron sin aviso</em></div>
                    <div className="c-ok"><strong>{c.tn}</strong><span>Bien descartados</span><em>se quedaron, sin aviso</em></div>
                  </div>
                  <div className="cmp-score">
                    <span>Calidad de la lista de riesgo</span>
                    <i><b style={{ width: `${(quality ?? 0) * 100}%` }} /></i>
                    <div><em>azar</em><b className="num">{num(r.m.pr_auc)}</b><em>perfecto</em></div>
                  </div>
                </>
              ) : <p className="empty">Sin resultados todavía. No se rellenan con estimaciones.</p>}
              {r.partial && <p className="note">Evaluado sólo en {r.run.rows_scored} empleados ({r.run.positives} se fueron); las demás llamadas fallaron. Con tan pocos casos la nota no es comparable con la de los otros modelos.</p>}
            </article>
          );
        })}
      </div>
    </Exhibit>
  );
}

function Glossary({ d }) {
  const prev = d.positives / (d.positives + d.negatives);
  const terms = [
    ["Alerta correcta", "TP", "La persona se fue y el modelo la marcó como riesgo. Es lo que queremos maximizar."],
    ["Falsa alarma", "FP", "El modelo marcó riesgo pero la persona se quedó. Cuesta una conversación o un incentivo innecesario."],
    ["Se le escapó", "FN", "La persona se fue y el modelo no avisó. Suele ser el error más caro."],
    ["Detección", "Recall", "De todos los que se fueron, qué porcentaje marcó el modelo."],
    ["Acierto de alertas", "Precisión", "De todas las alertas, qué porcentaje era real."],
    ["Calidad de la lista", "PR-AUC", `Si ordenas a todos de mayor a menor riesgo, ¿quedan arriba los que realmente se van? Al azar da ${num(prev, 2)}; perfecto es 1.`],
    ["Separación", "ROC-AUC", "Tomando al azar uno que se fue y uno que se quedó: probabilidad de que el modelo le dé más riesgo al que se fue. 0,5 es lanzar una moneda."],
    ["Error de probabilidad", "Brier", "Distancia promedio entre la probabilidad dicha y lo que pasó. Más bajo es mejor."],
  ];
  return (
    <Exhibit n={2} title="Cómo leer esta comparación" meta="glosario en palabras simples">
      <div className="glossary">
        {terms.map(([plain, tech, text]) => (
          <div key={tech}><b>{plain}</b><span className="mono">{tech}</span><p>{text}</p></div>
        ))}
      </div>
    </Exhibit>
  );
}

function Benchmark({ summary, typed, sample, d }) {
  const state = sample[0]?.typed_state || "";
  const runs = PROVIDERS.map(([id, name]) => [name, typedRun(typed, id)]).filter(([, r]) => r);
  const executed = 2 + runs.filter(([, r]) => r.metrics?.pr_auc != null).length;
  return (
    <>
      <SectionHead kicker="COMPARACIÓN DE MODELOS" meta="ML clásico vs modelos de decisión tipada (Jev, Laya)">
        {runs.length
          ? (() => {
              const full = runs.filter(([, r]) => r.rows_scored >= summary.split.test && r.metrics?.pr_auc != null);
              const beat = full.filter(([, r]) => r.metrics.pr_auc > d.logistic.pr_auc);
              return full.length
                ? `Sobre los mismos ${summary.split.test} empleados, ${beat.length ? beat.map(([n]) => n).join(" y ") + " supera" : "la regresión logística sigue siendo el mejor modelo: ordena el riesgo mejor que " + full.map(([n]) => n).join(" y ")} `
                : `${executed} de 4 modelos con métricas; ningún modelo tipado cubrió el holdout completo todavía`;
            })()
          : `Sobre los mismos ${summary.split.test} empleados, la regresión logística ordena el riesgo ${num(d.logistic.pr_auc / d.dummy.pr_auc, 1)} veces mejor que no tener modelo; Jev y Laya esperan su API key`}
      </SectionHead>
      <KeyMessage items={[
        (() => { const c = compareRows(summary, typed, d)[1].c; return c && <>Con alerta desde 50% de riesgo, la <b>regresión logística detecta {c.tp} de {c.tp + c.fn}</b> personas que se fueron, a cambio de <b>{c.fp} falsas alarmas</b>.</>; })(),
        <><b>Sin modelo no se detecta a nadie</b>: es la vara mínima que cualquier modelo debe superar.</>,
        <>Los porcentajes de riesgo de la logística están <b>inflados</b> (ver calibración en Modelo): sirven para ordenar a quién mirar primero, no como probabilidad literal.</>,
        runs.length
          ? <>Modelos tipados: {runs.map(([name, r]) => r.rows_scored >= summary.split.test
              ? <React.Fragment key={name}><b>{name}</b> detecta {r.confusion.tp} de {r.confusion.tp + r.confusion.fn} con {r.confusion.fp} falsas alarmas (lista de riesgo {num(r.metrics.pr_auc)} vs {num(r.classical_same_rows.pr_auc)} de la logística). </React.Fragment>
              : <React.Fragment key={name}><b>{name}</b> sólo alcanzó {r.rows_scored} de {summary.split.test} empleados: no comparable. </React.Fragment>)}</>
          : <>Jev y Laya leerán la <b>misma ficha de cada empleado</b> en texto, sin saber si se fue; sus casillas quedan vacías hasta que corran.</>,
      ].filter(Boolean)} />
      <PlainCompare summary={summary} typed={typed} d={d} />
      <Glossary d={d} />
      <Exhibit n={3} title="Detalle técnico" meta={`holdout n=${summary.split.test} · mismas métricas`}>
        <BenchTable summary={summary} typed={typed} d={d} />
      </Exhibit>
      {runs.length > 0 && (
        <Exhibit n={4} title="Cara a cara en las mismas filas" meta="typed model vs logística sobre el subconjunto efectivamente puntuado">
          <table className="dense">
            <thead><tr><th>Modelo</th><th className="r">Filas</th><th className="r">Salidas</th><th className="r">PR-AUC typed</th><th className="r">PR-AUC logística</th><th className="r">ROC-AUC typed</th><th className="r">ROC-AUC logística</th><th className="r">p95 latencia</th></tr></thead>
            <tbody>{runs.map(([name, r]) => (
              <tr key={name}>
                <td><b>{name}</b></td><td className="r num">{r.rows_scored}</td><td className="r num">{r.positives}</td>
                <td className="r num"><b>{r.metrics.pr_auc != null ? num(r.metrics.pr_auc) : "—"}</b></td><td className="r num">{r.classical_same_rows.pr_auc != null ? num(r.classical_same_rows.pr_auc) : "—"}</td>
                <td className="r num"><b>{r.metrics.roc_auc != null ? num(r.metrics.roc_auc) : "—"}</b></td><td className="r num">{r.classical_same_rows.roc_auc != null ? num(r.classical_same_rows.roc_auc) : "—"}</td>
                <td className="r num muted">{r.latency_ms_p95 != null ? `${fmt(0).format(r.latency_ms_p95)} ms` : "—"}</td>
              </tr>
            ))}</tbody>
          </table>
        </Exhibit>
      )}
      <div className="grid-2">
        <Exhibit n={runs.length ? 5 : 4} title="Protocolo de comparación" meta="condiciones para publicar métricas">
          <ol className="steps">
            <li><b>Mismo holdout</b><span>{summary.split.test} filas, semilla {summary.split.random_state}, hash {summary.run?.dataset_hash}</span></li>
            <li><b>Misma entrada</b><span>Fila tabular serializada como state string, sin la etiqueta</span></li>
            <li><b>Mismas métricas</b><span>PR-AUC, ROC-AUC, Brier y F1; la logística se recalcula sobre las mismas filas puntuadas</span></li>
            <li><b>Latencia</b><span>p50 y p95 por llamada, medidos en el runner</span></li>
          </ol>
          <pre className="state-str cmd">{`uv run python scripts/benchmark_typed_models.py --provider jev
uv run python scripts/benchmark_typed_models.py --provider laya`}</pre>
        </Exhibit>
        <Exhibit n={runs.length ? 6 : 5} title="Ejemplo de state string" meta={`empleado row_id ${sample[0]?.row_id ?? 0}`}>
          <pre className="state-str">{state}</pre>
        </Exhibit>
      </div>
    </>
  );
}

/* ---------- Simulador ---------- */

function useLevers(summary, row) {
  const c = Object.fromEntries(summary.top_coefficients.map((x) => [x.feature, x.coefficient]));
  const has = (k) => typeof c[k] === "number";
  const levers = [];
  if (row.OverTime === "Yes" && has("cat__OverTime_No") && has("cat__OverTime_Yes"))
    levers.push({ id: "ot", label: "Eliminar horas extra", detail: "OverTime Yes → No", delta: c["cat__OverTime_No"] - c["cat__OverTime_Yes"] });
  if (row.BusinessTravel === "Travel_Frequently" && has("cat__BusinessTravel_Non-Travel") && has("cat__BusinessTravel_Travel_Frequently"))
    levers.push({ id: "bt", label: "Suspender viajes frecuentes", detail: "Travel_Frequently → Non-Travel", delta: c["cat__BusinessTravel_Non-Travel"] - c["cat__BusinessTravel_Travel_Frequently"] });
  [
    ["num__JobSatisfaction", "Subir satisfacción laboral", "+1 DE"],
    ["num__EnvironmentSatisfaction", "Mejorar ambiente de trabajo", "+1 DE"],
    ["num__JobInvolvement", "Aumentar involucramiento", "+1 DE"],
  ].forEach(([k, label, detail]) => has(k) && levers.push({ id: k, label, detail, delta: c[k] }));
  if (has("num__YearsSinceLastPromotion"))
    levers.push({ id: "promo", label: "Promover antes", detail: "−1 DE años sin promoción", delta: -c["num__YearsSinceLastPromotion"] });
  return levers;
}

function Simulator({ summary, sample, rows, d, selected, setSelected }) {
  const row = sample[selected] || sample[0];
  const sorted = useMemo(() => [...rows].sort((a, b) => b.classical_score - a.classical_score), [rows]);
  const [list, more] = usePaged(sorted, 25);
  const levers = useLevers(summary, row);
  const [on, setOn] = useState({});
  useEffect(() => setOn({}), [selected]);

  const shift = levers.reduce((s, l) => s + (on[l.id] ? l.delta : 0), 0);
  const base = row.classical_score;
  const scen = sigmoid(logit(Math.min(0.9999, Math.max(0.0001, base))) + shift);
  const b0 = band(base, d.thr), b1 = band(scen, d.thr);

  return (
    <>
      <SectionHead kicker="SIMULADOR" meta="what-if sobre el log-odds del modelo · efecto ceteris paribus">
        {row.JobRole} · {row.Department}: riesgo {pct(base)} → {pct(scen)} con {Object.values(on).filter(Boolean).length} palanca(s) activas
      </SectionHead>
      <div className="grid-2 sim">
        <Exhibit n={1} title="Seleccionar empleado" meta="holdout filtrado · ordenado por score">
          <div className="table-wrap">
            <table className="dense">
              <thead><tr><th>#</th><th>Cargo</th><th>Horas extra</th><th className="w-bar">Score</th></tr></thead>
              <tbody>{list.length ? list.map((r) => (
                <tr key={r.idx} className={`pick ${r.idx === selected ? "sel" : ""}`} onClick={() => setSelected(r.idx)}>
                  <td className="muted">{r.idx + 1}</td><td><b>{r.JobRole}</b><br /><span className="muted small">{r.Department}</span></td>
                  <td>{r.OverTime === "Yes" ? <Pill tone="bad">Sí</Pill> : <span className="muted">No</span>}</td>
                  <td><ScoreBar v={r.classical_score} thr={d.thr} /></td>
                </tr>
              )) : <tr><td colSpan={4} className="empty">Sin empleados para los filtros.</td></tr>}</tbody>
            </table>
            {more}
          </div>
        </Exhibit>
        <div className="sim-right">
          <section className="kpis k3">
            <Kpi label="RIESGO ACTUAL" value={pct(base)} delta={<Pill tone={b0.tone}>{b0.label}</Pill>} sub={<>Real: <b>{row.y_true ? "se fue" : "sigue"}</b> · {outcome(row, d.thr)}</>} chart={<ScoreBar v={base} thr={d.thr} />} />
            <Kpi label="RIESGO ESCENARIO" value={pct(scen)} delta={<Delta v={(scen - base) * 100} unit=" pp" good={scen <= base} />} sub={<>Banda: <b>{b1.label}</b></>} chart={<ScoreBar v={scen} thr={d.thr} />} />
            <Kpi label="Δ LOG-ODDS" value={shift === 0 ? num(0, 2) : (shift < 0 ? "−" : "+") + num(Math.abs(shift), 2)} delta={<Pill tone="neutral">odds × {num(Math.exp(shift), 2)}</Pill>} sub="Suma de coeficientes activados" chart={<MicroBars items={[["Actual", base], ["Escenario", scen, true]]} />} />
          </section>
          <Exhibit n={2} title="Palancas de retención" meta="solo variables con coeficiente publicado">
            {levers.length ? (
              <table className="dense">
                <thead><tr><th /><th>Palanca</th><th>Cambio</th><th className="r">Δ log-odds</th><th className="r">Odds ×</th></tr></thead>
                <tbody>{levers.map((l) => (
                  <tr key={l.id} className={`pick ${on[l.id] ? "sel" : ""}`} onClick={() => setOn((s) => ({ ...s, [l.id]: !s[l.id] }))}>
                    <td><span className={`check ${on[l.id] ? "on" : ""}`} /></td>
                    <td><b>{l.label}</b></td><td className="muted">{l.detail}</td>
                    <td className={`r num ${l.delta < 0 ? "pos" : "neg"}`}>{l.delta < 0 ? "−" : "+"}{num(Math.abs(l.delta), 2)}</td>
                    <td className="r num">{num(Math.exp(l.delta), 2)}</td>
                  </tr>
                ))}</tbody>
              </table>
            ) : <p className="empty">Sin palancas disponibles para este perfil.</p>}
            <p className="note">Desplaza el log-odds con los coeficientes del modelo (variables numéricas en desviaciones estándar). Es una lectura del modelo, no un efecto causal.</p>
          </Exhibit>
          <Exhibit n={3} title="Perfil" meta={`row_id ${row.row_id ?? selected}`}>
            <div className="profile">
              {[["Edad", row.Age], ["Ingreso", money(row.MonthlyIncome)], ["Nivel", row.JobLevel], ["Horas extra", row.OverTime === "Yes" ? "Sí" : "No"], ["Viaje", row.BusinessTravel.replace("Travel_", "")], ["Estado civil", row.MaritalStatus], ["Años en empresa", row.YearsAtCompany], ["Años s/ promoción", row.YearsSinceLastPromotion], ["Empresas previas", row.NumCompaniesWorked], ["Distancia", `${row.DistanceFromHome} km`]].map(([k, v]) => (
                <div key={k}><span>{k}</span><b>{v}</b></div>
              ))}
              {[["Sat. laboral", row.JobSatisfaction], ["Sat. ambiente", row.EnvironmentSatisfaction], ["Involucramiento", row.JobInvolvement], ["Balance vida", row.WorkLifeBalance]].map(([k, v]) => (
                <div key={k}><span>{k}</span><Dots v={v} /></div>
              ))}
            </div>
          </Exhibit>
        </div>
      </div>
    </>
  );
}

/* ---------- Torneo e híbridos ---------- */

/* Paired bootstrap interval of PR-AUC(candidate) − PR-AUC(production logistic), in plain words. */
function verdict(vs) {
  if (!vs) return { label: "Referencia", tone: "" };
  if (vs.low > 0) return { label: "Mejor que la logística", tone: "good" };
  if (vs.high < 0) return { label: "Peor que la logística", tone: "bad" };
  return { label: "Empate técnico", tone: "warn" };
}

const HYBRIDS = [
  ["jev", "Jev solo", "lee la ficha"],
  ["jev_hybrid", "Jev + modelo", "ficha + score y factores de la logística"],
  ["laya", "Laya solo", "lee la ficha"],
  ["laya_hybrid", "Laya + modelo", "ficha + score y factores de la logística"],
];

function Lab({ summary, typed, tournament, sample, d }) {
  if (!tournament) {
    return (
      <>
        <SectionHead kicker="TORNEO E HÍBRIDOS">Torneo de modelos pendiente</SectionHead>
        <pre className="state-str cmd">uv run python scripts/run_tournament.py</pre>
      </>
    );
  }
  const n = tournament.holdout_size;
  const cands = tournament.candidates;
  const prod = cands.find((c) => c.id === tournament.production);
  const dummy = cands.find((c) => c.id === "dummy");
  const unbalanced = cands.find((c) => c.id === "logistic_unbalanced");
  const challengers = cands.filter((c) => !["dummy", tournament.production, "logistic_unbalanced"].includes(c.id));
  const better = challengers.filter((c) => c.vs_production?.low > 0);
  const bestChallenger = challengers.reduce((a, b) => (b.holdout.pr_auc > a.holdout.pr_auc ? b : a), challengers[0]);
  const ranked = [...cands].sort((a, b) => b.cv.mean - a.cv.mean);
  const maxPr = Math.max(...cands.map((c) => Math.max(c.cv.mean, c.holdout.pr_auc)));
  const boruta = cands.filter((c) => c.id.startsWith("boruta"));
  const hybrids = HYBRIDS.map(([id, name, input]) => ({ id, name, input, run: typedRun(typed, id) })).filter((h) => h.run);
  const hyb = (id) => hybrids.find((h) => h.id === id)?.run;
  const pos = prod.holdout.confusion.tp + prod.holdout.confusion.fn;
  const ties = challengers.filter((c) => verdict(c.vs_production).tone === "warn");
  const worse = challengers.filter((c) => verdict(c.vs_production).tone === "bad");
  const withModel = hybrids.filter((h) => h.id.endsWith("_hybrid"));
  const echoes = withModel.length > 0 && withModel.every((h) => (h.run.vs_classical?.rank_agreement ?? 0) > 0.9);
  const hybridMsg = withModel.length ? (
    <>Darle a los LLM el resultado del modelo: {withModel.map((h) => {
      const solo = hyb(h.id.replace("_hybrid", ""));
      const vs = h.run.vs_classical;
      return (
        <React.Fragment key={h.id}><b>{h.name}</b> pasa de {solo ? num(solo.metrics.pr_auc) : "—"} a {num(h.run.metrics.pr_auc)} ({verdict(vs).label.toLowerCase()}){vs?.rank_agreement != null && <>, y ordena a las personas casi igual que la logística (correlación {num(vs.rank_agreement, 2)})</>}. </React.Fragment>
      );
    })}{echoes ? <>El LLM <b>adopta el score del modelo</b>: sirve como interfaz, no como fuente de señal adicional.</> : <>El LLM <b>combina</b> el score con su propia lectura de la ficha.</>}</>
  ) : <>La prueba híbrida (LLM + modelo) espera resultados: <code>benchmark_typed_models.py --variant hybrid</code>.</>;

  const accuracyRows = [
    { name: "Nadie se va (sin modelo)", acc: tournament.accuracy_if_nobody_leaves, c: { tp: 0, fp: 0 } },
    unbalanced && { name: unbalanced.name, acc: unbalanced.holdout.accuracy, c: unbalanced.holdout.confusion },
    { name: prod.name, acc: prod.holdout.accuracy, c: prod.holdout.confusion },
  ].filter(Boolean);

  return (
    <>
      <SectionHead kicker="TORNEO E HÍBRIDOS" meta={`${challengers.length} retadores · ajustados sólo con train · holdout n=${n}`}>
        {better.length
          ? `${better.map((c) => c.name).join(" y ")} supera a la regresión logística con diferencia estadística`
          : `Ningún modelo más complejo le gana con claridad a la regresión logística: el mejor retador (${bestChallenger.name}) pasa de ${num(prod.holdout.pr_auc)} a ${num(bestChallenger.holdout.pr_auc)}, dentro del margen de azar`}
      </SectionHead>
      <KeyMessage items={[
        <>El <b>“accuracy” engaña</b> en este problema: decir que <b>nadie se va acierta {pct(tournament.accuracy_if_nobody_leaves, 0)}</b> y no detecta a ninguna de las {pos} personas que se fueron.{unbalanced && <> Una logística optimizada para acertar llega a <b>{pct(unbalanced.holdout.accuracy, 0)}</b>, pero detecta sólo {unbalanced.holdout.confusion.tp} de {pos}.</>}</>,
        <>De {challengers.length} retadores ajustados (Random Forest, XGBoost, LightGBM, Boruta, ensamble): <b>{ties.length} empatan</b> con la logística y <b>{worse.length} son peores</b>{better.length ? <> y {better.length} la superan</> : <>; ninguno la supera</>}. {worse.length > 0 && <>Peores: {worse.map((c) => c.name).join(", ")}.</>}</>,
        boruta.length > 0 && <><b>Boruta</b> deja sólo {boruta.map((c) => c.n_features).join(" y ")} de {prod.n_features} variables y <b>empeora</b> la lista ({boruta.map((c) => num(c.holdout.pr_auc)).join(" / ")}): con {int(summary.split.train)} personas descarta señales débiles que, sumadas, sí predicen.</>,
        hybridMsg,
      ].filter(Boolean)} />

      <div className="grid-2 wide-left">
        <Exhibit n={1} title="¿Por qué el “accuracy” se ve bajo?" meta={`holdout n=${n} · ${pos} se fueron`}>
          <table className="dense">
            <thead><tr><th>Estrategia</th><th className="w-bar">Accuracy (acierto global)</th><th className="r">Detecta</th><th className="r">Falsas alarmas</th></tr></thead>
            <tbody>{accuracyRows.map((r) => (
              <tr key={r.name}>
                <td><b>{r.name}</b></td>
                <td><span className="scorebar"><i><b style={{ width: `${r.acc * 100}%` }} /></i><span className="num">{pct(r.acc, 1)}</span></span></td>
                <td className="r num"><b>{r.c.tp}</b> de {pos}</td>
                <td className="r num">{r.c.fp}</td>
              </tr>
            ))}</tbody>
          </table>
          <p className="note">Sólo {pct(1 - tournament.accuracy_if_nobody_leaves, 0)} de las personas se va, así que el acierto global premia no alertar. Por eso comparamos con la <b>calidad de la lista</b> (PR-AUC), que mide si quienes se van quedan arriba en el ranking de riesgo.</p>
        </Exhibit>
        <Exhibit n={2} title="Lo que dice la evidencia" meta="lectura para negocio">
          <ol className="steps">
            <li><b>Techo del dataset</b><span>Ningún enfoque pasa de {num(Math.max(...cands.map((c) => c.holdout.roc_auc)), 2)} de ROC-AUC: el límite lo pone la información disponible, no el algoritmo.</span></li>
            <li><b>{better.length ? "Hay un retador mejor" : "Simplicidad gana"}</b><span>{better.length ? `${better.map((c) => c.name).join(" y ")} supera a la logística con diferencia estadística.` : "La logística empata o supera a todos los retadores y además se explica factor por factor."}</span></li>
            <li><b>LLM como capa</b><span>{withModel.length ? `Con el resultado del modelo: ${withModel.map((h) => `${h.name} ${num(h.run.metrics.pr_auc)}`).join(", ")} vs logística ${num(prod.holdout.pr_auc)}.${echoes ? " Lo igualan porque repiten su orden." : ""}` : "Pendiente de la corrida híbrida."}</span></li>
          </ol>
        </Exhibit>
      </div>

      <Exhibit n={3} title="Torneo de modelos" meta="ordenado por validación cruzada en train · el holdout se mira una sola vez">
        <div className="table-wrap">
          <table className="dense lab-table">
            <thead><tr><th>Modelo</th><th className="w-bar">Calidad de lista · CV (±1σ)</th><th className="r">Holdout PR-AUC</th><th className="r">ROC-AUC</th><th className="r">Accuracy</th><th className="r">Detecta</th><th className="r">Falsas alarmas</th><th>vs logística (IC 95%)</th><th className="r">Variables</th></tr></thead>
            <tbody>{ranked.map((c) => {
              const v = verdict(c.vs_production);
              return (
                <tr key={c.id} className={c.id === tournament.production ? "hl" : ""}>
                  <td><b title={c.plain}>{c.name}</b><div className="muted small">{c.family}</div></td>
                  <td><span className="scorebar"><i><b className={c.id === tournament.production ? "hot-blue" : ""} style={{ width: `${(c.cv.mean / maxPr) * 100}%` }} /></i><span className="num">{num(c.cv.mean)} ± {num(c.cv.std, 2)}</span></span></td>
                  <td className="r num"><b>{num(c.holdout.pr_auc)}</b></td>
                  <td className="r num">{num(c.holdout.roc_auc)}</td>
                  <td className="r num">{pct(c.holdout.accuracy, 0)}</td>
                  <td className="r num">{c.holdout.confusion.tp}/{pos}</td>
                  <td className="r num">{c.holdout.confusion.fp}</td>
                  <td>{c.vs_production
                    ? <><Pill tone={v.tone}>{v.label}</Pill><div className="muted small num">{num(c.vs_production.low)} a {num(c.vs_production.high)}</div></>
                    : <span className="muted">{c.id === "dummy" ? "vara mínima" : "modelo actual"}</span>}</td>
                  <td className="r num muted">{c.n_features ?? "—"}</td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
        <p className="note">Detecta y falsas alarmas con alerta desde 50% de riesgo. “vs logística” es la diferencia de PR-AUC en el holdout con bootstrap pareado (1.000 remuestreos): si el intervalo cruza el 0, la diferencia no se distingue del azar. {tournament.notes?.[1] && "La validación de los modelos ajustados reutiliza train, lo que los favorece levemente: aun así no ganan."}</p>
      </Exhibit>

      <Exhibit n={4} title="¿Y si el LLM recibe el resultado del modelo?" meta="mismos empleados · misma pregunta · cambia sólo la ficha">
        {hybrids.length ? (
          <div className="table-wrap">
            <table className="dense">
              <thead><tr><th>Enfoque</th><th>Qué recibe</th><th className="w-bar">Calidad de lista (PR-AUC)</th><th className="r">ROC-AUC</th><th className="r">Detecta</th><th className="r">Falsas alarmas</th><th className="r">Promedio 50/50 con logística</th><th className="r">Latencia p50</th></tr></thead>
              <tbody>
                <tr className="hl"><td><b>{prod.name}</b></td><td className="muted">tabla de datos</td><td><span className="scorebar"><i><b className="hot-blue" style={{ width: `${prod.holdout.pr_auc * 100}%` }} /></i><span className="num">{num(prod.holdout.pr_auc)}</span></span></td><td className="r num">{num(prod.holdout.roc_auc)}</td><td className="r num">{prod.holdout.confusion.tp}/{pos}</td><td className="r num">{prod.holdout.confusion.fp}</td><td className="r muted">—</td><td className="r num muted">local</td></tr>
                {hybrids.map((h) => (
                  <tr key={h.id} className={h.run.rows_scored < n ? "pending" : ""}>
                    <td><b>{h.name}</b>{h.run.rows_scored < n && <Pill tone="warn">{h.run.rows_scored}/{n}</Pill>}</td>
                    <td className="muted">{h.input}</td>
                    <td><span className="scorebar"><i><b style={{ width: `${(h.run.metrics.pr_auc ?? 0) * 100}%` }} /></i><span className="num">{h.run.metrics.pr_auc != null ? num(h.run.metrics.pr_auc) : "—"}</span></span></td>
                    <td className="r num">{h.run.metrics.roc_auc != null ? num(h.run.metrics.roc_auc) : "—"}</td>
                    <td className="r num">{h.run.confusion.tp}/{h.run.confusion.tp + h.run.confusion.fn}</td>
                    <td className="r num">{h.run.confusion.fp}</td>
                    <td className="r num">{h.run.blend_with_classical?.metrics?.pr_auc != null ? num(h.run.blend_with_classical.metrics.pr_auc) : "—"}</td>
                    <td className="r num muted">{h.run.latency_ms_p50 != null ? `${fmt(0).format(h.run.latency_ms_p50)} ms` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="empty">Sin corridas híbridas todavía.</p>}
        <p className="note">El enfoque híbrido usa al modelo de lenguaje como capa de despliegue: recibe la ficha del empleado más el score de la logística, su percentil y los factores que más suben y bajan su riesgo. “Promedio 50/50” combina ambos puntajes sin ajustar pesos, para no contaminar el holdout.</p>
        {sample[0]?.hybrid_state && (
          <details className="state-details">
            <summary>Ver la ficha híbrida que recibe el LLM (empleado {sample[0].row_id})</summary>
            <pre className="state-str">{sample[0].hybrid_state}</pre>
          </details>
        )}
      </Exhibit>
    </>
  );
}

/* ---------- Metodología ---------- */

function Methodology({ summary, typed }) {
  const cfg = summary.model_config || {};
  const runs = PROVIDERS.filter(([id]) => typedRun(typed, id));
  const steps = [
    ["Split congelado y estratificado", `${summary.split.train} train / ${summary.split.test} test, semilla ${summary.split.random_state}`, true],
    ["EDA sólo sobre train", "Correlación, VIF, PCA y tasas por segmento sin tocar el holdout", summary.eda?.scope === "train_only"],
    ["Reducción de redundancia", `${summary.eda.high_correlation_pairs.length} pares |r| ≥ 0,75; VIF máx ${num(summary.eda.highest_vif[0].vif, 1)}; L1 hace la selección`, true],
    ["PCA como lente, no como modelo", `${summary.pca.components_for_80pct_variance} de ${summary.pca.numeric_features} componentes para 80%: no compensa perder interpretabilidad`, true],
    ["Logística parsimoniosa", `PR-AUC ${num(summary.metrics.at(-1).pr_auc)} en holdout · CV ${num(summary.cv.mean)} ± ${num(summary.cv.std)}`, true],
    ["Calibración y curvas", "Tabla de confiabilidad, PR y ROC exportadas por el pipeline", Boolean(summary.calibration)],
    ["Jev y Laya en las mismas filas", runs.length ? `${runs.map(([, n]) => n).join(" y ")} corridos` : "Adapters listos; faltan API keys", runs.length === PROVIDERS.length],
    ["Métricas de negocio", "Costo por umbral en la app; latencia en el runner tipado", runs.length > 0 ? true : "partial"],
  ];
  return (
    <>
      <SectionHead kicker="METODOLOGÍA" meta="docs/methodology.md · docs/acceptance-checks.md">
        Split congelado, EDA sólo en train, un modelo que RR.HH. puede auditar y un benchmark que no publica lo que no corrió
      </SectionHead>
      <div className="grid-2 wide-left">
        <Exhibit n={1} title="Pipeline y estado" meta={`${steps.filter((x) => x[2] === true).length} de ${steps.length} completos`}>
          <table className="dense">
            <thead><tr><th>#</th><th>Paso</th><th>Evidencia</th><th>Estado</th></tr></thead>
            <tbody>{steps.map(([name, evidence, ok], i) => (
              <tr key={name}>
                <td className="muted num">{String(i + 1).padStart(2, "0")}</td><td><b>{name}</b></td><td className="muted">{evidence}</td>
                <td>{ok === true ? <Pill tone="good">Listo</Pill> : ok === "partial" ? <Pill tone="warn">Parcial</Pill> : <Pill tone="warn">Pendiente</Pill>}</td>
              </tr>
            ))}</tbody>
          </table>
        </Exhibit>
        <Exhibit n={2} title="Configuración del modelo" meta="model_config">
          <table className="dense">
            <tbody>
              {[["Estimador", cfg.estimator], ["Solver · penalización", cfg.solver && `${cfg.solver} · ${cfg.penalty}`], ["C (inverso de regularización)", cfg.C != null && num(cfg.C, 2)], ["class_weight", cfg.class_weight], ["Preprocesamiento", cfg.preprocessing], ["Columnas eliminadas", cfg.dropped_columns?.join(", ")], ["Validación cruzada", cfg.cv], ["Tamaño test", cfg.test_size != null && pct(cfg.test_size, 0)]]
                .filter(([, v]) => v).map(([k, v]) => <tr key={k}><td>{k}</td><td className="r"><b>{v}</b></td></tr>)}
            </tbody>
          </table>
        </Exhibit>
      </div>
      <div className="grid-2">
        <Exhibit n={3} title="Reproducir" meta="desde la raíz del repo">
          <pre className="state-str cmd">{`uv sync --extra dev && npm install
uv run python scripts/run_pipeline.py     # artefactos en app/public y reports/
uv run pytest -q && uv run ruff check .
npm run build                             # sitio estático en app/dist`}</pre>
        </Exhibit>
        <Exhibit n={4} title="Límites conocidos" meta="lo que esta demo no prueba">
          <ol className="steps">
            <li><b>Umbral optimista</b><span>Se eligió maximizando F1 sobre el mismo holdout que se reporta</span></li>
            <li><b>Probabilidades infladas</b><span>class_weight balanced sin calibración posterior</span></li>
            <li><b>Dataset sintético</b><span>IBM HR es ficticio: sirve para método, no para conclusiones de negocio reales</span></li>
            <li><b>Simulador correlacional</b><span>Mueve el log-odds con coeficientes; no estima efectos causales</span></li>
          </ol>
        </Exhibit>
      </div>
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
