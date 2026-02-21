import { useEffect, useMemo, useState } from "react";

async function fetchJson(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Request failed");
  }
  return res.json();
}

function formatNum(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const num = Number(value);
  return Number.isInteger(num) ? `${num}` : num.toFixed(digits);
}

export default function App() {
  const [config, setConfig] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
  const [configFailed, setConfigFailed] = useState(false);
  const [status, setStatus] = useState(null);
  const [benchmark, setBenchmark] = useState([]);
  const [errorSummary, setErrorSummary] = useState([]);
  const [errorLength, setErrorLength] = useState([]);
  const [errorSignal, setErrorSignal] = useState([]);
  const [errorConfidence, setErrorConfidence] = useState([]);
  const [images, setImages] = useState([]);
  const [activeTab, setActiveTab] = useState("setup");

  const DEFAULT_MODELS = ["modernbert", "cryptobert", "finbert", "bert-base", "roberta-base"];

  useEffect(() => {
    fetchJson("/api/config")
      .then((data) => {
        setConfig(data);
        setSelectedModels(data.models || DEFAULT_MODELS);
      })
      .catch(() => {
        setConfig({ models: DEFAULT_MODELS, datasets: { train: "", valid: "" } });
        setSelectedModels(DEFAULT_MODELS);
        setConfigFailed(true);
      });
    refreshStatus();
  }, []);

  const refreshStatus = async () => {
    const data = await fetchJson("/api/status");
    setStatus(data);
  };

  useEffect(() => {
    if (!status || status.status !== "running") return;
    const timer = setInterval(refreshStatus, 2000);
    return () => clearInterval(timer);
  }, [status]);

  const loadResults = async () => {
    const bench = await fetchJson("/api/benchmark");
    setBenchmark(bench.rows || []);
    const summary = await fetchJson("/api/errors/error_summary");
    setErrorSummary(summary.rows || []);
    const length = await fetchJson("/api/errors/error_by_length");
    setErrorLength(length.rows || []);
    const signal = await fetchJson("/api/errors/error_by_signal");
    setErrorSignal(signal.rows || []);
    const conf = await fetchJson("/api/errors/error_by_confidence");
    setErrorConfidence(conf.rows || []);
    const imgs = await fetchJson("/api/images");
    setImages(imgs.images || []);
  };

  useEffect(() => {
    if (status?.status === "completed") {
      loadResults();
      setActiveTab("overview");
    }
  }, [status]);

  const canShowTabs = status?.status === "completed";

  const leaderboard = useMemo(() => {
    return [...benchmark].sort((a, b) => (b.f1_macro ?? 0) - (a.f1_macro ?? 0));
  }, [benchmark]);

  const startTraining = async () => {
    await fetchJson("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ models: selectedModels }),
    });
    refreshStatus();
  };

  const progress = status?.progress ?? 0;
  const progressPercent = Math.round(progress * 100);

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="eyebrow">Quant NLP Research Platform</p>
          <h1>Model Benchmark Dashboard</h1>
          <p className="sub">
            Run the full training + benchmark pipeline and compare five BERT-family models with clear trade-offs.
          </p>
        </div>
        <div className="status">
          <div className={`pill ${status?.status || "idle"}`}>
            {status?.status || "idle"}
          </div>
          <button className="btn" onClick={refreshStatus}>
            Refresh
          </button>
        </div>
      </header>

      {status && status.status === "running" && (
        <section className="progress-card">
          <div className="progress-inner">
            <span>Training progress</span>
            <strong>{progressPercent}% complete</strong>
          </div>
          <div className="progress-track">
            <div className="progress-bar" style={{ width: `${progressPercent}%` }} />
          </div>
        </section>
      )}

      <section className="card">
        <h2>Experiment Setup</h2>
        <p className="muted">
          Select models and launch training. Tabs unlock once the pipeline finishes.
        </p>
        <div className="grid two">
          <div>
            <h3>Models</h3>
            <div className="chip-group">
              {(config?.models || DEFAULT_MODELS).map((model) => {
                const checked = selectedModels.includes(model);
                return (
                  <label key={model} className={`chip ${checked ? "active" : ""}`}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => {
                        setSelectedModels((prev) =>
                          e.target.checked ? [...prev, model] : prev.filter((m) => m !== model)
                        );
                      }}
                    />
                    {model}
                  </label>
                );
              })}
            </div>
          </div>
          <div>
            <h3>Dataset</h3>
            <div className="dataset">
              <div>
                <span>Train</span>
                <code>{config?.datasets?.train || "dataset/bitcoin_sent_train.csv"}</code>
              </div>
              <div>
                <span>Valid</span>
                <code>{config?.datasets?.valid || "dataset/bitcoin_sent_valid.csv"}</code>
              </div>
            </div>
            {configFailed && (
              <p className="muted">
                Unable to load config from backend; using defaults.
              </p>
            )}
          </div>
        </div>
        <div className="actions">
          <button className="btn primary" onClick={startTraining} disabled={!selectedModels.length || status?.status === "running"}>
            Run Training + Benchmark
          </button>
          {status?.status === "failed" && <p className="error">Job failed: {status.error}</p>}
        </div>
      </section>

      <section className="card logs">
        <h2>Job Logs</h2>
        <div className="log-box">
          {(status?.logs || []).slice(-200).map((line, idx) => (
            <div key={idx}>{line}</div>
          ))}
        </div>
      </section>

      <nav className={`tabs ${canShowTabs ? "" : "locked"}`}>
        {[
          { id: "overview", label: "Overview" },
          { id: "benchmarks", label: "Benchmarks" },
          { id: "errors", label: "Error Analysis" },
          { id: "artifacts", label: "Artifacts" },
        ].map((tab) => (
          <button
            key={tab.id}
            className={activeTab === tab.id ? "active" : ""}
            onClick={() => canShowTabs && setActiveTab(tab.id)}
            disabled={!canShowTabs}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {activeTab === "overview" && canShowTabs && (
        <section className="grid three">
          <div className="card">
            <h3>Leaderboard (Macro F1)</h3>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>F1</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((row) => (
                  <tr key={row.model}>
                    <td>{row.model}</td>
                    <td>{formatNum(row.f1_macro)}</td>
                    <td>{formatNum(row.accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card">
            <h3>Error Summary</h3>
            <div className="summary-grid">
              {errorSummary.map((row) => (
                <div key={row.key} className="summary-card">
                  <span>{row.key.replaceAll("_", " ")}</span>
                  <strong>{formatNum(row.value)}</strong>
                </div>
              ))}
            </div>
          </div>
          <div className="card">
            <h3>Latency Snapshot</h3>
            <ul className="list">
              {leaderboard.map((row) => (
                <li key={row.model}>
                  <span>{row.model}</span>
                  <strong>{formatNum(row.latency_ms_per_tweet, 2)} ms</strong>
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {activeTab === "benchmarks" && canShowTabs && (
        <section className="card">
          <h2>Benchmark Results</h2>
          <table>
            <thead>
              <tr>
                {benchmark[0] &&
                  Object.keys(benchmark[0]).map((key) => <th key={key}>{key}</th>)}
              </tr>
            </thead>
            <tbody>
              {benchmark.map((row) => (
                <tr key={row.model}>
                  {Object.keys(row).map((key) => (
                    <td key={key}>{formatNum(row[key])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {activeTab === "errors" && canShowTabs && (
        <section className="grid three">
          <div className="card">
            <h3>Error by Length</h3>
            <table>
              <thead>
                <tr>
                  <th>Bucket</th>
                  <th>Error rate</th>
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                {errorLength.map((row, idx) => (
                  <tr key={idx}>
                    <td>{row.length_bucket}</td>
                    <td>{formatNum(row.error_rate)}</td>
                    <td>{formatNum(row.total, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card">
            <h3>Error by Signal</h3>
            <table>
              <thead>
                <tr>
                  <th>Signal</th>
                  <th>Value</th>
                  <th>Error rate</th>
                </tr>
              </thead>
              <tbody>
                {errorSignal.map((row, idx) => (
                  <tr key={idx}>
                    <td>{row.signal}</td>
                    <td>{row.value}</td>
                    <td>{formatNum(row.error_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card">
            <h3>Error by Confidence</h3>
            <table>
              <thead>
                <tr>
                  <th>Bucket</th>
                  <th>Error rate</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {errorConfidence.map((row, idx) => (
                  <tr key={idx}>
                    <td>{row.confidence_bucket}</td>
                    <td>{formatNum(row.error_rate)}</td>
                    <td>{formatNum(row.accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {activeTab === "artifacts" && canShowTabs && (
        <section className="grid two">
          {images.map((img) => (
            <div className="card" key={img}>
              <h3>{img}</h3>
              <img src={`/api/images/${img}`} alt={img} />
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
