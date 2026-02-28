import React, { useEffect, useMemo, useState } from "react";
import BacktestView from "./components/BacktestView";
import AnalyzeView from "./components/AnalyzeView";
import MarketGPTView from "./components/MarketGPTView";
import ExplainabilityCard from "./components/ExplainabilityCard";
import SessionAnalysis from "./components/SessionAnalysis";

async function fetchJson(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Request failed");
  }
  return res.json();
}

function formatNum(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (Number.isNaN(num)) return value;
  return Number.isInteger(num) ? `${num}` : num.toFixed(digits);
}

export default function App() {
  const [config, setConfig] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);
  const [configFailed, setConfigFailed] = useState(false);
  const [status, setStatus] = useState(null);
  const [trainedModels, setTrainedModels] = useState([]);
  const [benchmark, setBenchmark] = useState([]);
  const [errorSummaries, setErrorSummaries] = useState({});
  const [analysisModel, setAnalysisModel] = useState("");
  const [errorLength, setErrorLength] = useState([]);
  const [errorSignal, setErrorSignal] = useState([]);
  const [errorConfidence, setErrorConfidence] = useState([]);
  const [images, setImages] = useState([]);
  const [activeTab, setActiveTab] = useState("setup");
  const [explainModel, setExplainModel] = useState("modernbert");
  const [explainText, setExplainText] = useState("");
  const [explaining, setExplaining] = useState(false);
  const [explainResult, setExplainResult] = useState(null);
  const [clearing, setClearing] = useState(false);
  const [clearWeights, setClearWeights] = useState(false);

  const handleExplain = async () => {
    setExplaining(true);
    try {
      // Call multi-method explain for richer results
      const res = await fetch("/api/explain/multi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: explainModel, text: explainText }),
      });
      const data = await res.json();
      setExplainResult(data);
    } catch (err) {
      console.error("Explanation failed", err);
    } finally {
      setExplaining(false);
    }
  };

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
    fetchTrainedModels();
  }, []);

  const fetchTrainedModels = async () => {
    try {
      const data = await fetchJson("/api/models/trained");
      setTrainedModels(data?.models || []);
    } catch (err) {
      console.error("Failed to fetch trained models", err);
    }
  };

  const [refreshing, setRefreshing] = useState(false);

  const refreshStatus = async (manual = false) => {
    if (manual) setRefreshing(true);
    try {
      const data = await fetchJson("/api/status");
      setStatus(data);
    } catch (err) {
      console.error("Refresh failed:", err);
      // If server is unreachable, we don't want to show "running" forever
      setStatus(prev => prev?.status === "running" ? { ...prev, status: "error" } : prev);
    } finally {
      if (manual) setRefreshing(false);
    }
  };

  useEffect(() => {
    if (!status || (status.status !== "running" && status.status !== "clearing")) return;
    const timer = setInterval(() => refreshStatus(false), status.status === "clearing" ? 1000 : 3000);
    return () => clearInterval(timer);
  }, [status]);

  const loadResults = async () => {
    try {
      const [bench, errSums] = await Promise.all([
        fetchJson("/api/benchmark"),
        fetchJson("/api/errors/summaries/all")
      ]);
      setBenchmark(bench?.rows || []);
      setErrorSummaries(errSums || {});

      const models = Object.keys(errSums || {});
      if (models.length > 0 && !analysisModel) {
        setAnalysisModel(models[0]);
      }

      const imgs = await fetchJson("/api/images");
      setImages(imgs.images || []);
    } catch (err) {
      console.warn("Failed to load generic results", err);
    }
  };

  const loadAnalysisData = async (model) => {
    if (!model) return;
    try {
      const [len, sig, conf] = await Promise.all([
        fetchJson(`/api/errors/error_by_length?model=${model}`),
        fetchJson(`/api/errors/error_by_signal?model=${model}`),
        fetchJson(`/api/errors/error_by_confidence?model=${model}`),
      ]);
      setErrorLength(len?.rows || []);
      setErrorSignal(sig?.rows || []);
      setErrorConfidence(conf?.rows || []);
    } catch (err) {
      console.warn("Failed to load analysis for", model, err);
    }
  };

  useEffect(() => {
    if (analysisModel) {
      loadAnalysisData(analysisModel);
    }
  }, [analysisModel]);

  useEffect(() => {
    if (status?.status === "completed") {
      loadResults();
      setActiveTab("overview");
    }
    // If we transition from clearing to idle, force a reload
    if (clearing && status?.status === "idle") {
      window.location.reload();
    }
  }, [status, clearing]);

  const canShowTabs = status?.status === "completed";

  const leaderboard = useMemo(() => {
    return [...benchmark].sort((a, b) => (b.f1_macro ?? 0) - (a.f1_macro ?? 0));
  }, [benchmark]);

  const startTraining = async () => {
    try {
      await fetchJson("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ models: selectedModels }),
      });
      refreshStatus();
    } catch (err) {
      alert("Failed to start training: " + err.message);
    }
  };

  const allSelectedTrained = useMemo(() => {
    if (!selectedModels.length) return false;
    return selectedModels.every(m => trainedModels.find(t => t.model === m)?.has_weights);
  }, [selectedModels, trainedModels]);

  const allSelectedCached = useMemo(() => {
    if (!selectedModels.length) return false;
    return selectedModels.every(m => {
      const t = trainedModels.find(tm => tm.model === m);
      return t?.has_weights && t?.has_benchmark && t?.has_analysis;
    });
  }, [selectedModels, trainedModels]);

  const total_steps = status?.total_steps ?? 0;
  const current_step = status?.current_step ?? 0;
  const intra = status?.intra_step_progress ?? 0;

  const progressPercent = total_steps > 0
    ? Math.round(((current_step + intra) / total_steps) * 100)
    : 0;

  const groupedImages = useMemo(() => {
    const groups = {};
    images.forEach((img) => {
      // Expecting format like "modelname_rest_of_file.png"
      const name = img.split("_")[0];
      if (!groups[name]) groups[name] = [];
      groups[name].push(img);
    });
    return groups;
  }, [images]);


  const clearSession = async () => {
    const msg = clearWeights
      ? "Are you sure you want to clear EVERYTHING, including trained model weights?"
      : "Are you sure you want to clear benchmarks and analysis? Model weights will be KEPT.";

    if (!window.confirm(msg)) return;

    setClearing(true);
    try {
      await fetchJson("/api/clear-session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clear_models: clearWeights })
      });
      // Don't reload yet! The background task is just starting.
      // polling will detect the "clearing" status and eventually "idle"
      refreshStatus();
      setClearing(false);
    } catch (err) {
      alert("Failed to clear session: " + err.message);
      setClearing(false);
    }
  };

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
          <div className="btn-group" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <label className="checkbox-label" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.9rem' }}>
              <input
                type="checkbox"
                checked={clearWeights}
                onChange={(e) => setClearWeights(e.target.checked)}
              />
              Clear weights
            </label>
            <button className="btn refresh-btn" onClick={() => refreshStatus(true)} disabled={refreshing}>
              {refreshing ? "Refreshing..." : "Refresh"}
            </button>
            <button className="btn danger" onClick={clearSession} disabled={status?.status === "running" || status?.status === "clearing" || clearing}>
              {clearing || status?.status === "clearing" ? "Clearing..." : "Clear Session"}
            </button>
          </div>
        </div>
      </header>

      {status && (status.status === "running" || status.status === "failed") && (
        <section className="progress-card">
          <div className="progress-inner">
            <span>
              {status?.current_model?.startsWith("Analyzing") || status?.current_model?.startsWith("Benchmarking")
                ? `${status.current_model}...`
                : status?.current_model
                  ? `Training ${status.current_model}...`
                  : "Processing..."}
            </span>
            <strong>{progressPercent}% complete</strong>
          </div>
          <div className="progress-track">
            <div className="progress-bar" style={{ width: `${progressPercent}%` }} />
          </div>
          <div className="log-box mini">
            {(status?.logs || []).slice(-100).map((line, idx) => (
              <div key={idx}>{line}</div>
            ))}
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
          <div style={{ marginBottom: '1rem' }}>
            <button className="btn primary" onClick={startTraining} disabled={!selectedModels.length || status?.status === "running"}>
              {allSelectedCached ? "Launch Dashboard (Instant)" : allSelectedTrained ? "Re-run Benchmark (Skip Training)" : "Run Training + Benchmark"}
            </button>
          </div>
          {allSelectedCached ? (
            <p className="success" style={{ padding: '0.5rem', background: 'rgba(74, 222, 128, 0.1)', border: '1px solid var(--success)', borderRadius: '4px', fontSize: '0.9rem' }}>
              <strong>Notice:</strong> All models, benchmarks, and analysis results are cached.
              Clicking the button will jump instantly to the presenting tabs.
            </p>
          ) : allSelectedTrained && (
            <p className="success" style={{ padding: '0.5rem', background: 'rgba(74, 222, 128, 0.1)', border: '1px solid var(--success)', borderRadius: '4px', fontSize: '0.9rem' }}>
              <strong>Notice:</strong> All selected models are already trained.
              Clicking the button will skip training and proceed straight to benchmarking.
              To retrain, check "Clear weights" and click "Clear Session" first.
            </p>
          )}
          {status?.status === "failed" && <p className="error">Job failed: {status.error}</p>}
        </div>
      </section>

      <nav className="tabs">
        {[
          { id: "benchmarks", label: "Benchmarks", requiresTrain: true },
          { id: "backtest", label: "Backtest", requiresTrain: true },
          { id: "gpt", label: "Market GPT", requiresTrain: true },
          { id: "analyze", label: "Live Analysis", requiresTrain: false },
          { id: "artifacts", label: "Session Analysis", requiresTrain: true },
        ].map((tab) => {
          const locked = tab.requiresTrain && !canShowTabs;
          return (
            <button
              key={tab.id}
              className={`${activeTab === tab.id ? "active" : ""} ${locked ? "tab-locked" : ""}`}
              onClick={() => !locked && setActiveTab(tab.id)}
              disabled={locked}
            >
              {tab.label}
            </button>
          );
        })}
      </nav>



      {activeTab === "benchmarks" && canShowTabs && (
        <section className="card">
          <div className="flex-between" style={{ marginBottom: "1.5rem" }}>
            <h2>Model Benchmarks</h2>
            <button className="btn mini" onClick={loadResults}>Refresh</button>
          </div>

          {/* ── Visual Performance Charts ── */}
          {benchmark.length > 0 && (() => {
            const COLORS = ["#818cf8", "#51cf66", "#ff9f43", "#ff6b6b", "#34d399"];
            const models = benchmark.map(r => r.model);
            const f1s = benchmark.map(r => Number(r.f1_macro) || 0);
            const accs = benchmark.map(r => Number(r.accuracy) || 0);
            const lats = benchmark.map(r => Number(r.latency_ms_per_tweet) || 0);
            const maxF1 = Math.max(...f1s, 0.001);
            const maxAcc = Math.max(...accs, 0.001);
            const maxLat = Math.max(...lats, 0.001);

            const BAR_H = 26, GAP = 8, LABEL_W = 96, BAR_MAX_W = 220;
            const chartH = (BAR_H + GAP) * models.length + 20;

            const BarChart = ({ values, max, colors, title, unit = "" }) => (
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: "0.78rem", color: "#8a8fb5", marginBottom: "0.5rem", fontWeight: 600 }}>{title}</p>
                <svg viewBox={`0 0 ${LABEL_W + BAR_MAX_W + 60} ${chartH}`}
                  style={{ width: "100%", height: chartH }}>
                  {values.map((v, i) => {
                    const bW = (v / max) * BAR_MAX_W;
                    const y = 10 + i * (BAR_H + GAP);
                    return (
                      <g key={i}>
                        <text x={LABEL_W - 6} y={y + BAR_H * 0.68}
                          textAnchor="end" fontSize="11" fill="#c4c8f0">
                          {models[i]}
                        </text>
                        <rect x={LABEL_W} y={y} width={bW} height={BAR_H}
                          rx="4" fill={colors[i % colors.length]} opacity="0.8" />
                        <text x={LABEL_W + bW + 6} y={y + BAR_H * 0.68}
                          fontSize="10" fill={colors[i % colors.length]} fontWeight="700">
                          {v.toFixed(3)}{unit}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            );

            return (
              <div style={{ marginBottom: "2rem" }}>
                <div style={{ display: "flex", gap: "1.5rem", marginBottom: "1rem" }}>
                  <BarChart values={f1s} max={maxF1} colors={COLORS} title="F1 Macro Score (higher = better)" />
                  <BarChart values={accs} max={maxAcc} colors={COLORS} title="Accuracy (higher = better)" />
                  <BarChart values={lats} max={maxLat} colors={COLORS} title="Latency ms/tweet (lower = better)" unit="ms" />
                </div>
              </div>
            );
          })()}

          {/* ── Raw data table (collapsed view) ── */}
          <details style={{ marginBottom: "1.5rem" }}>
            <summary style={{ cursor: "pointer", fontSize: "0.82rem", color: "#8a8fb5", userSelect: "none" }}>
              Show raw benchmark data
            </summary>
            <table style={{ marginTop: "0.75rem", marginBottom: "2rem" }}>
              <thead>
                <tr>
                  {benchmark[0] &&
                    Object.keys(benchmark[0]).filter(k => k !== "slang_accuracy").map((key) => <th key={key}>{key}</th>)}
                </tr>
              </thead>
              <tbody>
                {benchmark.map((row) => (
                  <tr key={row.model}>
                    {Object.keys(row).filter(k => k !== "slang_accuracy").map((key) => (
                      <td key={key}>{formatNum(row[key])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Integrated Error Analysis */}
            <div className="error-analysis-integrated" style={{ borderTop: "1px solid #333", paddingTop: "1.5rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.5rem" }}>
                <h3 style={{ margin: 0, fontSize: "1rem" }}>Deep Error Attribution</h3>
                <select
                  value={analysisModel}
                  onChange={(e) => setAnalysisModel(e.target.value)}
                  style={{ background: "#111", color: "white", padding: "0.3rem 0.6rem", borderRadius: "4px", fontSize: "0.85rem", border: "1px solid #333" }}
                >
                  {Object.keys(errorSummaries).map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>

              <div className="grid three" style={{ gap: "1rem" }}>
                <div className="card mini">
                  <h4 style={{ fontSize: "0.8rem", color: "#8a8fb5", marginBottom: "0.5rem" }}>Error by Length</h4>
                  <table className="tiny">
                    <thead>
                      <tr><th>Bucket</th><th>Error</th><th>N</th></tr>
                    </thead>
                    <tbody>
                      {errorLength.map((row, idx) => (
                        <tr key={idx}>
                          <td>{row.length_bucket}</td>
                          <td style={{ color: "var(--red)" }}>{formatNum(row.error_rate)}</td>
                          <td>{row.total}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="card mini">
                  <h4 style={{ fontSize: "0.8rem", color: "#8a8fb5", marginBottom: "0.5rem" }}>Error by Signal</h4>
                  <table className="tiny">
                    <thead>
                      <tr><th>Signal</th><th>Val</th><th>Error</th></tr>
                    </thead>
                    <tbody>
                      {errorSignal.map((row, idx) => (
                        <tr key={idx}>
                          <td style={{ fontSize: "0.7rem" }}>{row.signal}</td>
                          <td>{row.value}</td>
                          <td style={{ color: "var(--red)" }}>{formatNum(row.error_rate)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="card mini">
                  <h4 style={{ fontSize: "0.8rem", color: "#8a8fb5", marginBottom: "0.5rem" }}>Error by Confidence</h4>
                  <table className="tiny">
                    <thead>
                      <tr><th>Bucket</th><th>Error</th><th>Acc</th></tr>
                    </thead>
                    <tbody>
                      {errorConfidence.map((row, idx) => (
                        <tr key={idx}>
                          <td>{row.confidence_bucket}</td>
                          <td style={{ color: "var(--red)" }}>{formatNum(row.error_rate)}</td>
                          <td>{formatNum(row.accuracy)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </details>

          {/* ── Explainability section ── */}
          <div className="explain-search-section card" style={{ marginTop: "2rem", borderTop: "1px solid #333", paddingTop: "2rem" }}>
            <h3>Explain a Custom Prediction</h3>
            <p className="muted">Select a model and enter a headline — runs <strong>3 explainability methods simultaneously</strong> (Occlusion, Integrated Gradients, LIME).</p>
            <div className="flex-group" style={{ marginBottom: "1rem" }}>
              <select
                value={explainModel}
                onChange={(e) => setExplainModel(e.target.value)}
                style={{ background: "#111", color: "white", padding: "0.5rem", borderRadius: "4px" }}
              >
                {selectedModels.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
              <input
                type="text"
                placeholder="Enter a crypto headline..."
                value={explainText}
                onChange={(e) => setExplainText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !explaining && explainText && handleExplain()}
                style={{ flex: 1, padding: "0.5rem", background: "#111", border: "1px solid #333", color: "white", borderRadius: "4px" }}
              />
              <button
                className="btn primary"
                onClick={handleExplain}
                disabled={explaining || !explainText}
              >
                {explaining ? "Analysing with 3 methods…" : "⚡ Explain Prediction"}
              </button>
            </div>

            {explainResult && (
              <ExplainabilityCard result={explainResult} originalText={explainText} />
            )}
          </div>
        </section>
      )}


      {activeTab === "backtest" && (
        <BacktestView models={config?.models || DEFAULT_MODELS} />
      )}

      {activeTab === "gpt" && (
        <MarketGPTView />
      )}

      {activeTab === "analyze" && (
        <AnalyzeView models={config?.models || DEFAULT_MODELS} />
      )}

      {activeTab === "artifacts" && canShowTabs && (
        <SessionAnalysis
          benchmark={benchmark}
          errorSummaries={errorSummaries}
          leaderboard={leaderboard}
        />
      )}
    </div>
  );
}
