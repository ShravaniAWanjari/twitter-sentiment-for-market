import React, { useEffect, useMemo, useState } from "react";
import BacktestView from "./components/BacktestView";
import AnalyzeView from "./components/AnalyzeView";
import MarketGPTView from "./components/MarketGPTView";
import ExplainabilityCard from "./components/ExplainabilityCard";

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
      const res = await fetch("/api/explain", {
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
          { id: "overview", label: "Overview", requiresTrain: true },
          { id: "benchmarks", label: "Benchmarks", requiresTrain: true },
          { id: "errors", label: "Error Analysis", requiresTrain: true },
          { id: "backtest", label: "Backtest", requiresTrain: true },
          { id: "gpt", label: "Market GPT", requiresTrain: true },
          { id: "analyze", label: "Live Analysis", requiresTrain: false },
          { id: "artifacts", label: "Artifacts", requiresTrain: true },
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

      {activeTab === "overview" && (canShowTabs || false) && (
        <section className="grid three">
          <div className="card full-width">
            <h3>Leaderboard & Accuracy Overview</h3>
            <div className="leaderboard-container">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>F1 Macro</th>
                    <th>Accuracy</th>
                    <th>Sample Size</th>
                    <th>Error Rate</th>
                    <th>Avg Conf</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.map((row) => {
                    const sum = errorSummaries[row.model] || [];
                    const getVal = (key) => sum.find(r => r.key === key)?.value;
                    return (
                      <tr key={row.model}>
                        <td><strong>{row.model}</strong></td>
                        <td>{formatNum(row.f1_macro)}</td>
                        <td>{formatNum(row.accuracy)}</td>
                        <td>{getVal("total_samples")}</td>
                        <td style={{ color: "var(--red)" }}>{formatNum(getVal("error_rate"))}</td>
                        <td>{formatNum(getVal("avg_confidence"))}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          <div className="card">
            <h3>Latency Breakdown</h3>
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
          <div className="flex-between">
            <h2>Model Benchmarks</h2>
            <div className="flex-group">
              <button className="btn mini" onClick={loadResults}>Refresh Results</button>
            </div>
          </div>
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

          <div className="explain-search-section card" style={{ marginTop: "2rem", borderTop: "1px solid #333", paddingTop: "2rem" }}>
            <h3>Explain a Custom Prediction</h3>
            <p className="muted">Select a model and enter a headline to see its internal attribution.</p>
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
                style={{ flex: 1, padding: "0.5rem", background: "#111", border: "1px solid #333", color: "white", borderRadius: "4px" }}
              />
              <button
                className="btn primary"
                onClick={handleExplain}
                disabled={explaining || !explainText}
              >
                {explaining ? "Explaining..." : "Explain Prediction"}
              </button>
            </div>

            {explainResult && (
              <ExplainabilityCard result={explainResult} originalText={explainText} />
            )}
          </div>
        </section>
      )}

      {activeTab === "errors" && canShowTabs && (
        <div className="error-analysis-container">
          <div className="card filter-bar" style={{ marginBottom: "1rem", display: "flex", alignItems: "center", gap: "1rem" }}>
            <span>Analyzing results for:</span>
            <select
              value={analysisModel}
              onChange={(e) => setAnalysisModel(e.target.value)}
              className="model-select"
              style={{ background: "#111", color: "white", padding: "0.5rem", borderRadius: "4px" }}
            >
              {Object.keys(errorSummaries).map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <p className="muted" style={{ margin: 0 }}>Detailed metrics fluctuate per model architecture and pre-training.</p>
          </div>
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
        </div>
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
        <div className="artifacts-container">
          {Object.entries(groupedImages).map(([model, modelImages]) => (
            <section key={model} className="model-artifacts-section">
              <h2 className="model-heading">{model.toUpperCase()}</h2>
              <div className="grid two">
                {modelImages.map((img) => (
                  <div className="card" key={img}>
                    <h3>{img.split("_").slice(1).join(" ").replace(".png", "")}</h3>
                    <img src={`/api/images/${img}`} alt={img} />
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
