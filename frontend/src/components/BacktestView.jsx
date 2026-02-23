import { useState, useEffect } from "react";

export default function BacktestView({ models }) {
    const [selectedModel, setSelectedModel] = useState(models[0] || "modernbert");
    const [strategy, setStrategy] = useState("Momentum");
    const [threshold, setThreshold] = useState(0.5);
    const [running, setRunning] = useState(false);
    const [result, setResult] = useState(null);

    useEffect(() => {
        fetch("/api/backtest/latest")
            .then(res => res.ok ? res.json() : null)
            .then(data => setResult(data))
            .catch(() => { });
    }, []);

    const runBacktest = async () => {
        setRunning(true);
        try {
            const res = await fetch("/api/backtest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: selectedModel, strategy, threshold }),
            });
            const data = await res.json();
            setResult(data);
        } catch (err) {
            alert("Backtest failed: " + err.message);
        } finally {
            setRunning(false);
        }
    };

    const formatPct = (val) => {
        try {
            return (val * 100).toFixed(2) + "%";
        } catch (e) {
            return "0.00%";
        }
    };

    const [samples, setSamples] = useState([]);
    useEffect(() => {
        if (result) {
            console.log("BacktestView: Fetching samples for result", result);
            fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: selectedModel, headlines: [
                        "BTC price surges as demand from ETFs reaches record levels",
                        "Market volatility spikes as traders weigh regulatory risks",
                        "Crypto mining activity declines following network upgrade"
                    ]
                }),
            })
                .then(res => res.ok ? res.json() : [])
                .then(data => {
                    console.log("BacktestView: Received samples", data);
                    setSamples(Array.isArray(data) ? data : []);
                })
                .catch(err => console.error("BacktestView: Samples fetch failed", err));
        }
    }, [result, selectedModel]);

    try {
        console.log("BacktestView: Rendering component", { models, selectedModel, running, resultSamples: samples.length });
        if (!models || !Array.isArray(models)) {
            return <div className="card">Error: Models list is missing.</div>;
        }

        return (
            <div className="backtest-view">
                <section className="card">
                    <h2>Backtest Configuration</h2>
                    <div className="grid three">
                        <div>
                            <label>Model</label>
                            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                                {models.map(m => <option key={m} value={m}>{m}</option>)}
                            </select>
                        </div>
                        <div>
                            <label>Strategy</label>
                            <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                                <option value="Momentum">Momentum</option>
                                <option value="RSI">RSI (Mean Reversion)</option>
                            </select>
                        </div>
                        <div>
                            <label>Sentiment Threshold: {threshold}</label>
                            <input
                                type="range" min="0" max="1" step="0.1"
                                value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))}
                            />
                        </div>
                    </div>
                    <button className="btn primary" onClick={runBacktest} disabled={running}>
                        {running ? "Running Backtest..." : "Run Backtest"}
                    </button>
                </section>

                {result && result.metrics && (
                    <div className="result-container">
                        <div className="grid two">
                            <section className="card">
                                <h3>Strategy Metrics</h3>
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Metric</th>
                                            <th>Baseline</th>
                                            <th>NLP Gated</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>Total Return</td>
                                            <td>{formatPct(result.metrics?.baseline?.total_return || 0)}</td>
                                            <td className="highlight-success">{formatPct(result.metrics?.gated?.total_return || 0)}</td>
                                        </tr>
                                        <tr>
                                            <td>Sharpe Ratio</td>
                                            <td>{(result.metrics?.baseline?.sharpe_ratio || 0).toFixed(2)}</td>
                                            <td>{(result.metrics?.gated?.sharpe_ratio || 0).toFixed(2)}</td>
                                        </tr>
                                        <tr>
                                            <td>Max Drawdown</td>
                                            <td>{formatPct(result.metrics?.baseline?.max_drawdown || 0)}</td>
                                            <td>{formatPct(result.metrics?.gated?.max_drawdown || 0)}</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </section>
                            <section className="card">
                                <h3>Equity Curve</h3>
                                <div className="chart-placeholder" style={{ height: "200px", background: "var(--bg-card)", border: "1px dashed #444", borderRadius: "8px", display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
                                    <svg width="100%" height="100%" viewBox="0 0 100 40" preserveAspectRatio="none">
                                        {(() => {
                                            const baseline = result?.equity_curve?.baseline || [];
                                            const gated = result?.equity_curve?.gated || [];
                                            const baselineMax = baseline.length > 0 ? baseline.reduce((a, b) => Math.max(a, b), 1) : 1;
                                            const gatedMax = gated.length > 0 ? gated.reduce((a, b) => Math.max(a, b), 1) : 1;

                                            return (
                                                <>
                                                    {baseline.length > 0 && (
                                                        <polyline
                                                            fill="none"
                                                            stroke="#666"
                                                            strokeWidth="0.5"
                                                            points={baseline.map((v, i) =>
                                                                `${(i / baseline.length) * 100},${40 - (v / baselineMax) * 20}`
                                                            ).join(" ")}
                                                        />
                                                    )}
                                                    {gated.length > 0 && (
                                                        <polyline
                                                            fill="none"
                                                            stroke="#4dabf7"
                                                            strokeWidth="1"
                                                            points={gated.map((v, i) =>
                                                                `${(i / gated.length) * 100},${40 - (v / gatedMax) * 20}`
                                                            ).join(" ")}
                                                        />
                                                    )}
                                                </>
                                            );
                                        })()}
                                    </svg>
                                    <div style={{ position: "absolute", top: 10, right: 10, fontSize: "10px" }}>
                                        <span style={{ color: "#666" }}>-- Baseline</span><br />
                                        <span style={{ color: "#4dabf7" }}>-- NLP Gated</span>
                                    </div>
                                </div>
                            </section>
                        </div>

                        <section className="card">
                            <h3>Explainability for Sample Headlines</h3>
                            <div className="headline-list">
                                {Array.isArray(samples) && samples.map((h, i) => (
                                    <div key={i} className="headline-item" style={{ borderBottom: "1px solid #222", paddingBottom: "1rem", marginBottom: "1rem" }}>
                                        <div className="flex-between">
                                            <strong>{h?.headline || "Untitled Headline"}</strong>
                                            <span className={`badge ${(h?.sentiment || "neutral").toLowerCase()}`}>{h?.sentiment || "Neutral"}</span>
                                        </div>
                                        <div className="token-explain">
                                            {Array.isArray(h?.tokens) && h.tokens.map((t, ti) => (
                                                <span key={ti} className="token" style={{ backgroundColor: `rgba(77, 171, 247, ${(t?.score || 0) * 0.5})` }}>
                                                    {(t?.token || "").replace("Ġ", " ")}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </section>
                    </div>
                )}
            </div>
        );
    } catch (renderError) {
        console.error("BacktestView: Render error", renderError);
        return (
            <div className="card error-box" style={{ padding: "2rem", textAlign: "center", color: "#ff8f8f" }}>
                <h3>Something went wrong rendering the backtest results.</h3>
                <p>{renderError.message}</p>
                <button className="btn" onClick={() => window.location.reload()}>Reload Page</button>
            </div>
        );
    }
}
