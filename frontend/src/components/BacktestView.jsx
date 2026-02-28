import { useState, useEffect } from "react";

const formatPct = (val) => {
    if (val === undefined || val === null) return "0.00%";
    return (val * 100).toFixed(2) + "%";
};

const formatCur = (val) => {
    if (val === undefined || val === null) return "$0.00";
    return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(val);
};

export default function BacktestView() {
    const [trainedModels, setTrainedModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState("");
    const [strategy, setStrategy] = useState("Momentum");
    const [threshold, setThreshold] = useState(0.5);
    const [initialBalance, setInitialBalance] = useState(10000);
    const [riskPerTrade, setRiskPerTrade] = useState(0.02);

    const [running, setRunning] = useState(false);
    const [result, setResult] = useState(null);
    const [samples, setSamples] = useState([]);

    useEffect(() => {
        // Fetch only models that have finished training
        fetch("/api/models/trained")
            .then(res => res.json())
            .then(data => {
                setTrainedModels(data.models || []);
                if (data.models && data.models.length > 0) {
                    setSelectedModel(data.models[0].model);
                }
            })
            .catch(err => console.error("Failed to fetch trained models", err));

        // Load latest result if any
        fetch("/api/backtest/latest")
            .then(res => res.ok ? res.json() : null)
            .then(data => {
                if (data) {
                    setResult(data);
                    fetchSamples(data.model);
                }
            })
            .catch(() => { });
    }, []);

    const fetchSamples = async (modelId) => {
        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: modelId, headlines: [
                        "BTC price surges as demand from ETFs reaches record levels",
                        "Market volatility spikes as traders weigh regulatory risks",
                        "Crypto mining activity declines following network upgrade"
                    ]
                }),
            });
            const data = await res.json();
            setSamples(Array.isArray(data) ? data : []);
        } catch (err) {
            console.error("Failed to fetch samples", err);
        }
    };

    const runBacktest = async () => {
        if (!selectedModel) return;
        setRunning(true);
        try {
            const res = await fetch("/api/backtest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: selectedModel,
                    strategy,
                    sentiment_threshold: threshold,
                    initial_balance: initialBalance,
                    risk_per_trade: riskPerTrade
                }),
            });
            const data = await res.json();
            setResult(data);
            fetchSamples(selectedModel);
        } catch (err) {
            alert("Backtest failed: " + err.message);
        } finally {
            setRunning(false);
        }
    };

    const renderEquityChart = (data, color, baselineColor = "#444") => {
        if (!data || !data.dates || data.dates.length < 2) return null;

        const baseline = (data.baseline || []).filter(v => v !== null && !isNaN(v));
        const gated = (data.gated || []).filter(v => v !== null && !isNaN(v));

        if (baseline.length < 2 && gated.length < 2) return null;

        const combined = [...baseline, ...gated];
        const min = combined.length > 0 ? Math.min(...combined) : 0;
        const max = combined.length > 0 ? Math.max(...combined) : 100;
        const range = (max - min) || 1;

        const getPoints = (series) => {
            if (!series || series.length < 2) return "";
            return series.map((v, i) => {
                const x = (i / (series.length - 1)) * 100;
                const y = 100 - ((v - min) / range) * 100;
                return `${x},${y}`;
            }).join(" ");
        };

        const baselinePoints = getPoints(baseline);
        const gatedPoints = getPoints(gated);

        return (
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="chart-svg">
                {baselinePoints && (
                    <polyline fill="none" stroke={baselineColor} strokeWidth="0.5" strokeDasharray="1,1" points={baselinePoints} />
                )}
                {gatedPoints && (
                    <polyline fill="none" stroke={color} strokeWidth="1.8" points={gatedPoints} />
                )}
            </svg>
        );
    };

    return (
        <div className="pro-dashboard">
            <div className="dashboard-sidebar">
                <div className="sidebar-header">
                    <h3>Simulation Engine</h3>
                    <p className="muted">Jan 2025 Tick Data Grounded</p>
                </div>

                <div className="config-group">
                    <label>Selected Model</label>
                    <select
                        disabled={running}
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                    >
                        {trainedModels.length === 0 && <option>No trained models yet</option>}
                        {trainedModels.map(m => (
                            <option key={m.model} value={m.model}>
                                {m.model} {m.has_weights ? "✓" : "(no weights)"}
                            </option>
                        ))}
                    </select>
                </div>

                <div className="config-group">
                    <label>Core Strategy</label>
                    <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                        <option value="Momentum">Trend Momentum</option>
                        <option value="RSI">Mean Reversion (RSI)</option>
                    </select>
                </div>

                <div className="config-group">
                    <label>Initial Balance: {formatCur(initialBalance)}</label>
                    <input
                        type="range" min="1000" max="100000" step="1000"
                        value={initialBalance} onChange={(e) => setInitialBalance(parseInt(e.target.value))}
                    />
                </div>

                <div className="config-group">
                    <label>Risk Per Trade: {(riskPerTrade * 100).toFixed(1)}%</label>
                    <input
                        type="range" min="0.005" max="0.1" step="0.005"
                        value={riskPerTrade} onChange={(e) => setRiskPerTrade(parseFloat(e.target.value))}
                    />
                </div>

                <div className="config-group">
                    <label>Sentiment Threshold: {threshold}</label>
                    <input
                        type="range" min="0" max="1" step="0.05"
                        value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    />
                    <p className="tiny muted">Gated entry if NLP score &lt; {threshold}</p>
                </div>

                <button
                    className="btn primary full-width"
                    onClick={runBacktest}
                    disabled={running || !selectedModel}
                >
                    {running ? "Processing Ticks..." : "Launch Backtest"}
                </button>
            </div>

            <main className="dashboard-content">
                {!result ? (
                    <div className="empty-state">
                        <div className="pulse-icon">📊</div>
                        <h2>Ready for Simulation</h2>
                        <p>Configure your model and capital parameters to begin the Jan 2025 tick-data analysis.</p>
                    </div>
                ) : (
                    <div className="dashboard-grid">
                        <div className="stats-header card">
                            <div className="stat-tile">
                                <span>Total Return</span>
                                <strong className={(result.metrics?.gated?.total_return || 0) >= 0 ? "trend-up" : "trend-down"}>
                                    {formatPct(result.metrics?.gated?.total_return)}
                                </strong>
                            </div>
                            <div className="stat-tile">
                                <span>Sharpe Ratio</span>
                                <strong>{result.metrics?.gated?.sharpe_ratio?.toFixed(2) || "0.00"}</strong>
                            </div>
                            <div className="stat-tile">
                                <span>Win Rate</span>
                                <strong>{formatPct(result.metrics?.gated?.win_rate)}</strong>
                            </div>
                            <div className="stat-tile">
                                <span>Max Drawdown</span>
                                <strong className="trend-down">{formatPct(result.metrics?.gated?.max_drawdown)}</strong>
                            </div>
                            <div className="stat-tile">
                                <span>Profit Factor</span>
                                <strong>{result.metrics?.gated?.profit_factor?.toFixed(2) || "0.00"}</strong>
                            </div>
                            <div className="stat-tile">
                                <span>Final Balance</span>
                                <strong>{formatCur(result.metrics?.gated?.final_balance)}</strong>
                            </div>
                        </div>

                        <div className="hero-chart card">
                            <div className="flex-between">
                                <h3>Cumulative Equity Curve</h3>
                                <div className="legend">
                                    <span className="dot baseline"></span> Baseline
                                    <span className="dot gated"></span> NLP Augmented
                                </div>
                            </div>
                            <div className="chart-container">
                                {renderEquityChart(result.equity_curve, "var(--primary)")}
                            </div>
                        </div>

                        <div className="side-cards">
                            <section className="card">
                                <h3>NLP Impact Over Samples</h3>
                                <div className="headline-scroll">
                                    {samples.map((h, i) => (
                                        <div key={i} className="mini-analysis">
                                            <p>{h.headline}</p>
                                            <div className="flex-between">
                                                <span className={`pill mini ${h.sentiment.toLowerCase()}`}>{h.sentiment}</span>
                                                <span className="tiny muted">Conf: {(h.confidence * 100).toFixed(0)}%</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </section>

                            <section className="card">
                                <h3>Drawdown Profile</h3>
                                <div className="mini-chart" style={{ height: "100px" }}>
                                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="chart-svg">
                                        {(() => {
                                            const ds = result.metrics?.gated?.drawdown_series || [0];
                                            const maxDD = Math.max(...ds.map(v => Math.abs(v))) || 0.01;
                                            const points = ds.map((v, i) => {
                                                const x = (i / (ds.length - 1)) * 100;
                                                const y = (Math.abs(v) / maxDD) * 100;
                                                return `${x},${y}`;
                                            }).join(" ");
                                            return (
                                                <polyline
                                                    fill="rgba(255, 107, 107, 0.2)"
                                                    stroke="#ff6b6b"
                                                    strokeWidth="1.5"
                                                    points={`0,0 ${points} 100,0`}
                                                />
                                            );
                                        })()}
                                    </svg>
                                </div>
                                <p className="tiny muted text-center">Auto-scaled relative to max drawdown (Jan 2025)</p>
                            </section>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}
