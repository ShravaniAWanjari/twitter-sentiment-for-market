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
    const [showDrawdownModal, setShowDrawdownModal] = useState(false);

    // Backtest headline analysis state
    const [hlSamples, setHlSamples] = useState([]);
    const [hlLoading, setHlLoading] = useState(false);
    const [hlError, setHlError] = useState(null);
    const [selectedHl, setSelectedHl] = useState(null);

    useEffect(() => {
        // Fetch only models that have finished training and auto-select the best one
        fetch("/api/models/trained")
            .then(res => res.json())
            .then(data => {
                const models = data.models || [];
                setTrainedModels(models);
                if (models.length > 0) {
                    // Sort by f1_macro descending to get the best model
                    const sorted = [...models].sort((a, b) => (b.f1_macro || 0) - (a.f1_macro || 0));
                    setSelectedModel(sorted[0].model);
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
        setHlSamples([]);
        setSelectedHl(null);
        setHlError(null);
        try {
            const res = await fetch("/api/backtest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model: selectedModel,
                    strategy,
                    sentiment_threshold: threshold,
                    initial_balance: initialBalance,
                    risk_per_trade: riskPerTrade,
                    from_date: "2024-01-01",
                    to_date: new Date(Date.now() - 864e5).toISOString().slice(0, 10)
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

    const fetchHeadlineSamples = async () => {
        if (!result) return;
        setHlLoading(true);
        setHlError(null);
        setHlSamples([]);
        setSelectedHl(null);
        try {
            const res = await fetch("/api/backtest/headline-samples", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    from_date: result.from_date || "2024-01-01",
                    to_date: result.to_date || new Date(Date.now() - 864e5).toISOString().slice(0, 10),
                    model: result.model || selectedModel,
                }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${res.status}`);
            }
            const data = await res.json();
            setHlSamples(data.headlines || []);
            if (data.headlines?.length > 0) setSelectedHl(data.headlines[0]);
        } catch (err) {
            setHlError(err.message);
        } finally {
            setHlLoading(false);
        }
    };

    const renderEquityChart = (data) => {
        if (!data || !data.dates || data.dates.length < 2) return null;

        const baseline = data.baseline || [];
        const gated = data.gated || [];
        if (baseline.length < 2 && gated.length < 2) return null;

        // Chart dimensions with padding for axes
        const W = 800, H = 320;
        const PAD = { top: 20, right: 24, bottom: 40, left: 72 };
        const chartW = W - PAD.left - PAD.right;
        const chartH = H - PAD.top - PAD.bottom;

        const combined = [...baseline, ...gated].filter(v => v != null && !isNaN(v));
        const minV = Math.min(...combined);
        const maxV = Math.max(...combined);
        const range = (maxV - minV) || 1;

        // Pad range by 8% top and bottom for breathing room
        const vMin = minV - range * 0.08;
        const vMax = maxV + range * 0.08;
        const vRange = vMax - vMin;

        const toX = (i, len) => PAD.left + (i / (len - 1)) * chartW;
        const toY = (v) => PAD.top + chartH - ((v - vMin) / vRange) * chartH;

        const makePath = (series) =>
            series.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i, series.length).toFixed(1)},${toY(v).toFixed(1)}`).join(" ");

        const makeArea = (series) => {
            const line = makePath(series);
            const lastX = toX(series.length - 1, series.length).toFixed(1);
            const firstX = toX(0, series.length).toFixed(1);
            const baseY = (PAD.top + chartH).toFixed(1);
            return `${line} L${lastX},${baseY} L${firstX},${baseY} Z`;
        };

        // Y-axis grid: 5 ticks
        const yTicks = Array.from({ length: 5 }, (_, i) => {
            const val = vMin + (vRange * i) / 4;
            const y = toY(val);
            return { val, y };
        }).reverse();

        // X-axis ticks: pick ~6 evenly spaced dates
        const xTickCount = 6;
        const xTicks = Array.from({ length: xTickCount }, (_, i) => {
            const idx = Math.round((i / (xTickCount - 1)) * (data.dates.length - 1));
            return { label: data.dates[idx]?.slice(0, 10) || "", x: toX(idx, data.dates.length) };
        });

        // Performance delta
        const lastBase = baseline[baseline.length - 1];
        const lastGated = gated[gated.length - 1];
        const deltaRaw = lastBase && lastGated ? ((lastGated - lastBase) / lastBase) * 100 : 0;
        const deltaColor = deltaRaw >= 0 ? "#51cf66" : "#ff6b6b";
        const deltaSign = deltaRaw >= 0 ? "+" : "";

        const formatK = (v) => {
            if (v == null || isNaN(v)) return "$0";
            if (Math.abs(v) >= 1000) return `$${(v / 1000).toFixed(1)}k`;
            return `$${v.toFixed(0)}`;
        };

        return (
            <div style={{ position: "relative" }}>
                {/* Delta badge */}
                <div style={{
                    position: "absolute", top: 0, right: 0,
                    background: deltaRaw >= 0 ? "rgba(81,207,102,0.12)" : "rgba(255,107,107,0.12)",
                    border: `1px solid ${deltaColor}22`,
                    borderRadius: "8px", padding: "4px 10px",
                    fontSize: "0.78rem", color: deltaColor, fontWeight: 700
                }}>
                    NLP Alpha: {deltaSign}{deltaRaw.toFixed(2)}%
                </div>

                <svg
                    viewBox={`0 0 ${W} ${H}`}
                    style={{ width: "100%", height: "100%", overflow: "visible" }}
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <defs>
                        {/* Baseline gradient — muted gray */}
                        <linearGradient id="gradBase" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#888" stopOpacity="0.25" />
                            <stop offset="100%" stopColor="#888" stopOpacity="0.02" />
                        </linearGradient>
                        {/* NLP gradient — vivid indigo */}
                        <linearGradient id="gradGated" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.35" />
                            <stop offset="100%" stopColor="#6366f1" stopOpacity="0.02" />
                        </linearGradient>
                        {/* Clip to chart area */}
                        <clipPath id="chartClip">
                            <rect x={PAD.left} y={PAD.top} width={chartW} height={chartH} />
                        </clipPath>
                    </defs>

                    {/* ── Grid lines & Y-axis labels ── */}
                    {yTicks.map(({ val, y }, i) => (
                        <g key={i}>
                            <line
                                x1={PAD.left} y1={y} x2={PAD.left + chartW} y2={y}
                                stroke="rgba(255,255,255,0.05)" strokeWidth="1"
                                strokeDasharray={i === 0 ? "0" : "4,4"}
                            />
                            <text
                                x={PAD.left - 8} y={y + 4}
                                textAnchor="end" fontSize="10" fill="#8a8fb5"
                            >
                                {formatK(val)}
                            </text>
                        </g>
                    ))}

                    {/* ── X-axis date labels ── */}
                    {xTicks.map(({ label, x }, i) => (
                        <g key={i}>
                            <line
                                x1={x} y1={PAD.top} x2={x} y2={PAD.top + chartH + 4}
                                stroke="rgba(255,255,255,0.04)" strokeWidth="1"
                            />
                            <text
                                x={x} y={PAD.top + chartH + 18}
                                textAnchor="middle" fontSize="9.5" fill="#8a8fb5"
                            >
                                {label}
                            </text>
                        </g>
                    ))}

                    {/* ── Axis border ── */}
                    <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + chartH}
                        stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                    <line x1={PAD.left} y1={PAD.top + chartH} x2={PAD.left + chartW} y2={PAD.top + chartH}
                        stroke="rgba(255,255,255,0.1)" strokeWidth="1" />

                    {/* ── Clip group ── */}
                    <g clipPath="url(#chartClip)">
                        {/* Baseline area fill */}
                        {baseline.length >= 2 && (
                            <path d={makeArea(baseline)} fill="url(#gradBase)" />
                        )}
                        {/* NLP area fill */}
                        {gated.length >= 2 && (
                            <path d={makeArea(gated)} fill="url(#gradGated)" />
                        )}

                        {/* Baseline line — dashed gray */}
                        {baseline.length >= 2 && (
                            <path
                                d={makePath(baseline)}
                                fill="none"
                                stroke="#999"
                                strokeWidth="1.5"
                                strokeDasharray="5,3"
                            />
                        )}

                        {/* NLP Augmented line — solid bright indigo */}
                        {gated.length >= 2 && (
                            <path
                                d={makePath(gated)}
                                fill="none"
                                stroke="#818cf8"
                                strokeWidth="2.5"
                                strokeLinejoin="round"
                                strokeLinecap="round"
                            />
                        )}

                        {/* Start/end dot on NLP line */}
                        {gated.length >= 2 && (<>
                            <circle
                                cx={toX(0, gated.length)} cy={toY(gated[0])}
                                r="3.5" fill="#818cf8" stroke="#0c0d16" strokeWidth="1.5"
                            />
                            <circle
                                cx={toX(gated.length - 1, gated.length)} cy={toY(gated[gated.length - 1])}
                                r="5" fill="#818cf8" stroke="#0c0d16" strokeWidth="2"
                            />
                        </>)}
                    </g>
                </svg>

                {/* ── Legend ── */}
                <div style={{
                    display: "flex", gap: "1.5rem", justifyContent: "center",
                    marginTop: "0.5rem", fontSize: "0.8rem", color: "#c9cdfb"
                }}>
                    <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <svg width="24" height="4">
                            <line x1="0" y1="2" x2="24" y2="2"
                                stroke="#999" strokeWidth="1.5" strokeDasharray="5,3" />
                        </svg>
                        Baseline Strategy
                    </span>
                    <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <svg width="24" height="4">
                            <line x1="0" y1="2" x2="24" y2="2"
                                stroke="#818cf8" strokeWidth="2.5" />
                        </svg>
                        NLP Augmented
                    </span>
                </div>
            </div>
        );
    };

    const renderDrawdownChart = (data, dates, isFull = false) => {
        if (!data) return null;
        const dsG = data.gated || [0];
        const dsB = data.baseline || [0];

        // If mini view, only show last 60 points for "Recent Pressure"
        const miniWindow = 80;
        const seriesG = isFull ? dsG : dsG.slice(-miniWindow);
        const seriesB = isFull ? dsB : dsB.slice(-miniWindow);
        const seriesDates = isFull ? dates : dates.slice(-miniWindow);

        const W = isFull ? 800 : 300, H = isFull ? 300 : 120;
        const PAD = isFull ? { top: 20, right: 24, bottom: 40, left: 60 } : { top: 10, right: 5, bottom: 20, left: 5 };
        const chartW = W - PAD.left - PAD.right;
        const chartH = H - PAD.top - PAD.bottom;

        const combined = [...seriesG, ...seriesB];
        const maxAbsDD = Math.max(...combined.map(v => Math.abs(v)), 0.02); // at least 2% scale

        const toX = (i, len) => PAD.left + (i / (len - 1)) * chartW;
        const toY = (v) => PAD.top + (Math.abs(v) / maxAbsDD) * chartH;

        const makePath = (series) =>
            series.map((v, i) => `${i === 0 ? "M" : "L"}${toX(i, series.length).toFixed(1)},${toY(v).toFixed(1)}`).join(" ");

        const makeArea = (series) => {
            const line = makePath(series);
            const lastX = toX(series.length - 1, series.length).toFixed(1);
            const firstX = toX(0, series.length).toFixed(1);
            const baseY = PAD.top.toFixed(1); // Hanging from $0
            return `${line} L${lastX},${baseY} L${firstX},${baseY} Z`;
        };

        const xTickCount = isFull ? 6 : 0;
        const xTicks = Array.from({ length: xTickCount }, (_, i) => {
            const idx = Math.round((i / (xTickCount - 1)) * (seriesDates.length - 1));
            return { label: seriesDates[idx]?.slice(0, 10) || "", x: toX(idx, seriesDates.length) };
        });

        return (
            <div style={{ height: "100%", width: "100%" }}>
                <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "100%", overflow: "visible" }} xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="ddGradGated" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#ff6b6b" stopOpacity="0.3" />
                            <stop offset="100%" stopColor="#ff6b6b" stopOpacity="0.01" />
                        </linearGradient>
                    </defs>

                    {isFull && (
                        <g>
                            {[0, 0.25, 0.5, 0.75, 1].map((p, i) => {
                                const val = p * maxAbsDD;
                                const y = toY(-val);
                                return (
                                    <g key={i}>
                                        <line x1={PAD.left} y1={y} x2={PAD.left + chartW} y2={y} stroke="rgba(255,255,255,0.05)" />
                                        <text x={PAD.left - 8} y={y + 4} textAnchor="end" fontSize="10" fill="#8a8fb5">-{(val * 100).toFixed(1)}%</text>
                                    </g>
                                );
                            })}
                            {xTicks.map(({ label, x }, i) => (
                                <text key={i} x={x} y={PAD.top + chartH + 18} textAnchor="middle" fontSize="9.5" fill="#8a8fb5">{label}</text>
                            ))}
                        </g>
                    )}

                    <line x1={PAD.left} y1={PAD.top} x2={PAD.left + chartW} y2={PAD.top} stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                    <path d={makeArea(seriesB)} fill="rgba(255,255,255,0.02)" />
                    <path d={makeArea(seriesG)} fill="url(#ddGradGated)" />
                    <path d={makePath(seriesB)} fill="none" stroke="#555" strokeWidth="1" strokeDasharray="3,2" />
                    <path d={makePath(seriesG)} fill="none" stroke="#ff6b6b" strokeWidth="2" strokeLinejoin="round" />
                </svg>
            </div>
        );
    };


    return (
        <>
            <div className="pro-dashboard">
                <div className="dashboard-sidebar">
                    <div className="sidebar-header">
                        <h3>Simulation Engine</h3>
                        <p className="muted">2024 – Present • Daily BTC</p>
                    </div>

                    {selectedModel && (
                        <div className="config-group" style={{ marginBottom: '1.5rem' }}>
                            <p className="tiny muted" style={{ margin: 0 }}>
                                Model : <span style={{ color: "#818cf8", fontWeight: 600 }}>{selectedModel}</span>
                            </p>
                        </div>
                    )}

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
                            <div className="pulse-icon">BT</div>
                            <h2>Ready for Simulation</h2>
                            <p>Configure your model and capital parameters to begin the 2024 → present BTC daily analysis.</p>
                        </div>
                    ) : (
                        <>
                            {result.case_study && (
                                <div className="card" style={{
                                    marginBottom: "1.5rem",
                                    background: "linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(129,140,248,0.03) 100%)",
                                    border: "1px solid rgba(99,102,241,0.2)",
                                    padding: "1.2rem"
                                }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                                        <h3 style={{ margin: 0, display: "flex", alignItems: "center", gap: "8px" }}>
                                            CASE STUDY: {result.case_study.date}
                                        </h3>
                                        <div style={{
                                            padding: "4px 12px", borderRadius: "20px", fontSize: "0.75rem", fontWeight: 700,
                                            background: result.case_study.outcome_label.includes("Saved") || result.case_study.outcome_label.includes("Alpha") ? "rgba(81,207,102,0.15)" : "rgba(129,140,248,0.15)",
                                            color: result.case_study.outcome_label.includes("Saved") || result.case_study.outcome_label.includes("Alpha") ? "#51cf66" : "#818cf8",
                                            border: `1px solid ${result.case_study.outcome_label.includes("Saved") || result.case_study.outcome_label.includes("Alpha") ? "rgba(81,207,102,0.3)" : "rgba(129,140,248,0.3)"}`
                                        }}>
                                            {result.case_study.outcome_label.toUpperCase()}
                                        </div>
                                    </div>
                                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2rem" }}>
                                        <div>
                                            <p style={{ fontSize: "0.78rem", color: "#8a8fb5", marginBottom: "0.4rem", textTransform: "uppercase", letterSpacing: "0.5px" }}>Market Context</p>
                                            <p style={{ fontSize: "1rem", fontWeight: 500, lineHeight: 1.4, margin: "0 0 10px", color: "#e0e2ff" }}>"{result.case_study.headline}"</p>
                                            <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                                                <span style={{ fontSize: "0.8rem", color: "#8a8fb5" }}>Actual Market Move: <strong style={{ color: "#fff" }}>{result.case_study.real_market_impact}</strong></span>
                                            </div>
                                        </div>
                                        <div style={{ paddingLeft: "2rem", borderLeft: "1px solid rgba(255,255,255,0.08)" }}>
                                            <p style={{ fontSize: "0.78rem", color: "#8a8fb5", marginBottom: "0.4rem", textTransform: "uppercase", letterSpacing: "0.5px" }}>NLP Gating Strategy</p>
                                            <div style={{ display: "flex", gap: "1.5rem", marginBottom: "0.75rem" }}>
                                                <div>
                                                    <div style={{ fontSize: "0.65rem", color: "#8a8fb5", textTransform: "uppercase" }}>Model Verdict</div>
                                                    <div style={{ color: result.case_study.prediction === "Bullish" ? "#51cf66" : "#ff6b6b", fontWeight: 700, fontSize: "1rem" }}>
                                                        {result.case_study.prediction === "Bullish" ? "[BULLISH]" : "[BEARISH]"}
                                                    </div>
                                                </div>
                                                <div>
                                                    <div style={{ fontSize: "0.65rem", color: "#8a8fb5", textTransform: "uppercase" }}>Confidence</div>
                                                    <div style={{ color: "#818cf8", fontWeight: 700, fontSize: "1rem" }}>{(result.case_study.confidence * 100).toFixed(0)}%</div>
                                                </div>
                                                <div>
                                                    <div style={{ fontSize: "0.65rem", color: "#8a8fb5", textTransform: "uppercase" }}>Action</div>
                                                    <div style={{ fontWeight: 700, fontSize: "1rem", color: "#fff" }}>{result.case_study.gating_status}</div>
                                                </div>
                                            </div>
                                            <div style={{ display: "flex", gap: "8px", alignItems: "flex-start" }}>
                                                <span style={{ color: "#818cf8", fontSize: "0.8rem", fontWeight: 700 }}>NOTE:</span>
                                                <p style={{ fontSize: "0.82rem", color: "#c4c8f0", fontStyle: "italic", margin: 0, lineHeight: 1.4 }}>{result.case_study.outcome_desc}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
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
                                        <h3>Cumulative Equity Curve
                                            <span style={{ fontSize: "0.75rem", color: "#8a8fb5", fontWeight: 400, marginLeft: "0.75rem" }}>
                                                {result.from_date || "2024-01-01"} → {result.to_date || "today"}
                                            </span>
                                        </h3>
                                    </div>
                                    <div className="chart-container">
                                        {renderEquityChart(result.equity_curve)}
                                    </div>
                                </div>

                                {/* Side cards: keep only drawdown now */}
                                <div className="side-cards">
                                    <section className="card drawdown-card" onClick={() => setShowDrawdownModal(true)}>
                                        <div className="flex-between">
                                            <h3 style={{ margin: 0 }}>Drawdown Profile</h3>
                                            <span className="tiny muted">Recent Pressure • Show All</span>
                                        </div>
                                        <div className="mini-chart" style={{ height: "120px", marginTop: "1rem" }}>
                                            {renderDrawdownChart(result.drawdown_curves, result.equity_curve?.dates, false)}
                                        </div>
                                        <div style={{ display: "flex", justifyContent: "center", gap: "1rem", marginTop: "0.5rem" }}>
                                            <span style={{ fontSize: "0.65rem", color: "#666" }}>-- Baseline</span>
                                            <span style={{ fontSize: "0.65rem", color: "#ff6b6b" }}>━ NLP Augmented</span>
                                        </div>
                                        <p className="tiny muted text-center">Inverted timeline (0% at top). Click to enlarge.</p>
                                    </section>

                                    {/* Baseline vs NLP comparison card */}
                                    <section className="card">
                                        <h3>Strategy Comparison</h3>
                                        <div style={{ marginTop: "1rem" }}>
                                            {["total_return", "sharpe_ratio", "max_drawdown", "win_rate"].map(metric => {
                                                const base = result.metrics?.baseline?.[metric] || 0;
                                                const gated = result.metrics?.gated?.[metric] || 0;

                                                const isDiverging = ["total_return", "sharpe_ratio"].includes(metric);
                                                let maxScale = Math.max(Math.abs(base), Math.abs(gated), 0.001);
                                                if (metric === "win_rate") maxScale = 1.0;
                                                if (metric === "max_drawdown") maxScale = Math.max(maxScale, 0.1);

                                                const getBarParams = (val) => {
                                                    const absVal = Math.abs(val);
                                                    const wPct = (absVal / maxScale) * (isDiverging ? 50 : 100);
                                                    return {
                                                        width: `${Math.min(isDiverging ? 50 : 100, wPct)}%`,
                                                        left: isDiverging ? (val < 0 ? `${50 - wPct}%` : "50%") : "0%"
                                                    };
                                                };

                                                const bpBase = getBarParams(base);
                                                const bpGated = getBarParams(gated);

                                                return (
                                                    <div key={metric} style={{ marginBottom: "1rem" }}>
                                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", color: "#8a8fb5", marginBottom: "6px" }}>
                                                            <span>{metric.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}</span>
                                                            <span>
                                                                <span style={{ color: "#999", marginRight: "10px" }}>
                                                                    {metric === "sharpe_ratio" ? base.toFixed(2) : formatPct(base)}
                                                                </span>
                                                                <span style={{ color: "#a5b4fc", fontWeight: 700 }}>
                                                                    {metric === "sharpe_ratio" ? gated.toFixed(2) : formatPct(gated)}
                                                                </span>
                                                            </span>
                                                        </div>
                                                        <div style={{ height: "16px", background: "rgba(255,255,255,0.03)", borderRadius: "4px", position: "relative", overflow: "hidden", border: "1px solid rgba(255,255,255,0.05)" }}>
                                                            {isDiverging && (
                                                                <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: "1px", background: "rgba(255,255,255,0.15)", zIndex: 1 }} />
                                                            )}
                                                            <div style={{
                                                                position: "absolute", top: "2px", height: "5px",
                                                                left: bpBase.left, width: bpBase.width,
                                                                background: "#999",
                                                                opacity: 0.8, borderRadius: "2px", transition: "all 0.4s ease"
                                                            }} />
                                                            <div style={{
                                                                position: "absolute", bottom: "2px", height: "5px",
                                                                left: bpGated.left, width: bpGated.width,
                                                                background: "#a5b4fc",
                                                                borderRadius: "2px", transition: "all 0.4s ease"
                                                            }} />
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                        <p className="tiny muted" style={{ marginTop: "0.8rem" }}>
                                            <span style={{ color: "#999" }}>■</span> Baseline &nbsp;
                                            <span style={{ color: "#a5b4fc" }}>■</span> NLP Augmented
                                        </p>
                                    </section>
                                </div>
                            </div>

                            {/* ── Historical Headline Analysis Panel ── */}
                            <div className="card" style={{ marginTop: "1.5rem" }}>
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
                                    <div>
                                        <h3 style={{ margin: 0 }}>Historical Headline Analysis</h3>
                                        <p className="tiny muted" style={{ margin: "4px 0 0" }}>
                                            Fetch real Bitcoin headlines from the backtest period and run full NLP analysis
                                        </p>
                                    </div>
                                    <button
                                        className="btn primary"
                                        onClick={fetchHeadlineSamples}
                                        disabled={hlLoading || !result}
                                        style={{ whiteSpace: "nowrap", minWidth: "180px" }}
                                    >
                                        {hlLoading ? "Fetching & Analysing…" : "Analyse Period Headlines"}
                                    </button>
                                </div>

                                {hlError && (
                                    <div style={{ background: "rgba(255,107,107,0.1)", border: "1px solid rgba(255,107,107,0.3)", borderRadius: "8px", padding: "0.75rem 1rem", color: "#ff8fa3", fontSize: "0.85rem" }}>
                                        ERROR: {hlError}
                                    </div>
                                )}

                                {hlSamples.length > 0 && (
                                    <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", gap: "1.5rem", alignItems: "start" }}>
                                        {/* Headline selector */}
                                        <div>
                                            <p className="tiny muted" style={{ marginBottom: "0.5rem" }}>Select a headline to analyse:</p>
                                            {hlSamples.map((h, i) => (
                                                <div
                                                    key={i}
                                                    onClick={() => setSelectedHl(h)}
                                                    style={{
                                                        padding: "0.7rem 0.9rem",
                                                        borderRadius: "8px",
                                                        marginBottom: "0.5rem",
                                                        cursor: "pointer",
                                                        border: selectedHl === h ? "1px solid #6366f1" : "1px solid rgba(255,255,255,0.07)",
                                                        background: selectedHl === h ? "rgba(99,102,241,0.12)" : "rgba(255,255,255,0.03)",
                                                        transition: "all 0.15s",
                                                    }}
                                                >
                                                    <p style={{ fontSize: "0.82rem", margin: "0 0 6px", lineHeight: 1.4, color: selectedHl === h ? "#e0e2ff" : "#c4c8f0" }}>{h.headline}</p>
                                                    <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                                                        <span style={{
                                                            fontSize: "0.7rem", padding: "2px 8px", borderRadius: "20px", fontWeight: 600,
                                                            background: h.sentiment === "Bullish" ? "rgba(81,207,102,0.15)" : h.sentiment === "Bearish" ? "rgba(255,107,107,0.15)" : "rgba(255,195,0,0.15)",
                                                            color: h.sentiment === "Bullish" ? "#51cf66" : h.sentiment === "Bearish" ? "#ff6b6b" : "#ffc300",
                                                        }}>{h.sentiment}</span>
                                                        <span style={{ fontSize: "0.68rem", color: "#8a8fb5" }}>{h.source} · {h.pubDate?.slice(0, 10)}</span>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>

                                        {/* Deep analysis panel for selected headline */}
                                        {selectedHl && (
                                            <div>
                                                {/* Verdict */}
                                                <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.2rem", padding: "1rem", borderRadius: "10px", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)" }}>
                                                    <div style={{
                                                        width: 64, height: 64, borderRadius: "50%",
                                                        border: `4px solid ${selectedHl.sentiment === "Bullish" ? "#51cf66" : selectedHl.sentiment === "Bearish" ? "#ff6b6b" : "#ffc300"}`,
                                                        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                                                        fontSize: "1.4rem",
                                                    }}>
                                                        {selectedHl.sentiment === "Bullish" ? "B" : selectedHl.sentiment === "Bearish" ? "S" : "N"}
                                                    </div>
                                                    <div>
                                                        <div style={{ fontSize: "1.1rem", fontWeight: 700, color: selectedHl.sentiment === "Bullish" ? "#51cf66" : selectedHl.sentiment === "Bearish" ? "#ff6b6b" : "#ffc300" }}>
                                                            {selectedHl.sentiment}
                                                        </div>
                                                        <div style={{ fontSize: "0.82rem", color: "#8a8fb5" }}>
                                                            {(selectedHl.confidence * 100).toFixed(1)}% confidence · {selectedHl.model}
                                                        </div>
                                                        <div style={{ fontSize: "0.78rem", color: "#c4c8f0", marginTop: "4px", fontStyle: "italic" }}>
                                                            "{selectedHl.headline}"
                                                        </div>
                                                    </div>
                                                </div>

                                                {/* Token Attribution */}
                                                {selectedHl.explainability?.token_attributions && (
                                                    <div style={{ marginBottom: "1rem", padding: "0.8rem", borderRadius: "8px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                                                        <p style={{ fontSize: "0.78rem", color: "#8a8fb5", marginBottom: "0.5rem" }}>Token Attribution Map</p>
                                                        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", lineHeight: 2 }}>
                                                            {selectedHl.explainability.token_attributions.map((tok, i) => {
                                                                const score = tok.attribution_score ?? tok.score ?? 0;
                                                                const abs = Math.abs(score);
                                                                const intensity = Math.min(abs * 3, 1);
                                                                const bg = score > 0
                                                                    ? `rgba(81,207,102,${0.12 + intensity * 0.4})`
                                                                    : score < 0
                                                                        ? `rgba(255,107,107,${0.12 + intensity * 0.4})`
                                                                        : "rgba(255,255,255,0.05)";
                                                                return (
                                                                    <span key={i} title={`Score: ${score.toFixed(3)}`} style={{
                                                                        background: bg, borderRadius: "4px", padding: "2px 7px",
                                                                        fontWeight: abs > 0.15 ? 700 : 400,
                                                                        fontSize: "0.82rem",
                                                                        color: score > 0 ? "#a3f0b5" : score < 0 ? "#ffb3b3" : "#c4c8f0",
                                                                        border: abs > 0.2 ? `1px solid ${score > 0 ? "rgba(81,207,102,0.4)" : "rgba(255,107,107,0.4)"}` : "none",
                                                                    }}>
                                                                        {tok.token}
                                                                    </span>
                                                                );
                                                            })}
                                                        </div>
                                                    </div>
                                                )}

                                                {/* Power Words */}
                                                {selectedHl.top_tokens?.length > 0 && (
                                                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.8rem", marginBottom: "1rem" }}>
                                                        {[{ label: "Bullish Drivers", color: "#51cf66", items: selectedHl.top_tokens.filter(t => (t.score ?? 0) > 0).slice(0, 4) },
                                                        { label: "Bearish Signals", color: "#ff6b6b", items: selectedHl.top_tokens.filter(t => (t.score ?? 0) < 0).slice(0, 4) }].map(({ label, color, items }) => (
                                                            <div key={label} style={{ padding: "0.7rem", borderRadius: "8px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                                                                <p style={{ fontSize: "0.72rem", color, marginBottom: "0.5rem", fontWeight: 600 }}>{label}</p>
                                                                {items.length === 0 ? <p style={{ fontSize: "0.72rem", color: "#666" }}>None detected</p> : items.map((t, i) => (
                                                                    <div key={i} style={{ marginBottom: "4px" }}>
                                                                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", color: "#c4c8f0", marginBottom: "3px", alignItems: "center" }}>
                                                                            <span>{t.token}</span>
                                                                            <span style={{
                                                                                fontSize: "0.65rem",
                                                                                padding: "1px 5px",
                                                                                borderRadius: "4px",
                                                                                background: "rgba(255,255,255,0.05)",
                                                                                color: Math.abs(t.score) > 0.6 ? color : "#8a8fb5",
                                                                                fontWeight: Math.abs(t.score) > 0.6 ? 700 : 400,
                                                                                border: `1px solid ${Math.abs(t.score) > 0.6 ? color + '44' : 'transparent'}`
                                                                            }}>
                                                                                {Math.abs(t.score) > 0.7 ? "CRITICAL" : Math.abs(t.score) > 0.4 ? "HIGH" : "MID"}
                                                                            </span>
                                                                        </div>
                                                                        <div style={{ height: "2px", background: "rgba(255,255,255,0.06)", borderRadius: "2px", overflow: "hidden" }}>
                                                                            <div style={{ width: `${Math.min(100, Math.abs(t.score) * 100)}%`, height: "100%", background: color, borderRadius: "2px", opacity: 0.8 }} />
                                                                        </div>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}

                                                {/* Stability + Counterfactual */}
                                                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.8rem" }}>
                                                    {selectedHl.explainability?.stability_score != null && (
                                                        <div style={{ padding: "0.7rem", borderRadius: "8px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                                                            <p style={{ fontSize: "0.72rem", color: "#8a8fb5", marginBottom: "6px" }}>Prediction Stability</p>
                                                            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                                                                <div style={{ flex: 1, height: "6px", background: "rgba(255,255,255,0.06)", borderRadius: "3px" }}>
                                                                    <div style={{ width: `${((selectedHl.explainability.stability_score ?? 0) * 100).toFixed(0)}%`, height: "100%", background: "linear-gradient(90deg,#6366f1,#818cf8)", borderRadius: "3px" }} />
                                                                </div>
                                                                <span style={{ fontSize: "0.78rem", color: "#818cf8", fontWeight: 600 }}>
                                                                    {((selectedHl.explainability.stability_score ?? 0) * 100).toFixed(0)}%
                                                                </span>
                                                            </div>
                                                        </div>
                                                    )}
                                                    {selectedHl.explainability?.counterfactual && (
                                                        <div style={{ padding: "0.7rem", borderRadius: "8px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                                                            <p style={{ fontSize: "0.72rem", color: "#8a8fb5", marginBottom: "4px" }}>Counterfactual</p>
                                                            <p style={{ fontSize: "0.73rem", color: "#c4c8f0", fontStyle: "italic", margin: 0 }}>
                                                                "{selectedHl.explainability.counterfactual.modified_text?.slice(0, 80)}…"
                                                            </p>
                                                            <p style={{ fontSize: "0.7rem", color: selectedHl.explainability.counterfactual.flipped ? "#ff6b6b" : "#51cf66", marginTop: "4px", fontWeight: 600 }}>
                                                                {selectedHl.explainability.counterfactual.flipped ? "⚡ Prediction flips" : "✓ Prediction holds"}
                                                            </p>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {!hlLoading && hlSamples.length === 0 && !hlError && (
                                    <div style={{ textAlign: "center", padding: "2rem", color: "#8a8fb5", fontSize: "0.85rem" }}>
                                        Click <strong style={{ color: "#818cf8" }}>Analyse Period Headlines</strong> to fetch real Bitcoin news from your backtest window ({result?.from_date} → {result?.to_date}) and run full NLP analysis.
                                    </div>
                                )}
                            </div>
                        </>
                    )}
                </main>
            </div>

            {showDrawdownModal && result && (
                <div className="modal-overlay" onClick={() => setShowDrawdownModal(false)}>
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <button className="modal-close" onClick={() => setShowDrawdownModal(false)}>×</button>
                        <div style={{ marginBottom: "1.5rem" }}>
                            <h2 style={{ margin: 0 }}>Full Timeline Drawdown Profile</h2>
                            <p className="muted" style={{ margin: "5px 0 0" }}>
                                Comparative risk analysis of {result.model} vs baseline strategy (2024 – Present)
                            </p>
                        </div>
                        <div style={{ height: "400px", width: "100%", background: "rgba(0,0,0,0.15)", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.05)", padding: "1rem" }}>
                            {renderDrawdownChart(result.drawdown_curves, result.equity_curve?.dates, true)}
                        </div>
                        <div style={{ display: "flex", justifyContent: "center", gap: "2rem", marginTop: "1rem", fontSize: "0.9rem" }}>
                            <span style={{ display: "flex", alignItems: "center", gap: "8px", color: "#8a8fb5" }}>
                                <span style={{ width: "24px", height: "1px", background: "#555", borderTop: "1px dashed #555" }}></span>
                                Baseline Strategy Max DD: {formatPct(result.metrics?.baseline?.max_drawdown)}
                            </span>
                            <span style={{ display: "flex", alignItems: "center", gap: "8px", color: "#ff6b6b", fontWeight: 700 }}>
                                <span style={{ width: "24px", height: "2px", background: "#ff6b6b" }}></span>
                                NLP Augmented Max DD: {formatPct(result.metrics?.gated?.max_drawdown)}
                            </span>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
