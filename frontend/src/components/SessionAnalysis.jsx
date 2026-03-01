import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";


const fmt = (v, d = 4) => {
    if (v === null || v === undefined || v === "") return "-";
    const num = Number(v);
    if (Number.isNaN(num)) return v;
    return Number.isInteger(num) ? `${num}` : num.toFixed(d);
};
const fmtPct = (v) => v != null ? (v * 100).toFixed(2) + "%" : "-";
const fmtCur = (v) => v != null ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(v) : "-";
const COLORS = ["#818cf8", "#51cf66", "#ff9f43", "#ff6b6b", "#34d399"];

/* ── Placeholder card for sections with no data ── */
function EmptySection({ sectionNum, title, tabName }) {
    return (
        <div className="card" style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "0.7rem", marginBottom: "1.2rem" }}>
                {sectionNum}. {title}
            </h3>
            <div style={{
                textAlign: "center",
                padding: "2rem 1rem",
                borderRadius: "10px",
                background: "rgba(255,255,255,0.02)",
                border: "1px dashed rgba(255,255,255,0.08)"
            }}>
                <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem", opacity: 0.4 }}>Data Pending</div>
                <p style={{ color: "#8a8fb5", fontSize: "0.88rem", margin: 0 }}>
                    Please interact with <strong style={{ color: "#818cf8" }}>{tabName}</strong> to see final summary
                </p>
            </div>
        </div>
    );
}

export default function SessionAnalysis({ benchmark = [], errorSummaries = {}, leaderboard = [] }) {
    const reportRef = useRef(null);
    const [backtest, setBacktest] = useState(null);
    const [headlines, setHeadlines] = useState([]);
    const [chatSummary, setChatSummary] = useState("");
    const [loading, setLoading] = useState(true);
    const [downloading, setDownloading] = useState(false);
    const [errorData, setErrorData] = useState({ length: [], signal: [], confidence: [] });

    const bestModel = leaderboard.length > 0 ? leaderboard[0] : null;
    const hasBenchmark = benchmark.length > 0;

    useEffect(() => {
        const load = async () => {
            setLoading(true);
            try {
                // 1. Fetch latest backtest (fail silently)
                try {
                    const btRes = await fetch("/api/backtest/latest");
                    if (btRes.ok) {
                        const btData = await btRes.json();
                        if (btData && btData.model) setBacktest(btData);
                    }
                } catch (e) { console.warn("Backtest fetch failed:", e); }

                // 2. Fetch 3 live headlines and analyze (only if we have a model)
                if (bestModel?.model) {
                    try {
                        const newsRes = await fetch("/api/news/bitcoin-headlines");
                        const newsData = await newsRes.json();
                        const rawHeadlines = (newsData.headlines || []).slice(0, 3);
                        if (rawHeadlines.length > 0) {
                            const analyzeRes = await fetch("/api/analyze", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    model: bestModel.model,
                                    headlines: rawHeadlines.map(h => h.title)
                                })
                            });
                            const analyzed = await analyzeRes.json();
                            setHeadlines(rawHeadlines.map((h, i) => ({
                                ...h,
                                sentiment: analyzed[i]?.sentiment || "Unknown",
                                confidence: analyzed[i]?.confidence || 0
                            })));
                        }
                    } catch (e) { console.warn("Headlines fetch failed:", e); }
                }

                // 3. Get Market GPT summary (only if we have data to ground on)
                if (hasBenchmark) {
                    try {
                        const chatRes = await fetch("/api/chat", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ query: "Write a concise 3-paragraph executive summary of the current session results as a quant researcher. Cover model performance, backtest outcomes, and strategic recommendations. DO NOT USE ANY EMOJIS." })
                        });
                        const chatData = await chatRes.json();
                        setChatSummary(chatData.response || "");
                    } catch (e) { console.warn("Chat summary failed:", e); }
                }

                // 4. Fetch Deep Error Attribution for Best Model
                if (bestModel?.model) {
                    try {
                        const [len, sig, conf] = await Promise.all([
                            fetch(`/api/errors/error_by_length?model=${bestModel.model}`).then(r => r.json()),
                            fetch(`/api/errors/error_by_signal?model=${bestModel.model}`).then(r => r.json()),
                            fetch(`/api/errors/error_by_confidence?model=${bestModel.model}`).then(r => r.json()),
                        ]);
                        setErrorData({
                            length: len?.rows || [],
                            signal: sig?.rows || [],
                            confidence: conf?.rows || []
                        });
                    } catch (e) { console.warn("Error attribution fetch failed:", e); }
                }
            } catch (e) {
                console.error("Session analysis load failed:", e);
            } finally { setLoading(false); }
        };
        load();
    }, [bestModel?.model, hasBenchmark]);

    const handleDownloadPDF = async () => {
        setDownloading(true);
        try {
            const res = await fetch("/api/pdf");
            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "Session_Analysis_Report.pdf";
                a.click();
                window.URL.revokeObjectURL(url);
            } else {
                alert("PDF generation failed. Run a backtest first.");
            }
        } catch (e) {
            alert("Failed to download PDF: " + e.message);
        } finally { setDownloading(false); }
    };


    if (loading) {
        return (
            <div className="card" style={{ textAlign: "center", padding: "4rem" }}>
                <div style={{ fontSize: "1.5rem", marginBottom: "1rem" }}>ANALYSIS</div>
                <p style={{ color: "#818cf8", fontWeight: 600 }}>Compiling Session Analysis Report...</p>
                <p className="tiny muted">Aggregating benchmarks, backtest results, and live sentiment data</p>
            </div>
        );
    }

    const gated = backtest?.metrics?.gated || {};
    const baseline = backtest?.metrics?.baseline || {};
    const BAR_H = 22, GAP = 6, LABEL_W = 90, BAR_MAX_W = 180;

    return (
        <div ref={reportRef} className="session-analysis">
            {/* Header */}
            <div className="card" style={{ marginBottom: "1.5rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div>
                        <h2 style={{ margin: 0, fontSize: "1.4rem" }}>Session Analysis Report</h2>
                        <p className="muted" style={{ margin: "6px 0 0" }}>
                            Generated {new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })} · {benchmark.length} models evaluated
                        </p>
                    </div>
                    <button
                        id="pdf-download-btn"
                        className="btn primary"
                        onClick={handleDownloadPDF}
                        disabled={downloading}
                        style={{ whiteSpace: "nowrap" }}
                    >
                        {downloading ? "Generating…" : "Download PDF"}
                    </button>
                </div>
            </div>

            {/* ─── Section 1: Benchmark Overview ─── */}
            {hasBenchmark ? (
                <div className="card" style={{ marginBottom: "1.5rem" }}>
                    <h3 style={{ borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "0.7rem", marginBottom: "1.2rem" }}>
                        1. Model Benchmark Overview
                    </h3>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem", marginBottom: "1.5rem" }}>
                        {/* F1 chart */}
                        <div>
                            <p style={{ fontSize: "0.75rem", color: "#8a8fb5", fontWeight: 600, marginBottom: "0.5rem" }}>F1 Macro Score</p>
                            <svg viewBox={`0 0 ${LABEL_W + BAR_MAX_W + 60} ${(BAR_H + GAP) * benchmark.length + 10}`}
                                style={{ width: "100%", height: (BAR_H + GAP) * benchmark.length + 10 }}>
                                {benchmark.map((r, i) => {
                                    const v = Number(r.f1_macro) || 0;
                                    const maxV = Math.max(...benchmark.map(b => Number(b.f1_macro) || 0), 0.001);
                                    const bW = (v / maxV) * BAR_MAX_W;
                                    const y = 5 + i * (BAR_H + GAP);
                                    return (
                                        <g key={i}>
                                            <text x={LABEL_W - 6} y={y + BAR_H * 0.68} textAnchor="end" fontSize="10" fill="#c4c8f0">{r.model}</text>
                                            <rect x={LABEL_W} y={y} width={bW} height={BAR_H} rx="3" fill={COLORS[i % COLORS.length]} opacity="0.8" />
                                            <text x={LABEL_W + bW + 6} y={y + BAR_H * 0.68} fontSize="9" fill={COLORS[i % COLORS.length]} fontWeight="700">{v.toFixed(4)}</text>
                                        </g>
                                    );
                                })}
                            </svg>
                        </div>
                        {/* Accuracy chart */}
                        <div>
                            <p style={{ fontSize: "0.75rem", color: "#8a8fb5", fontWeight: 600, marginBottom: "0.5rem" }}>Accuracy</p>
                            <svg viewBox={`0 0 ${LABEL_W + BAR_MAX_W + 60} ${(BAR_H + GAP) * benchmark.length + 10}`}
                                style={{ width: "100%", height: (BAR_H + GAP) * benchmark.length + 10 }}>
                                {benchmark.map((r, i) => {
                                    const v = Number(r.accuracy) || 0;
                                    const maxV = Math.max(...benchmark.map(b => Number(b.accuracy) || 0), 0.001);
                                    const bW = (v / maxV) * BAR_MAX_W;
                                    const y = 5 + i * (BAR_H + GAP);
                                    return (
                                        <g key={i}>
                                            <text x={LABEL_W - 6} y={y + BAR_H * 0.68} textAnchor="end" fontSize="10" fill="#c4c8f0">{r.model}</text>
                                            <rect x={LABEL_W} y={y} width={bW} height={BAR_H} rx="3" fill={COLORS[i % COLORS.length]} opacity="0.8" />
                                            <text x={LABEL_W + bW + 6} y={y + BAR_H * 0.68} fontSize="9" fill={COLORS[i % COLORS.length]} fontWeight="700">{v.toFixed(4)}</text>
                                        </g>
                                    );
                                })}
                            </svg>
                        </div>
                    </div>
                    {/* Raw table */}
                    <table style={{ fontSize: "0.8rem" }}>
                        <thead>
                            <tr>
                                <th>Model</th><th>F1 Macro</th><th>Accuracy</th><th>Latency (ms)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {leaderboard.map((r, i) => (
                                <tr key={r.model} style={i === 0 ? { background: "rgba(129,140,248,0.08)" } : {}}>
                                    <td><strong>{r.model}</strong></td>
                                    <td style={i === 0 ? { color: "#818cf8", fontWeight: 700 } : {}}>{fmt(r.f1_macro)}</td>
                                    <td>{fmt(r.accuracy)}</td>
                                    <td>{fmt(r.latency_ms_per_tweet, 2)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>

                    {/* Deep Error Attribution Sub-section */}
                    <div style={{ marginTop: "1.5rem", paddingTop: "1.2rem", borderTop: "1px solid rgba(255,255,255,0.05)" }}>
                        <h4 style={{ fontSize: "0.9rem", color: "#818cf8", marginBottom: "0.8rem" }}>Deep Error Attribution: {bestModel.model}</h4>
                        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem" }}>
                            <div style={{ padding: "0.6rem", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
                                <div style={{ fontSize: "0.65rem", color: "#8a8fb5", textTransform: "uppercase", marginBottom: "6px" }}>By Text Length</div>
                                <table className="tiny" style={{ fontSize: "0.7rem", marginTop: 0 }}>
                                    <thead><tr><th>Len</th><th>Err</th></tr></thead>
                                    <tbody>
                                        {errorData.length.slice(0, 4).map((r, i) => (
                                            <tr key={i}><td>{r.length_bucket}</td><td style={{ color: "#ff8f8f" }}>{fmt(r.error_rate)}</td></tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <div style={{ padding: "0.6rem", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
                                <div style={{ fontSize: "0.65rem", color: "#8a8fb5", textTransform: "uppercase", marginBottom: "6px" }}>By Signal</div>
                                <table className="tiny" style={{ fontSize: "0.7rem", marginTop: 0 }}>
                                    <thead><tr><th>Sig</th><th>Err</th></tr></thead>
                                    <tbody>
                                        {errorData.signal.slice(0, 4).map((r, i) => (
                                            <tr key={i}><td>{r.value}</td><td style={{ color: "#ff8f8f" }}>{fmt(r.error_rate)}</td></tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            <div style={{ padding: "0.6rem", borderRadius: "8px", background: "rgba(0,0,0,0.15)" }}>
                                <div style={{ fontSize: "0.65rem", color: "#8a8fb5", textTransform: "uppercase", marginBottom: "6px" }}>By Confidence</div>
                                <table className="tiny" style={{ fontSize: "0.7rem", marginTop: 0 }}>
                                    <thead><tr><th>Conf</th><th>Err</th></tr></thead>
                                    <tbody>
                                        {errorData.confidence.slice(0, 4).map((r, i) => (
                                            <tr key={i}><td>{r.confidence_bucket}</td><td style={{ color: "#ff8f8f" }}>{fmt(r.error_rate)}</td></tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            ) : (
                <EmptySection sectionNum={1} title="Model Benchmark Overview" tabName="Benchmarks" />
            )}

            {/* ─── Section 2: Backtest Results ─── */}
            {backtest ? (
                <div className="card" style={{ marginBottom: "1.5rem" }}>
                    <h3 style={{ borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "0.7rem", marginBottom: "1.2rem" }}>
                        2. Backtest Simulation Results
                    </h3>
                    <p className="muted" style={{ marginBottom: "1rem", fontSize: "0.82rem" }}>
                        Model: <strong style={{ color: "#818cf8" }}>{backtest.model}</strong> · Strategy: {backtest.strategy} · Period: {backtest.from_date} → {backtest.to_date}
                    </p>

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem", marginBottom: "1.5rem" }}>
                        {[
                            { label: "Total Return (NLP)", value: fmtPct(gated.total_return), color: (gated.total_return || 0) >= 0 ? "#51cf66" : "#ff6b6b" },
                            { label: "Total Return (Base)", value: fmtPct(baseline.total_return), color: "#8a8fb5" },
                            { label: "Sharpe Ratio", value: (gated.sharpe_ratio || 0).toFixed(2), color: "#818cf8" },
                            { label: "Max Drawdown", value: fmtPct(gated.max_drawdown), color: "#ff6b6b" },
                            { label: "Win Rate", value: fmtPct(gated.win_rate), color: "#51cf66" },
                            { label: "Final Balance", value: fmtCur(gated.final_balance), color: "#fff" },
                        ].map(({ label, value, color }) => (
                            <div key={label} style={{ padding: "0.8rem", borderRadius: "8px", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
                                <div style={{ fontSize: "0.68rem", color: "#8a8fb5", textTransform: "uppercase", marginBottom: "4px" }}>{label}</div>
                                <div style={{ fontSize: "1.1rem", fontWeight: 700, color }}>{value}</div>
                            </div>
                        ))}
                    </div>

                    {/* Inline Equity Curve */}
                    {backtest.equity_curve && (() => {
                        const dates = backtest.equity_curve.dates || [];
                        const baseLine = backtest.equity_curve.baseline || [];
                        const gatedLine = backtest.equity_curve.gated || [];
                        if (dates.length < 2) return null;
                        const allVals = [...baseLine, ...gatedLine];
                        const minV = Math.min(...allVals);
                        const maxV = Math.max(...allVals);
                        const range = maxV - minV || 1;
                        const W = 600, H = 160, PAD = 30;
                        const toX = (i) => PAD + (i / (dates.length - 1)) * (W - PAD * 2);
                        const toY = (v) => PAD + (1 - (v - minV) / range) * (H - PAD * 2);
                        const makePath = (arr) => arr.map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
                        return (
                            <div style={{ marginBottom: "1rem" }}>
                                <p style={{ fontSize: "0.75rem", color: "#8a8fb5", fontWeight: 600, marginBottom: "0.5rem" }}>Equity Curve Comparison</p>
                                <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, background: "rgba(0,0,0,0.15)", borderRadius: "8px" }}>
                                    <path d={makePath(baseLine)} fill="none" stroke="#555" strokeWidth="1.5" strokeDasharray="4,3" />
                                    <path d={makePath(gatedLine)} fill="none" stroke="#818cf8" strokeWidth="2" />
                                </svg>
                                <div style={{ display: "flex", justifyContent: "center", gap: "1.5rem", marginTop: "0.5rem" }}>
                                    <span style={{ fontSize: "0.7rem", color: "#666" }}>-- Baseline</span>
                                    <span style={{ fontSize: "0.7rem", color: "#818cf8" }}>━ NLP Augmented</span>
                                </div>
                            </div>
                        );
                    })()}
                </div>
            ) : (
                <EmptySection sectionNum={2} title="Backtest Simulation Results" tabName="Backtest" />
            )}

            {/* ─── Section 3: Live Headline Sentiment ─── */}
            {headlines.length > 0 ? (
                <div className="card" style={{ marginBottom: "1.5rem" }}>
                    <h3 style={{ borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "0.7rem", marginBottom: "1.2rem" }}>
                        3. Live Headline Sentiment Snapshot
                    </h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: "0.8rem" }}>
                        {headlines.map((h, i) => (
                            <div key={i} style={{
                                padding: "0.9rem 1rem",
                                borderRadius: "8px",
                                background: "rgba(255,255,255,0.03)",
                                border: "1px solid rgba(255,255,255,0.06)",
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "center",
                                gap: "1rem"
                            }}>
                                <div style={{ flex: 1 }}>
                                    <p style={{ margin: 0, fontSize: "0.85rem", color: "#e0e2ff", lineHeight: 1.4 }}>{h.title}</p>
                                    <p style={{ margin: "4px 0 0", fontSize: "0.7rem", color: "#8a8fb5" }}>
                                        {h.source} · {h.pubDate?.slice(0, 10)}
                                    </p>
                                </div>
                                <div style={{ display: "flex", alignItems: "center", gap: "0.8rem", flexShrink: 0 }}>
                                    <span style={{
                                        padding: "3px 10px", borderRadius: "20px", fontSize: "0.72rem", fontWeight: 700,
                                        background: h.sentiment === "Bullish" ? "rgba(81,207,102,0.15)" : h.sentiment === "Bearish" ? "rgba(255,107,107,0.15)" : "rgba(255,195,0,0.15)",
                                        color: h.sentiment === "Bullish" ? "#51cf66" : h.sentiment === "Bearish" ? "#ff6b6b" : "#ffc300",
                                    }}>{h.sentiment}</span>
                                    <span style={{ fontSize: "0.78rem", color: "#818cf8", fontWeight: 600 }}>{(h.confidence * 100).toFixed(0)}%</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            ) : (
                <EmptySection sectionNum={3} title="Live Headline Sentiment Snapshot" tabName="Live Analysis" />
            )}

            {/* ─── Section 4: AI Research Summary ─── */}
            {chatSummary ? (
                <div className="card" style={{ marginBottom: "1.5rem" }}>
                    <h3 style={{ borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "0.7rem", marginBottom: "1.2rem" }}>
                        4. AI-Generated Research Summary
                    </h3>
                    <div style={{
                        padding: "1.2rem",
                        borderRadius: "10px",
                        background: "rgba(129,140,248,0.04)",
                        border: "1px solid rgba(129,140,248,0.12)",
                        fontSize: "0.88rem",
                        color: "#c4c8f0",
                        lineHeight: 1.7
                    }}>
                        <ReactMarkdown>
                            {chatSummary}
                        </ReactMarkdown>
                    </div>
                </div>
            ) : (
                <EmptySection sectionNum={4} title="AI-Generated Research Summary" tabName="Market GPT" />
            )
            }

            {/* ─── Section 5: Final Verdict ─── */}
            {
                bestModel ? (
                    <div className="card" style={{ marginBottom: "1.5rem", background: "linear-gradient(135deg, rgba(129,140,248,0.08) 0%, rgba(81,207,102,0.04) 100%)", border: "1px solid rgba(129,140,248,0.2)" }}>
                        <h3 style={{ borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "0.7rem", marginBottom: "1.2rem" }}>
                            5. Final Verdict
                        </h3>
                        <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "1.5rem", alignItems: "center" }}>
                            <div style={{
                                width: 80, height: 80, borderRadius: "50%",
                                border: "3px solid #818cf8",
                                display: "flex", alignItems: "center", justifyContent: "center",
                                fontSize: "1.2rem", background: "rgba(129,140,248,0.08)",
                                fontWeight: 800, color: "#818cf8"
                            }}>TOP</div>
                            <div>
                                <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "#818cf8", marginBottom: "4px" }}>
                                    {bestModel.model}
                                </div>
                                <p style={{ color: "#c4c8f0", fontSize: "0.88rem", lineHeight: 1.6, margin: 0 }}>
                                    <strong>{bestModel.model}</strong> achieved the highest F1 Macro of <strong style={{ color: "#51cf66" }}>{fmt(bestModel.f1_macro)}</strong> with
                                    an accuracy of <strong>{fmt(bestModel.accuracy)}</strong> at <strong>{fmt(bestModel.latency_ms_per_tweet, 2)}ms</strong> per headline.
                                    {backtest && (
                                        <> When deployed as the NLP gating backbone in backtesting, it produced a total return of <strong style={{ color: (gated.total_return || 0) >= 0 ? "#51cf66" : "#ff6b6b" }}>{fmtPct(gated.total_return)}</strong> with
                                            a Sharpe Ratio of <strong style={{ color: "#818cf8" }}>{(gated.sharpe_ratio || 0).toFixed(2)}</strong>, compared to the baseline return
                                            of {fmtPct(baseline.total_return)}.</>
                                    )}
                                </p>
                                <div style={{ display: "flex", gap: "1rem", marginTop: "0.8rem" }}>
                                    {[
                                        { label: "F1 Macro", val: fmt(bestModel.f1_macro), color: "#818cf8" },
                                        { label: "Accuracy", val: fmt(bestModel.accuracy), color: "#51cf66" },
                                        { label: "Latency", val: fmt(bestModel.latency_ms_per_tweet, 2) + "ms", color: "#ff9f43" },
                                    ].map(t => (
                                        <div key={t.label} style={{ padding: "6px 14px", borderRadius: "8px", background: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                                            <div style={{ fontSize: "0.6rem", color: "#8a8fb5", textTransform: "uppercase" }}>{t.label}</div>
                                            <div style={{ fontSize: "0.95rem", fontWeight: 700, color: t.color }}>{t.val}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <EmptySection sectionNum={5} title="Final Verdict" tabName="Benchmarks" />
                )
            }

            {/* Report Disclaimer */}
            <p className="tiny muted" style={{ textAlign: "center", marginTop: "2rem", paddingBottom: "1rem" }}>
                This analysis is for research purposes only and does not constitute financial advice.
            </p>
        </div >
    );
}
