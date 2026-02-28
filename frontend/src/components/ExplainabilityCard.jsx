import { useState } from "react";

// ─── Mini horizontal bar chart rendered purely in SVG ───────────────────────
function TokenBarChart({ tokens = [], maxItems = 6, height = 180 }) {
    if (!tokens || tokens.length === 0) return <p style={{ color: "#666", fontSize: "0.8rem" }}>No data</p>;
    const items = tokens.slice(0, maxItems);
    const maxW = Math.max(...items.map(t => Math.abs(t.weight || 0)), 0.001);
    const rowH = Math.floor(height / items.length);
    const labelW = 100, barMaxW = 160, W = labelW + barMaxW + 50;

    return (
        <svg viewBox={`0 0 ${W} ${height}`} style={{ width: "100%", maxWidth: W, height }}>
            {items.map((t, i) => {
                const w = t.weight || 0;
                const barW = (Math.abs(w) / maxW) * barMaxW;
                const isPos = w >= 0;
                const color = isPos ? "#51cf66" : "#ff6b6b";
                const y = i * rowH + rowH * 0.15;
                const bH = rowH * 0.6;
                return (
                    <g key={i}>
                        <text x={labelW - 6} y={y + bH * 0.7} textAnchor="end"
                            fontSize="10" fill="#c4c8f0" fontFamily="monospace">
                            {t.token?.slice(0, 12)}
                        </text>
                        <rect x={labelW} y={y} width={barW} height={bH}
                            rx="2" fill={color} opacity="0.75" />
                        <text x={labelW + barW + 4} y={y + bH * 0.75}
                            fontSize="9" fill={color} fontWeight="600">
                            {w > 0 ? "+" : ""}{w.toFixed(3)}
                        </text>
                    </g>
                );
            })}
        </svg>
    );
}

// ─── Method Agreement heatmap (top tokens × 3 methods) ──────────────────────
function AgreementHeatmap({ agreement = [] }) {
    if (!agreement.length) return null;
    const methods = ["occlusion", "integrated_gradients", "grad_input"];
    const labels = { occlusion: "Occlusion", integrated_gradients: "Integ. Grad.", grad_input: "Grad\u00d7Input" };
    const colW = 80, rowH = 26, labelW = 96, W = labelW + colW * 3 + 60, H = 28 + agreement.length * rowH;

    return (
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, height: H }}>
            {/* Header */}
            {methods.map((m, i) => (
                <text key={m} x={labelW + colW * i + colW / 2} y={16}
                    textAnchor="middle" fontSize="9.5" fill="#8a8fb5" fontWeight="600">
                    {labels[m]}
                </text>
            ))}
            {/* Rows */}
            {agreement.map((row, ri) => {
                const y = 24 + ri * rowH;
                return (
                    <g key={ri}>
                        {/* Agreed indicator */}
                        <text x={4} y={y + rowH * 0.68} fontSize="9" fill={row.all_agree ? "#51cf66" : "#ff9f43"}>
                            {row.all_agree ? "✓" : "~"}
                        </text>
                        <text x={18} y={y + rowH * 0.68} fontSize="10" fill="#c4c8f0" fontFamily="monospace">
                            {row.token?.slice(0, 11)}
                        </text>
                        {methods.map((m, ci) => {
                            const val = row[m];
                            if (val == null) return (
                                <rect key={m} x={labelW + colW * ci + 4} y={y + 2}
                                    width={colW - 8} height={rowH - 6} rx="3"
                                    fill="rgba(255,255,255,0.04)" />
                            );
                            const isPos = val >= 0;
                            const intensity = Math.min(0.85, Math.abs(val) * 2.5);
                            const fill = isPos
                                ? `rgba(81,207,102,${0.1 + intensity * 0.6})`
                                : `rgba(255,107,107,${0.1 + intensity * 0.6})`;
                            return (
                                <g key={m}>
                                    <rect x={labelW + colW * ci + 4} y={y + 2}
                                        width={colW - 8} height={rowH - 6} rx="3" fill={fill} />
                                    <text x={labelW + colW * ci + colW / 2 - 2} y={y + rowH * 0.72}
                                        textAnchor="middle" fontSize="9.5"
                                        fill={isPos ? "#a3f0b5" : "#ffb3b3"} fontWeight="600">
                                        {val > 0 ? "+" : ""}{val.toFixed(3)}
                                    </text>
                                </g>
                            );
                        })}
                    </g>
                );
            })}
        </svg>
    );
}

// ─── Model Sensitivity gauge (replaces Perturbation Test) ────────────────────
function SensitivityGauge({ sensitivity = {} }) {
    const drop = sensitivity.confidence_drop_pct ?? 0;
    const risk = sensitivity.risk_level || "low";
    const color = risk === "high" ? "#ff6b6b" : risk === "medium" ? "#ff9f43" : "#51cf66";
    const pct = Math.min(100, drop);
    const radius = 44, cx = 60, cy = 56, strokeW = 10;
    const circ = Math.PI * radius;  // half-circle
    const dashOffset = circ * (1 - pct / 100);

    return (
        <div style={{ display: "flex", gap: "1.2rem", alignItems: "flex-start" }}>
            {/* Semicircle gauge */}
            <svg width="120" height="68" viewBox="0 0 120 68" style={{ flexShrink: 0 }}>
                {/* Track */}
                <path d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
                    fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={strokeW} strokeLinecap="round" />
                {/* Fill */}
                <path d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
                    fill="none" stroke={color} strokeWidth={strokeW} strokeLinecap="round"
                    strokeDasharray={`${circ} ${circ}`}
                    strokeDashoffset={dashOffset}
                    style={{ transition: "stroke-dashoffset 0.6s ease" }} />
                {/* Label */}
                <text x={cx} y={cy - 8} textAnchor="middle" fontSize="18" fontWeight="800" fill={color}>
                    {drop.toFixed(0)}%
                </text>
                <text x={cx} y={cy + 6} textAnchor="middle" fontSize="9" fill="#8a8fb5">
                    confidence drop
                </text>
                <text x={cx - radius + 2} y={cy + 14} fontSize="8" fill="#666">0%</text>
                <text x={cx + radius - 20} y={cy + 14} fontSize="8" fill="#666">100%</text>
            </svg>

            <div style={{ paddingTop: "0.4rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
                    <span style={{
                        padding: "2px 8px", borderRadius: "20px", fontSize: "0.72rem",
                        fontWeight: 700, background: `${color}22`, color,
                        border: `1px solid ${color}44`
                    }}>{risk.toUpperCase()} SENSITIVITY</span>
                </div>
                {sensitivity.key_driver && (
                    <p style={{ fontSize: "0.78rem", color: "#c4c8f0", margin: "4px 0" }}>
                        Key driver: <strong style={{ color: "#818cf8" }}>"{sensitivity.key_driver}"</strong>
                    </p>
                )}
                <p style={{ fontSize: "0.75rem", color: "#8a8fb5", margin: 0, lineHeight: 1.4 }}>
                    {sensitivity.interpretation}
                </p>
            </div>
        </div>
    );
}

// ─── Per-method tab pane ─────────────────────────────────────────────────────
function MethodPane({ method, data }) {
    if (!data) return null;
    return (
        <div>
            {/* Method description */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.6rem", marginBottom: "1rem" }}>
                <div style={{ padding: "0.6rem 0.8rem", borderRadius: "8px", background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.2)" }}>
                    <p style={{ fontSize: "0.7rem", color: "#818cf8", marginBottom: "2px", fontWeight: 600 }}>⚙ How it works</p>
                    <p style={{ fontSize: "0.75rem", color: "#c4c8f0", margin: 0 }}>{data.description}</p>
                </div>
                <div style={{ padding: "0.6rem 0.8rem", borderRadius: "8px", background: "rgba(255,193,7,0.06)", border: "1px solid rgba(255,193,7,0.15)" }}>
                    <p style={{ fontSize: "0.7rem", color: "#ffc107", marginBottom: "2px", fontWeight: 600 }}>💹 Finance analogy</p>
                    <p style={{ fontSize: "0.75rem", color: "#c4c8f0", margin: 0 }}>{data.finance_note}</p>
                </div>
            </div>

            {/* Bullish / Bearish bars */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                <div>
                    <p style={{ fontSize: "0.75rem", color: "#51cf66", fontWeight: 600, marginBottom: "0.4rem" }}>
                        📈 Bullish Drivers
                    </p>
                    <TokenBarChart tokens={data.top_bullish} maxItems={5} height={130} />
                </div>
                <div>
                    <p style={{ fontSize: "0.75rem", color: "#ff6b6b", fontWeight: 600, marginBottom: "0.4rem" }}>
                        📉 Bearish Signals
                    </p>
                    <TokenBarChart tokens={data.top_bearish} maxItems={5} height={130} />
                </div>
            </div>

            {/* Highlighted HTML (only for Occlusion — it has the HTML) */}
            {method === "occlusion" && data.highlighted_html && (
                <div style={{ marginTop: "0.8rem", padding: "0.7rem", borderRadius: "6px", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", fontSize: "0.85rem", lineHeight: 1.8 }}
                    dangerouslySetInnerHTML={{ __html: data.highlighted_html }} />
            )}
        </div>
    );
}

// ─── Main ExplainabilityCard ─────────────────────────────────────────────────
export default function ExplainabilityCard({ result, originalText }) {
    const [activeMethod, setActiveMethod] = useState("occlusion");
    const [multiResult, setMultiResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [explainModel, setExplainModel] = useState(null);

    if (!result) return null;

    // Detect if this is multi-result or single-method result
    const isMulti = !!result.methods;

    const {
        highlighted_html,
        top_positive = [],
        top_negative = [],
        stability = {},
        counterfactual = {},
        method = "occlusion",
        model_sensitivity,
    } = result;

    const runMultiExplain = async (modelId, text) => {
        setLoading(true);
        try {
            const res = await fetch("/api/explain/multi", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_id: modelId, text }),
            });
            if (!res.ok) throw new Error("Multi-explain failed");
            const data = await res.json();
            setMultiResult(data);
            setActiveMethod("occlusion");
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    // If we have a multi result, render the rich view
    if (isMulti || multiResult) {
        const mr = multiResult || result;
        const methodKeys = ["occlusion", "integrated_gradients", "grad_input"];
        const methodLabels = { occlusion: "Occlusion", integrated_gradients: "Integrated Gradients", grad_input: "Grad \u00d7 Input" };
        const methodIcons = { occlusion: "🎭", integrated_gradients: "∇", grad_input: "∂f·x" };

        return (
            <div className="explainability-card card" style={{ marginTop: "1.5rem" }}>
                {/* Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.2rem" }}>
                    <div>
                        <h3 style={{ margin: 0 }}>Multi-Method Explainability</h3>
                        <p style={{ fontSize: "0.78rem", color: "#8a8fb5", margin: "4px 0 0" }}>
                            Prediction: <strong style={{
                                color: mr.base_prediction === "Bullish" ? "#51cf66" : mr.base_prediction === "Bearish" ? "#ff6b6b" : "#ffc107"
                            }}>{mr.base_prediction}</strong>
                            &nbsp;·&nbsp;
                            Confidence: <strong style={{ color: "#818cf8" }}>{((mr.base_confidence || 0) * 100).toFixed(1)}%</strong>
                        </p>
                    </div>
                    <p style={{ fontSize: "0.7rem", color: "#555", margin: 0, textAlign: "right", maxWidth: "200px" }}>
                        "{(mr.text || originalText || "").slice(0, 60)}…"
                    </p>
                </div>

                {/* Method tabs */}
                <div style={{ display: "flex", gap: "6px", marginBottom: "1rem", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "0.6rem" }}>
                    {methodKeys.map(m => (
                        <button key={m} onClick={() => setActiveMethod(m)}
                            style={{
                                padding: "5px 12px", borderRadius: "20px", fontSize: "0.78rem",
                                cursor: "pointer", border: "none", fontWeight: activeMethod === m ? 700 : 400,
                                background: activeMethod === m ? "rgba(99,102,241,0.25)" : "rgba(255,255,255,0.05)",
                                color: activeMethod === m ? "#818cf8" : "#8a8fb5",
                                transition: "all 0.15s",
                            }}>
                            {methodIcons[m]} {methodLabels[m]}
                        </button>
                    ))}
                </div>

                {/* Active method pane */}
                <MethodPane method={activeMethod} data={mr.methods?.[activeMethod]} />

                {/* Method Agreement heatmap */}
                {mr.method_comparison?.length > 0 && (
                    <div style={{ marginTop: "1.5rem", padding: "1rem", borderRadius: "8px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                        <p style={{ fontSize: "0.78rem", color: "#8a8fb5", marginBottom: "0.6rem", fontWeight: 600 }}>
                            📊 Cross-Method Token Agreement
                            <span style={{ fontSize: "0.68rem", fontWeight: 400, marginLeft: "8px" }}>
                                ✓ = all methods agree on direction &nbsp; ~ = methods disagree
                            </span>
                        </p>
                        <AgreementHeatmap agreement={mr.method_comparison} />
                    </div>
                )}

                {/* Model Sensitivity gauge */}
                {(mr.model_sensitivity || model_sensitivity) && (
                    <div style={{ marginTop: "1.2rem", padding: "1rem", borderRadius: "8px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}>
                        <p style={{ fontSize: "0.78rem", color: "#8a8fb5", marginBottom: "0.7rem", fontWeight: 600 }}>
                            ⚡ Model Sensitivity
                            <span style={{ fontSize: "0.68rem", fontWeight: 400, marginLeft: "8px" }}>
                                How much does removing the key word change the outcome?
                            </span>
                        </p>
                        <SensitivityGauge sensitivity={mr.model_sensitivity || model_sensitivity || {}} />
                    </div>
                )}

                {/* Counterfactual */}
                {(mr.counterfactual?.found) && (
                    <div style={{ marginTop: "1rem", padding: "0.8rem 1rem", borderRadius: "8px", background: "rgba(255,107,107,0.07)", border: "1px solid rgba(255,107,107,0.2)" }}>
                        <p style={{ fontSize: "0.78rem", color: "#ff8fa3", marginBottom: "4px", fontWeight: 600 }}>
                            🔄 Counterfactual Flip
                        </p>
                        <p style={{ fontSize: "0.78rem", color: "#c4c8f0", margin: "0 0 4px", fontStyle: "italic" }}>
                            "{mr.counterfactual.edited_text}"
                        </p>
                        <p style={{ fontSize: "0.75rem", color: "#ff6b6b", margin: 0 }}>
                            → Flips to <strong>{mr.counterfactual.flipped_label}</strong> at {((mr.counterfactual.new_score || 0) * 100).toFixed(1)}% confidence
                        </p>
                    </div>
                )}
            </div>
        );
    }

    // ── Single-method fallback (original ExplainabilityCard) + upgrade button ──
    return (
        <div className="explainability-card card">
            <div className="explain-header flex-between">
                <span className="muted">Explainability Method: <strong>{method}</strong></span>
                <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
                    {stability.score_0_1 !== undefined && (
                        <div className="stability-badge" title={`Score: ${stability.score_0_1.toFixed(2)}`}>
                            Stability: <span className={`status-${stability.label}`}>{stability.label?.toUpperCase()}</span>
                        </div>
                    )}
                    <button
                        className="btn mini"
                        onClick={() => runMultiExplain(explainModel || "modernbert", originalText)}
                        disabled={loading || !originalText}
                        style={{ fontSize: "0.72rem" }}
                    >
                        {loading ? "Running…" : "⚡ Deep Analysis (3 Methods)"}
                    </button>
                </div>
            </div>

            {highlighted_html && (
                <div className="highlighted-text" dangerouslySetInnerHTML={{ __html: highlighted_html }} />
            )}

            <div className="explain-grid">
                <div className="token-column">
                    <h4>Top Positive</h4>
                    <ul className="token-list">
                        {top_positive.map((t, i) => (
                            <li key={i} className="token-item pos">
                                <span>{t.token}</span>
                                <span className="weight">+{t.weight.toFixed(3)}</span>
                            </li>
                        ))}
                        {top_positive.length === 0 && <li className="muted">None found</li>}
                    </ul>
                </div>
                <div className="token-column">
                    <h4>Top Negative</h4>
                    <ul className="token-list">
                        {top_negative.map((t, i) => (
                            <li key={i} className="token-item neg">
                                <span>{t.token}</span>
                                <span className="weight">{t.weight.toFixed(3)}</span>
                            </li>
                        ))}
                        {top_negative.length === 0 && <li className="muted">None found</li>}
                    </ul>
                </div>
            </div>

            {counterfactual.found && (
                <div className="counterfactual-box">
                    <h4>Counterfactual Flip</h4>
                    <p className="muted">What would happen if we changed a key word?</p>
                    <div className="cf-content">
                        <div className="cf-text">"{counterfactual.edited_text}"</div>
                        <div className="cf-arrow">→</div>
                        <div className="cf-result">
                            <span className={`badge ${counterfactual.flipped_label?.toLowerCase()}`}>
                                {counterfactual.flipped_label}
                            </span>
                            <small>Confidence: {((counterfactual.new_score || 0) * 100).toFixed(1)}%</small>
                        </div>
                    </div>
                </div>
            )}

            {/* Model Sensitivity replaces raw Perturbation Test */}
            {model_sensitivity && (
                <div className="stability-details" style={{ marginTop: "1rem" }}>
                    <h4>Model Sensitivity</h4>
                    <SensitivityGauge sensitivity={model_sensitivity} />
                </div>
            )}
        </div>
    );
}
