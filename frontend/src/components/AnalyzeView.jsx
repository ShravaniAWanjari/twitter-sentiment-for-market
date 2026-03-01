import { useState } from "react";

export default function AnalyzeView({ models }) {
    const [headlines, setHeadlines] = useState([]);
    const [selectedIndex, setSelectedIndex] = useState(-1);
    const [fetching, setFetching] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);
    const [result, setResult] = useState(null);
    const [fetchError, setFetchError] = useState("");

    const fetchHeadlines = async () => {
        setFetching(true);
        setFetchError("");
        setHeadlines([]);
        setSelectedIndex(-1);
        setResult(null);
        try {
            const res = await fetch("/api/news/bitcoin-headlines");
            if (!res.ok) throw new Error(`API returned ${res.status}`);
            const data = await res.json();
            const items = data.headlines || [];
            if (items.length === 0) {
                setFetchError("No headlines found. The API may be rate-limited or unavailable.");
            }
            setHeadlines(items);
        } catch (err) {
            console.error("Failed to fetch headlines", err);
            setFetchError("Failed to fetch live headlines. Check API key or network connection.");
        } finally {
            setFetching(false);
        }
    };

    const analyzeHeadline = async (text) => {
        if (!text) return;
        setAnalyzing(true);
        setResult(null);
        try {
            const res = await fetch("/api/news/analyze-headline", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, model: "modernbert" }),
            });
            if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);
            const data = await res.json();
            setResult(data);
        } catch (err) {
            console.error("Analysis failed", err);
        } finally {
            setAnalyzing(false);
        }
    };

    const handleSelect = (e) => {
        const idx = parseInt(e.target.value, 10);
        setSelectedIndex(idx);
        setResult(null);
        if (idx >= 0 && headlines[idx]) {
            analyzeHeadline(headlines[idx].title);
        }
    };

    const sentimentConfig = {
        Bullish: { color: "#51cf66", bg: "rgba(81,207,102,0.12)", icon: "📈", label: "BULLISH" },
        Bearish: { color: "#ff6b6b", bg: "rgba(255,107,107,0.12)", icon: "📉", label: "BEARISH" },
        Neutral: { color: "#fcc419", bg: "rgba(252,196,25,0.12)", label: "NEUTRAL" },
    };

    const getSentimentStyle = (s) => sentimentConfig[s] || sentimentConfig.Neutral;

    return (
        <div className="headline-analyzer">
            {/* ── Hero Section ── */}
            <section className="ha-hero card">
                <div className="ha-hero-left">
                    <div className="ha-icon-wrap">
                        <span className="ha-icon" style={{ fontSize: "1.5rem" }}>₿</span>
                    </div>
                    <div>
                        <h2 className="ha-title">Bitcoin Headline Analyzer</h2>
                        <p className="muted" style={{ margin: 0 }}>
                            Fetch live BTC headlines and analyze sentiment with <strong>ModernBERT</strong> — the top-performing model (F1 0.861)
                        </p>
                    </div>
                </div>
                <button
                    className="btn primary ha-fetch-btn"
                    onClick={fetchHeadlines}
                    disabled={fetching}
                    id="fetch-headlines-btn"
                >
                    {fetching ? (
                        <>
                            <span className="ha-spinner" />
                            Fetching...
                        </>
                    ) : (
                        <>Fetch Live Headlines</>
                    )}
                </button>
            </section>

            {fetchError && (
                <div className="ha-error-banner">
                    <span>ERROR:</span> {fetchError}
                </div>
            )}

            {/* ── Headline Selector ── */}
            {headlines.length > 0 && (
                <section className="ha-selector card">
                    <label className="ha-select-label">
                        Select a headline to analyze
                    </label>
                    <div className="ha-select-wrapper">
                        <select
                            value={selectedIndex}
                            onChange={handleSelect}
                            className="ha-select"
                            id="headline-dropdown"
                        >
                            <option value={-1}>— Choose from {headlines.length} live headlines —</option>
                            {headlines.map((h, i) => (
                                <option key={i} value={i}>
                                    {h.title.length > 100 ? h.title.slice(0, 100) + "…" : h.title}
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* Preview cards */}
                    <div className="ha-preview-grid">
                        {headlines.map((h, i) => (
                            <button
                                key={i}
                                className={`ha-preview-card ${selectedIndex === i ? "active" : ""}`}
                                onClick={() => { setSelectedIndex(i); setResult(null); analyzeHeadline(h.title); }}
                            >
                                <div className="ha-preview-num">{i + 1}</div>
                                <div className="ha-preview-body">
                                    <div className="ha-preview-title">{h.title}</div>
                                    <div className="ha-preview-meta">
                                        <span>{h.source}</span>
                                        {h.pubDate && <span>• {new Date(h.pubDate).toLocaleString()}</span>}
                                    </div>
                                </div>
                            </button>
                        ))}
                    </div>
                </section>
            )}

            {/* ── Loading State ── */}
            {analyzing && (
                <section className="ha-loading card">
                    <div className="ha-loading-inner">
                        <div className="ha-loading-ring" />
                        <div>
                            <h3 style={{ margin: 0 }}>Analyzing Sentiment…</h3>
                            <p className="muted" style={{ margin: "0.25rem 0 0" }}>
                                Running ModernBERT inference with occlusion attribution
                            </p>
                        </div>
                    </div>
                </section>
            )}

            {/* ── Results ── */}
            {result && !analyzing && (
                <div className="ha-results">
                    {/* Sentiment Verdict */}
                    <section className="ha-verdict card" style={{ borderLeft: `4px solid ${getSentimentStyle(result.sentiment).color}` }}>
                        <div className="ha-verdict-header">
                            <div className="ha-verdict-badge" style={{ background: getSentimentStyle(result.sentiment).bg, color: getSentimentStyle(result.sentiment).color }}>
                                <span className="ha-verdict-icon">{getSentimentStyle(result.sentiment).icon}</span>
                                <span className="ha-verdict-label">{getSentimentStyle(result.sentiment).label}</span>
                            </div>
                            <div className="ha-confidence-ring">
                                <svg viewBox="0 0 36 36" className="ha-ring-svg">
                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                        fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="3" />
                                    <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                        fill="none"
                                        stroke={getSentimentStyle(result.sentiment).color}
                                        strokeWidth="3"
                                        strokeDasharray={`${(result.confidence * 100).toFixed(0)}, 100`}
                                        strokeLinecap="round" />
                                </svg>
                                <div className="ha-ring-text">
                                    <strong>{Math.round(result.confidence * 100)}%</strong>
                                    <small>conf</small>
                                </div>
                            </div>
                        </div>
                        <div className="ha-verdict-headline">
                            "{result.headline}"
                        </div>
                        <div className="ha-verdict-model">
                            Analyzed by <strong>{result.model}</strong>
                        </div>
                    </section>

                    {/* Highlighted Text */}
                    {result.explainability?.highlighted_html && (
                        <section className="ha-highlight card">
                            <h3>🔍 Token Attribution Map</h3>
                            <p className="muted" style={{ marginTop: 0 }}>
                                Words highlighted by intensity — <span style={{ color: "#51cf66" }}>green = supports prediction</span>, <span style={{ color: "#ff8f8f" }}>red = opposes prediction</span>
                            </p>
                            <div className="highlighted-text" dangerouslySetInnerHTML={{ __html: result.explainability.highlighted_html }} />
                        </section>
                    )}

                    {/* Power Words Grid */}
                    <div className="ha-power-grid">
                        <section className="ha-power-card card">
                            <h3 style={{ color: "#51cf66" }}>↑ Top Positive Drivers</h3>
                            <p className="muted" style={{ marginTop: 0, fontSize: "0.8rem" }}>
                                Tokens that push the model toward <strong>{result.sentiment}</strong>
                            </p>
                            <div className="ha-token-list">
                                {(result.explainability?.top_positive || []).map((t, i) => (
                                    <div key={i} className="ha-token-row pos">
                                        <div className="ha-token-rank">{i + 1}</div>
                                        <div className="ha-token-word">{t.token}</div>
                                        <div className="ha-token-bar-wrap">
                                            <div className="ha-token-bar pos" style={{ width: `${Math.min(100, Math.abs(t.weight) * 300)}%` }} />
                                        </div>
                                        <div className="ha-token-weight">+{t.weight.toFixed(4)}</div>
                                    </div>
                                ))}
                                {(!result.explainability?.top_positive || result.explainability.top_positive.length === 0) && (
                                    <p className="muted">No significant positive drivers detected</p>
                                )}
                            </div>
                        </section>

                        <section className="ha-power-card card">
                            <h3 style={{ color: "#ff8f8f" }}>↓ Top Negative Drivers</h3>
                            <p className="muted" style={{ marginTop: 0, fontSize: "0.8rem" }}>
                                Tokens that oppose the predicted sentiment
                            </p>
                            <div className="ha-token-list">
                                {(result.explainability?.top_negative || []).map((t, i) => (
                                    <div key={i} className="ha-token-row neg">
                                        <div className="ha-token-rank">{i + 1}</div>
                                        <div className="ha-token-word">{t.token}</div>
                                        <div className="ha-token-bar-wrap">
                                            <div className="ha-token-bar neg" style={{ width: `${Math.min(100, Math.abs(t.weight) * 300)}%` }} />
                                        </div>
                                        <div className="ha-token-weight">{t.weight.toFixed(4)}</div>
                                    </div>
                                ))}
                                {(!result.explainability?.top_negative || result.explainability.top_negative.length === 0) && (
                                    <p className="muted">No significant negative drivers detected</p>
                                )}
                            </div>
                        </section>
                    </div>

                    {/* Stability + Counterfactual */}
                    <div className="ha-insight-grid">
                        {result.explainability?.stability && (
                            <section className="ha-insight-card card">
                                <h3>🛡️ Prediction Stability</h3>
                                <div className="ha-stability-meter">
                                    <div className="ha-stability-bar-track">
                                        <div
                                            className={`ha-stability-bar-fill ${result.explainability.stability.label}`}
                                            style={{ width: `${(result.explainability.stability.score_0_1 * 100).toFixed(0)}%` }}
                                        />
                                    </div>
                                    <div className="ha-stability-labels">
                                        <span>Fragile</span>
                                        <span className={`ha-stability-value ${result.explainability.stability.label}`}>
                                            {result.explainability.stability.label.toUpperCase()} ({(result.explainability.stability.score_0_1 * 100).toFixed(0)}%)
                                        </span>
                                        <span>Robust</span>
                                    </div>
                                </div>
                                <p className="muted" style={{ fontSize: "0.8rem", marginTop: "0.75rem" }}>
                                    Measures how much the prediction changes when the top token is removed.
                                    {result.explainability.stability.perturbations?.[0] && (
                                        <span> Removing the top word shifted confidence by <strong>{(Math.abs(result.explainability.stability.perturbations[0].diff) * 100).toFixed(1)}%</strong>.</span>
                                    )}
                                </p>
                            </section>
                        )}

                        {result.explainability?.counterfactual?.found && (
                            <section className="ha-insight-card card">
                                <h3>🔄 Counterfactual Flip</h3>
                                <p className="muted" style={{ marginTop: 0, fontSize: "0.85rem" }}>
                                    Replacing a single key word flips the entire prediction:
                                </p>
                                <div className="ha-cf-flow">
                                    <div className="ha-cf-original">
                                        <small>Original</small>
                                        <div className="ha-cf-text">"{result.headline}"</div>
                                        <span className="badge" style={{ background: getSentimentStyle(result.sentiment).bg, color: getSentimentStyle(result.sentiment).color }}>
                                            {result.sentiment}
                                        </span>
                                    </div>
                                    <div className="ha-cf-arrow">→</div>
                                    <div className="ha-cf-flipped">
                                        <small>Modified</small>
                                        <div className="ha-cf-text">"{result.explainability.counterfactual.edited_text}"</div>
                                        <span className="badge" style={{
                                            background: getSentimentStyle(result.explainability.counterfactual.flipped_label).bg,
                                            color: getSentimentStyle(result.explainability.counterfactual.flipped_label).color
                                        }}>
                                            {result.explainability.counterfactual.flipped_label}
                                        </span>
                                    </div>
                                </div>
                            </section>
                        )}
                    </div>
                </div>
            )}

            {/* ── Empty state ── */}
            {!fetching && headlines.length === 0 && !result && (
                <section className="ha-empty card">
                    <div className="ha-empty-inner">
                        <div className="ha-empty-icon">₿</div>
                        <h3>Ready to Analyze</h3>
                        <p className="muted">
                            Click <strong>"Fetch Live Headlines"</strong> to pull the latest Bitcoin news, then select a headline to run deep sentiment analysis.
                        </p>
                    </div>
                </section>
            )}

            <style>{`
                .headline-analyzer {
                    margin-top: 1rem;
                }

                /* Hero */
                .ha-hero {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 1.5rem;
                    flex-wrap: wrap;
                    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(245,158,11,0.06)) !important;
                    border: 1px solid rgba(99,102,241,0.2) !important;
                }
                .ha-hero-left {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                }
                .ha-icon-wrap {
                    width: 52px;
                    height: 52px;
                    border-radius: 14px;
                    background: linear-gradient(135deg, #f59e0b, #f97316);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.5rem;
                    box-shadow: 0 4px 20px rgba(245,158,11,0.3);
                    flex-shrink: 0;
                }
                .ha-icon { filter: brightness(10); }
                .ha-title {
                    margin: 0;
                    font-size: 1.35rem;
                    background: linear-gradient(135deg, #fff, #c9cdfb);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                .ha-fetch-btn {
                    gap: 0.5rem;
                    padding: 0.75rem 1.5rem !important;
                    font-size: 0.95rem !important;
                    white-space: nowrap;
                }
                .ha-spinner {
                    width: 16px; height: 16px;
                    border: 2px solid rgba(255,255,255,0.2);
                    border-top-color: #fff;
                    border-radius: 50%;
                    animation: spin 0.6s linear infinite;
                }
                @keyframes spin { to { transform: rotate(360deg); } }

                /* Error */
                .ha-error-banner {
                    margin-top: 1rem;
                    padding: 0.75rem 1rem;
                    background: rgba(239,68,68,0.1);
                    border: 1px solid rgba(239,68,68,0.3);
                    border-radius: 10px;
                    color: #fca5a5;
                    font-size: 0.9rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                /* Selector */
                .ha-selector {
                    padding-bottom: 1.5rem !important;
                }
                .ha-select-label {
                    display: block;
                    font-size: 0.8rem;
                    color: #9aa0ff;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    margin-bottom: 0.75rem;
                }
                .ha-select-wrapper {
                    position: relative;
                }
                .ha-select {
                    width: 100%;
                    background: #0c0d16;
                    border: 1px solid #2e3158;
                    color: #eef0ff;
                    padding: 0.75rem 1rem;
                    border-radius: 10px;
                    font-size: 0.9rem;
                    cursor: pointer;
                    appearance: none;
                    transition: border-color 0.2s;
                }
                .ha-select:focus {
                    outline: none;
                    border-color: #6366f1;
                }
                .ha-select-wrapper::after {
                    content: "▾";
                    position: absolute;
                    right: 1rem;
                    top: 50%;
                    transform: translateY(-50%);
                    color: #6366f1;
                    pointer-events: none;
                }

                /* Preview Cards */
                .ha-preview-grid {
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                    margin-top: 1rem;
                }
                .ha-preview-card {
                    display: flex;
                    align-items: flex-start;
                    gap: 0.75rem;
                    padding: 0.85rem 1rem;
                    background: rgba(255,255,255,0.02);
                    border: 1px solid #1f2240;
                    border-radius: 10px;
                    cursor: pointer;
                    transition: all 0.2s;
                    text-align: left;
                    color: #eef0ff;
                    width: 100%;
                    font-family: inherit;
                    font-size: inherit;
                }
                .ha-preview-card:hover {
                    background: rgba(99,102,241,0.06);
                    border-color: rgba(99,102,241,0.3);
                    transform: translateX(4px);
                }
                .ha-preview-card.active {
                    background: rgba(99,102,241,0.1);
                    border-color: #6366f1;
                    box-shadow: 0 0 20px rgba(99,102,241,0.15);
                }
                .ha-preview-num {
                    width: 28px; height: 28px;
                    border-radius: 8px;
                    background: #1f2240;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 700;
                    font-size: 0.8rem;
                    color: #6366f1;
                    flex-shrink: 0;
                }
                .ha-preview-card.active .ha-preview-num {
                    background: #6366f1;
                    color: #fff;
                }
                .ha-preview-body { flex: 1; min-width: 0; }
                .ha-preview-title {
                    font-size: 0.88rem;
                    line-height: 1.4;
                    color: #eef0ff;
                }
                .ha-preview-meta {
                    font-size: 0.72rem;
                    color: #8a8fb5;
                    margin-top: 0.25rem;
                    display: flex;
                    gap: 0.5rem;
                }

                /* Loading */
                .ha-loading {
                    border: 1px dashed rgba(99,102,241,0.3) !important;
                    background: rgba(99,102,241,0.04) !important;
                }
                .ha-loading-inner {
                    display: flex;
                    align-items: center;
                    gap: 1.25rem;
                }
                .ha-loading-ring {
                    width: 40px; height: 40px;
                    border: 3px solid rgba(99,102,241,0.15);
                    border-top-color: #6366f1;
                    border-radius: 50%;
                    animation: spin 0.8s linear infinite;
                    flex-shrink: 0;
                }

                /* Verdict */
                .ha-verdict {
                    animation: fadeIn 0.4s ease;
                }
                .ha-verdict-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1rem;
                }
                .ha-verdict-badge {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    padding: 0.6rem 1.25rem;
                    border-radius: 12px;
                    font-weight: 700;
                    font-size: 1.1rem;
                    letter-spacing: 0.05em;
                }
                .ha-verdict-icon { font-size: 1.5rem; }
                .ha-verdict-headline {
                    font-size: 1.05rem;
                    color: #d1d4ff;
                    font-style: italic;
                    line-height: 1.5;
                    padding: 0.75rem 0;
                    border-top: 1px solid rgba(255,255,255,0.05);
                    border-bottom: 1px solid rgba(255,255,255,0.05);
                }
                .ha-verdict-model {
                    margin-top: 0.5rem;
                    font-size: 0.8rem;
                    color: #8a8fb5;
                }

                /* Confidence Ring */
                .ha-confidence-ring {
                    position: relative;
                    width: 72px; height: 72px;
                }
                .ha-ring-svg {
                    width: 100%; height: 100%;
                    transform: rotate(-90deg);
                }
                .ha-ring-text {
                    position: absolute;
                    inset: 0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    font-size: 0.7rem;
                    color: #eef0ff;
                }
                .ha-ring-text strong { font-size: 1rem; line-height: 1; }
                .ha-ring-text small { color: #8a8fb5; font-size: 0.6rem; }

                /* Highlight */
                .ha-highlight { animation: fadeIn 0.5s ease 0.1s both; }
                .ha-highlight h3 { margin-top: 0; }

                /* Power Words */
                .ha-power-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1.5rem;
                }
                .ha-power-card {
                    animation: fadeIn 0.5s ease 0.2s both;
                }
                .ha-power-card h3 { margin-top: 0; font-size: 1rem; }
                .ha-token-list {
                    display: flex;
                    flex-direction: column;
                    gap: 0.4rem;
                }
                .ha-token-row {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 0.5rem 0.6rem;
                    border-radius: 8px;
                    transition: background 0.2s;
                }
                .ha-token-row:hover { background: rgba(255,255,255,0.04); }
                .ha-token-row.pos { border-left: 3px solid rgba(81,207,102,0.4); }
                .ha-token-row.neg { border-left: 3px solid rgba(255,143,143,0.4); }
                .ha-token-rank {
                    width: 22px; height: 22px;
                    border-radius: 6px;
                    background: rgba(255,255,255,0.05);
                    display: flex; align-items: center; justify-content: center;
                    font-size: 0.7rem; font-weight: 700;
                    color: #8a8fb5;
                    flex-shrink: 0;
                }
                .ha-token-word {
                    font-family: "JetBrains Mono", monospace;
                    font-size: 0.9rem;
                    min-width: 80px;
                }
                .ha-token-bar-wrap {
                    flex: 1;
                    height: 6px;
                    background: rgba(255,255,255,0.04);
                    border-radius: 3px;
                    overflow: hidden;
                }
                .ha-token-bar {
                    height: 100%;
                    border-radius: 3px;
                    transition: width 0.6s ease;
                }
                .ha-token-bar.pos { background: linear-gradient(90deg, #51cf66, #40c057); }
                .ha-token-bar.neg { background: linear-gradient(90deg, #ff6b6b, #fa5252); }
                .ha-token-weight {
                    font-family: "JetBrains Mono", monospace;
                    font-size: 0.75rem;
                    color: #8a8fb5;
                    min-width: 55px;
                    text-align: right;
                }

                /* Insight Grid */
                .ha-insight-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1.5rem;
                }
                .ha-insight-card {
                    animation: fadeIn 0.5s ease 0.3s both;
                }
                .ha-insight-card h3 { margin-top: 0; font-size: 1rem; }

                /* Stability */
                .ha-stability-meter { margin-top: 0.75rem; }
                .ha-stability-bar-track {
                    height: 10px;
                    background: rgba(255,255,255,0.05);
                    border-radius: 5px;
                    overflow: hidden;
                }
                .ha-stability-bar-fill {
                    height: 100%;
                    border-radius: 5px;
                    transition: width 0.8s ease;
                }
                .ha-stability-bar-fill.high { background: linear-gradient(90deg, #51cf66, #40c057); }
                .ha-stability-bar-fill.medium { background: linear-gradient(90deg, #fcc419, #fab005); }
                .ha-stability-bar-fill.low { background: linear-gradient(90deg, #ff6b6b, #fa5252); }
                .ha-stability-labels {
                    display: flex;
                    justify-content: space-between;
                    font-size: 0.7rem;
                    color: #8a8fb5;
                    margin-top: 0.35rem;
                }
                .ha-stability-value { font-weight: 700; }
                .ha-stability-value.high { color: #51cf66; }
                .ha-stability-value.medium { color: #fcc419; }
                .ha-stability-value.low { color: #ff6b6b; }

                /* Counterfactual */
                .ha-cf-flow {
                    display: flex;
                    align-items: stretch;
                    gap: 1rem;
                    margin-top: 0.75rem;
                }
                .ha-cf-original, .ha-cf-flipped {
                    flex: 1;
                    padding: 0.75rem;
                    background: rgba(255,255,255,0.03);
                    border-radius: 8px;
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }
                .ha-cf-original small, .ha-cf-flipped small {
                    font-size: 0.7rem;
                    color: #8a8fb5;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                }
                .ha-cf-text {
                    font-size: 0.85rem;
                    color: #d1d4ff;
                    font-style: italic;
                    flex: 1;
                }
                .ha-cf-arrow {
                    display: flex;
                    align-items: center;
                    font-size: 1.5rem;
                    color: #6366f1;
                }

                /* Empty */
                .ha-empty {
                    min-height: 300px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .ha-empty-inner {
                    text-align: center;
                }
                .ha-empty-icon {
                    font-size: 3.5rem;
                    margin-bottom: 1rem;
                    animation: pulse 2s infinite;
                }

                /* Responsive */
                @media (max-width: 768px) {
                    .ha-power-grid,
                    .ha-insight-grid {
                        grid-template-columns: 1fr;
                    }
                    .ha-cf-flow {
                        flex-direction: column;
                    }
                    .ha-cf-arrow {
                        justify-content: center;
                        transform: rotate(90deg);
                    }
                }
            `}</style>
        </div>
    );
}
