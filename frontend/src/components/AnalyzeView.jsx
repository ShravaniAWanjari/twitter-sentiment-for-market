import { useState } from "react";

export default function AnalyzeView({ models }) {
    const [selectedModel, setSelectedModel] = useState(models[0] || "modernbert");
    const [headlines, setHeadlines] = useState([]);
    const [analyzing, setAnalyzing] = useState(false);
    const [chatQuery, setChatQuery] = useState("");
    const [chatHistory, setChatHistory] = useState([]);
    const [snapshot, setSnapshot] = useState(null);

    useState(() => {
        fetch("/api/backtest/latest").then(res => res.json()).then(data => setSnapshot(data)).catch(() => { });
    }, []);

    const fetchLive = async () => {
        setAnalyzing(true);
        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: selectedModel }),
            });
            const data = await res.json();
            setHeadlines(data);
        } finally {
            setAnalyzing(false);
        }
    };

    const sendChat = async () => {
        if (!chatQuery) return;
        const userMsg = { role: "user", text: chatQuery };
        setChatHistory(prev => [...prev, userMsg]);
        setChatQuery("");

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: chatQuery }),
            });
            const data = await res.json();
            setChatHistory(prev => [...prev, { role: "assistant", text: data.response }]);
        } catch (err) {
            setChatHistory(prev => [...prev, { role: "assistant", text: "Error connecting to chat." }]);
        }
    };

    try {
        return (
            <div className="analyze-view">
                <div className="grid two">
                    <section className="card">
                        <div className="flex-between">
                            <h2>Market Sentiment</h2>
                            <div className="flex-group">
                                <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                                    {(models || []).map(m => <option key={m} value={m}>{m}</option>)}
                                </select>
                                <button className="btn primary" onClick={fetchLive} disabled={analyzing}>
                                    {analyzing ? "Analyzing..." : "Fetch & Analyze News"}
                                </button>
                            </div>
                        </div>

                        <div className="headline-list">
                            {(headlines || []).map((h, i) => (
                                <div key={i} className="headline-item card">
                                    <div className="flex-between">
                                        <strong>{h?.headline || "Untitled Headline"}</strong>
                                        <span className={`badge ${(h?.sentiment || "neutral").toLowerCase()}`}>
                                            {h?.sentiment || "Neutral"} ({Math.round((h?.confidence || 0) * 100)}%)
                                        </span>
                                    </div>
                                    <div className="token-explain">
                                        {(h?.tokens || []).map((t, ti) => (
                                            <span key={ti} className="token" style={{ backgroundColor: `rgba(77, 171, 247, ${(t?.score || 0) * 0.5})`, borderBottom: (t?.score || 0) > 0.7 ? "2px solid #4dabf7" : "none" }}>
                                                {(t?.token || "").replace("Ġ", " ")}
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className="actions" style={{ marginTop: "1rem" }}>
                            <button className="btn" onClick={() => window.open("/api/pdf")}>Download Analysis PDF</button>
                        </div>
                    </section>

                    <div className="right-panel flex-column" style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                        {snapshot && snapshot.metrics && (
                            <section className="card snapshot-card">
                                <h3>Latest Backtest Snapshot</h3>
                                <div className="flex-between">
                                    <div>
                                        <small>Strategy</small>
                                        <div><strong>{snapshot?.strategy || "Unknown"} ({snapshot?.model || "Unknown"})</strong></div>
                                    </div>
                                    <div>
                                        <small>Gated Sharpe</small>
                                        <div className="highlight-success">{(snapshot?.metrics?.gated?.sharpe_ratio || 0).toFixed(2)}</div>
                                    </div>
                                </div>
                            </section>
                        )}

                        <section className="card chat-section">
                            <h2>Market GPT (Grounded)</h2>
                            <div className="chat-box">
                                {chatHistory.length === 0 && <p className="muted">Ask about latest performance or strategy...</p>}
                                {chatHistory.map((m, i) => (
                                    <div key={i} className={`chat-msg ${m?.role || "assistant"}`}>
                                        <strong>{m?.role === "user" ? "You" : "Assistant"}:</strong> {m?.text || ""}
                                    </div>
                                ))}
                            </div>
                            <div className="chat-input-group">
                                <input
                                    type="text" value={chatQuery} onChange={(e) => setChatQuery(e.target.value)}
                                    placeholder="Ask about the results..."
                                    onKeyDown={(e) => e.key === "Enter" && sendChat()}
                                />
                                <button className="btn" onClick={sendChat}>Send</button>
                            </div>
                        </section>
                    </div>
                </div>
            </div>
        );
    } catch (e) {
        console.error("AnalyzeView fallback render", e);
        return <div className="card">Error loading analysis view.</div>;
    }
}
