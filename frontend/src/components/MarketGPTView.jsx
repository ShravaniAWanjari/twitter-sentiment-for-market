import { useState, useEffect, useRef } from "react";

/* ── Simple Markdown-to-JSX renderer ── */
function RenderMarkdown({ text }) {
    if (!text) return null;
    const lines = text.split("\n");
    const elements = [];
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];

        // Tables
        if (line.trim().startsWith("|")) {
            const tableRows = [];
            let headerRow = null;

            // Look for table block
            while (i < lines.length && lines[i].trim().startsWith("|")) {
                const row = lines[i].trim().split("|").filter((_, idx, arr) => idx > 0 && idx < arr.length - 1).map(c => c.trim());

                if (lines[i].includes("---")) {
                    // Skip separator
                } else if (!headerRow) {
                    headerRow = row;
                } else {
                    tableRows.push(row);
                }
                i++;
            }

            if (headerRow) {
                elements.push(
                    <div key={`table-${i}`} style={{ overflowX: "auto", margin: "1.2rem 0", background: "rgba(255,255,255,0.02)", borderRadius: "8px", border: "1px solid rgba(129,140,248,0.2)" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem", textAlign: "left" }}>
                            <thead>
                                <tr style={{ background: "rgba(129,140,248,0.08)" }}>
                                    {headerRow.map((cell, idx) => (
                                        <th key={idx} style={{ padding: "10px 12px", borderBottom: "1px solid rgba(129,140,248,0.2)", color: "#a5b4fc", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", fontSize: "0.7rem" }}>{parseBold(cell)}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {tableRows.map((row, rowIdx) => (
                                    <tr key={rowIdx} style={{ borderBottom: rowIdx === tableRows.length - 1 ? "none" : "1px solid rgba(255,255,255,0.03)", background: rowIdx % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)" }}>
                                        {row.map((cell, cellIdx) => (
                                            <td key={cellIdx} style={{ padding: "10px 12px", color: cellIdx === 0 ? "#fff" : "#c4c8f0" }}>{parseBold(cell)}</td>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                );
            }
            continue;
        }

        // Headings
        if (line.startsWith("### ")) {
            elements.push(<h4 key={i} style={{ color: "#a5b4fc", margin: "1.2rem 0 0.4rem", fontSize: "0.9rem", fontWeight: 700, borderBottom: "1px solid rgba(129,140,248,0.15)", paddingBottom: "0.3rem" }}>{parseBold(line.slice(4))}</h4>);
            i++; continue;
        }
        if (line.startsWith("## ")) {
            elements.push(<h3 key={i} style={{ color: "#a5b4fc", margin: "1.2rem 0 0.5rem", fontSize: "1rem", fontWeight: 700 }}>{parseBold(line.slice(3))}</h3>);
            i++; continue;
        }

        // Bullet points
        if (line.trimStart().startsWith("- ") || line.trimStart().startsWith("* ")) {
            const bullet = line.trimStart().slice(2);
            elements.push(
                <div key={i} style={{ display: "flex", gap: "8px", marginBottom: "4px", paddingLeft: "0.5rem" }}>
                    <span style={{ color: "#a5b4fc", flexShrink: 0 }}>•</span>
                    <span style={{ fontSize: "0.84rem", color: "#c4c8f0", lineHeight: 1.6 }}>{parseBold(bullet)}</span>
                </div>
            );
            i++; continue;
        }

        // Empty lines
        if (line.trim() === "") {
            elements.push(<div key={i} style={{ height: "0.4rem" }} />);
            i++; continue;
        }

        // Regular paragraph
        elements.push(<p key={i} style={{ fontSize: "0.86rem", color: "#c4c8f0", lineHeight: 1.7, margin: "0 0 0.5rem" }}>{parseBold(line)}</p>);
        i++;
    }

    return <>{elements}</>;
}

function parseBold(text) {
    if (!text) return text;
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((p, i) => {
        if (p.startsWith("**") && p.endsWith("**")) {
            return <strong key={i} style={{ color: "#e0e2ff", fontWeight: 600 }}>{p.slice(2, -2)}</strong>;
        }
        return p;
    });
}


export default function MarketGPTView() {
    const [chatQuery, setChatQuery] = useState("");
    const [chatHistory, setChatHistory] = useState([]);
    const [thinking, setThinking] = useState(false);
    const [thinkingPhase, setThinkingPhase] = useState("Analysing context...");
    const [snapshot, setSnapshot] = useState(null);
    const chatRef = useRef(null);

    useEffect(() => {
        fetch("/api/backtest/latest")
            .then(res => res.ok ? res.json() : null)
            .then(data => setSnapshot(data))
            .catch(() => { });
    }, []);

    useEffect(() => {
        if (chatRef.current) {
            chatRef.current.scrollTop = chatRef.current.scrollHeight;
        }
    }, [chatHistory, thinking, thinkingPhase]);

    const sendChat = async () => {
        if (!chatQuery || thinking) return;

        const query = chatQuery;
        const userMsg = { role: "user", text: query };
        setChatHistory(prev => [...prev, userMsg]);
        setChatQuery("");
        setThinking(true);

        const phases = [
            "Analysing context...",
            "Cross-referencing backtest data...",
            "Synthesising research insights...",
        ];

        let phaseIdx = 0;
        setThinkingPhase(phases[0]);
        const interval = setInterval(() => {
            phaseIdx++;
            if (phaseIdx < phases.length) {
                setThinkingPhase(phases[phaseIdx]);
            }
        }, 1200);

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query }),
            });
            const data = await res.json();
            clearInterval(interval);
            setChatHistory(prev => [...prev, { role: "assistant", text: data.response }]);
        } catch (err) {
            clearInterval(interval);
            setChatHistory(prev => [...prev, { role: "assistant", text: "Error connecting to chat." }]);
        } finally {
            setThinking(false);
            setThinkingPhase("Analysing context...");
        }
    };

    return (
        <div className="market-gpt-container" style={{ maxWidth: '900px', margin: '0 auto', paddingTop: '1.5rem' }}>
            <div className="chat-card active" style={{ height: '700px', display: 'flex', flexDirection: 'column' }}>
                <div className="chat-header">
                    <div className="chat-dot"></div>
                    <strong>Market GPT</strong>
                    <span className="tiny muted" style={{ marginLeft: 'auto' }}>Grounded in Latest Backtest</span>
                </div>

                <div className="chat-thread" ref={chatRef} style={{ flex: 1, overflowY: 'auto' }}>
                    {chatHistory.length === 0 && (
                        <div className="empty-state">
                            <div className="pulse-icon">💬</div>
                            <h2 style={{ color: '#fff' }}>Market Intelligence Assistant</h2>
                            <p className="muted">Ask me about backtest alpha, sentiment grounding, or model comparisons.</p>

                        </div>
                    )}

                    {chatHistory.map((m, i) => (
                        m.role === "user" ? (
                            <div key={i} className="chat-bubble user">
                                {m.text}
                            </div>
                        ) : (
                            <div key={i} className="gpt-response-card">
                                <RenderMarkdown text={m.text} />
                            </div>
                        )
                    ))}

                    {thinking && (
                        <div className="gpt-thinking-indicator">
                            <div className="thinking-dots">
                                <span /><span /><span />
                            </div>
                            <span style={{ fontSize: "0.8rem", color: "#818cf8" }}>{thinkingPhase}</span>
                        </div>
                    )}
                </div>

                <div className="chat-input-area">
                    <div className="chat-input-group">
                        <input
                            type="text"
                            value={chatQuery}
                            onChange={(e) => setChatQuery(e.target.value)}
                            placeholder="Query system insights..."
                            onKeyDown={(e) => e.key === "Enter" && sendChat()}
                        />
                        <button className="chat-send-btn" onClick={sendChat} disabled={thinking}>
                            <svg width="16" height="16" viewBox="0 0 24 23" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path d="M22 2L11 13M22 2L15 22L11 13M11 13L2 9L22 2" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
