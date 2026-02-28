import { useState, useEffect, useRef } from "react";

export default function MarketGPTView() {
    const [chatQuery, setChatQuery] = useState("");
    const [chatHistory, setChatHistory] = useState([]);
    const [thinking, setThinking] = useState(false);
    const [thinkingPhase, setThinkingPhase] = useState("Thinking...");
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

        const keywords = ["btc", "bitcoin", "alpha", "sharpe", "sentiment", "strategy", "rsi", "momentum"];
        const found = keywords.find(k => query.toLowerCase().includes(k)) || "market";

        const phases = [
            "Thinking...",
            "Analyzing market trends...",
            "Scouring backtest data...",
            `Formulating answer for ${found.toUpperCase()}...`
        ];

        let phaseIdx = 0;
        const interval = setInterval(() => {
            phaseIdx++;
            if (phaseIdx < phases.length) {
                setThinkingPhase(phases[phaseIdx]);
            }
        }, 800);

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
            setThinkingPhase("Thinking...");
        }
    };

    return (
        <div className="market-gpt-container" style={{ maxWidth: '900px', margin: '0 auto', paddingTop: '1.5rem' }}>
            <div className="chat-card active" style={{ height: '650px' }}>
                <div className="chat-header">
                    <div className="chat-dot"></div>
                    <strong>Market GPT</strong>
                    <span className="tiny muted" style={{ marginLeft: 'auto' }}>Grounded in Latest Backtest</span>
                </div>

                <div className="chat-thread" ref={chatRef}>
                    {chatHistory.length === 0 && (
                        <div className="empty-state">
                            <div className="pulse-icon">💬</div>
                            <h2 style={{ color: '#fff' }}>Market Intelligence Assistant</h2>
                            <p className="muted">Ask me about backtest alpha, sentiment grounding, or January 2025 trends.</p>
                            {snapshot && (
                                <div className="card mini-card" style={{ marginTop: '2rem', textAlign: 'left', background: 'rgba(99, 102, 241, 0.05)' }}>
                                    <p className="tiny muted">System Context Loaded:</p>
                                    <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem' }}>
                                        <span><strong>Strategy:</strong> {snapshot.strategy}</span>
                                        <span><strong>Model:</strong> {snapshot.model}</span>
                                        <span className="text-success"><strong>Alpha:</strong> +{(snapshot.metrics?.gated?.total_return - snapshot.metrics?.baseline?.total_return).toFixed(2)}%</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                    {chatHistory.map((m, i) => (
                        <div key={i} className={`chat-bubble ${m.role}`}>
                            {m.text}
                        </div>
                    ))}
                    {thinking && (
                        <div className="chat-bubble assistant thinking">
                            <span></span> {thinkingPhase}
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
