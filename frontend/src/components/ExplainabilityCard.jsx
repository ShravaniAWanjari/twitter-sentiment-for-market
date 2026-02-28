import { useState } from "react";

export default function ExplainabilityCard({ result, originalText }) {
    if (!result) return null;

    const {
        highlighted_html,
        top_positive = [],
        top_negative = [],
        stability = {},
        counterfactual = {},
        method = "occlusion"
    } = result;

    return (
        <div className="explainability-card card">
            <div className="explain-header flex-between">
                <span className="muted">Explainability Method: <strong>{method}</strong></span>
                {stability.score_0_1 !== undefined && (
                    <div className="stability-badge" title={`Score: ${stability.score_0_1.toFixed(2)}`}>
                        Stability: <span className={`status-${stability.label}`}>{stability.label.toUpperCase()}</span>
                    </div>
                )}
            </div>

            <div className="highlighted-text" dangerouslySetInnerHTML={{ __html: highlighted_html }} />

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
                            <span className={`badge ${counterfactual.flipped_label.toLowerCase()}`}>
                                {counterfactual.flipped_label}
                            </span>
                            <small>Confidence: {(counterfactual.new_score * 100).toFixed(1)}%</small>
                        </div>
                    </div>
                </div>
            )}

            {stability.perturbations && stability.perturbations.length > 0 && (
                <div className="stability-details">
                    <h4>Perturbation Test</h4>
                    <table className="mini-table">
                        <thead>
                            <tr>
                                <th>Type</th>
                                <th>Score Δ</th>
                            </tr>
                        </thead>
                        <tbody>
                            {stability.perturbations.map((p, i) => (
                                <tr key={i}>
                                    <td>{p.type.replace(/_/g, " ")}</td>
                                    <td className={p.diff > 0 ? "text-success" : "text-error"}>
                                        {p.diff > 0 ? "+" : ""}{p.diff.toFixed(3)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}
