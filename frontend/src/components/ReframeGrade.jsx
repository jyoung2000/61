import React, { useState } from 'react';

// Letter-grade colours (mirrors the reference reframer's grade palette).
const GRADE_COLORS = { A: '#22c55e', B: '#84cc16', C: '#eab308', D: '#f97316', F: '#ef4444' };
const SEVERITY_COLORS = { HIGH: '#ef4444', MED: '#f59e0b', LOW: 'var(--text-muted)' };

function qualityColor(q) {
  if (q >= 75) return '#22c55e';
  if (q >= 50) return '#eab308';
  if (q >= 30) return '#f97316';
  return '#ef4444';
}

function fmtTime(sec) {
  const s = Math.max(0, Math.round(sec || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

// One sub-category: label, raw value, and a quality bar (fuller = better).
function Metric({ label, value, unit, goodHigh = true, hint }) {
  const v = Number.isFinite(value) ? value : 0;
  const quality = goodHigh ? v : Math.max(0, 100 - v);
  const color = qualityColor(quality);
  return (
    <div style={{ flex: '1 1 200px', minWidth: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4, gap: 8 }}>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ fontSize: 13, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>
          {Math.round(v)}{unit}
        </span>
      </div>
      <div style={{ height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${Math.max(0, Math.min(100, quality))}%`,
          background: color, transition: 'width 0.4s ease',
        }} />
      </div>
      {hint && <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 3 }}>{hint}</div>}
    </div>
  );
}

/**
 * Reframe quality grade card — A-F grade, 0-100 overall score and the
 * per-axis sub-categories produced by the backend reframe evaluator.
 * Responsive: the metric rows wrap from 3 → 2 → 1 column as width shrinks.
 */
export default function ReframeGrade({ report, isMobile = false }) {
  const [showProblems, setShowProblems] = useState(false);
  if (!report || typeof report !== 'object') return null;

  const grade = String(report.grade || 'F');
  const gradeColor = GRADE_COLORS[grade] || '#888';
  const overall = Math.round(report.overall_score || 0);
  const watchability = Math.round(report.watchability_score || 0);
  const problems = Array.isArray(report.problems) ? report.problems : [];

  return (
    <div className="slide-in" style={{
      background: 'var(--bg-panel)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)', padding: isMobile ? 16 : 20,
      marginBottom: 16, boxShadow: 'var(--shadow-sm)',
    }}>
      {/* Header — grade badge + overall score */}
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 16, marginBottom: 18 }}>
        <div style={{
          width: isMobile ? 64 : 78, height: isMobile ? 64 : 78, flexShrink: 0,
          borderRadius: 'var(--radius-md)', background: `${gradeColor}22`,
          border: `2px solid ${gradeColor}`, display: 'flex',
          alignItems: 'center', justifyContent: 'center',
        }}>
          <span style={{ fontSize: isMobile ? 34 : 44, fontWeight: 800, lineHeight: 1, color: gradeColor }}>
            {grade}
          </span>
        </div>
        <div style={{ flex: '1 1 200px', minWidth: 0 }}>
          <h3 style={{ fontSize: 14, margin: '0 0 6px', color: 'var(--accent-cyan)' }}>Reframe Quality</h3>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginBottom: 6 }}>
            <span style={{ fontSize: 24, fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
              {overall}
            </span>
            <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>/ 100 overall</span>
          </div>
          <div style={{ height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{
              height: '100%', width: `${Math.max(0, Math.min(100, overall))}%`,
              background: gradeColor, transition: 'width 0.4s ease',
            }} />
          </div>
        </div>
      </div>

      {/* Core sub-categories */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 16 }}>
        <Metric label="Face coverage" value={report.face_coverage_pct} unit="%" />
        <Metric label="Saliency focus" value={report.saliency_accuracy_pct} unit="%" />
        <Metric label="Centering" value={report.centering_pct} unit="%" />
        <Metric label="Stability" value={report.stability_score} unit="/100" />
        <Metric label="Cut coherence" value={report.cut_coherence_score} unit="/100" />
        <Metric label="Edge violations" value={report.edge_violation_pct} unit="%" goodHigh={false} />
      </div>

      {/* Watchability group */}
      <div style={{
        background: 'var(--bg-base)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)', padding: 14,
        marginBottom: problems.length ? 14 : 0,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10, gap: 8 }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Watchability
          </span>
          <span style={{ fontSize: 15, fontWeight: 800, fontFamily: 'var(--font-mono)', color: qualityColor(watchability) }}>
            {watchability}/100
          </span>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14 }}>
          <Metric label="Hold quality" value={report.hold_quality} unit="/100" hint="Subjects held long enough" />
          <Metric label="Decisiveness" value={report.transition_decisiveness} unit="/100" hint="Moves are committed" />
          <Metric label="Motion budget" value={report.motion_budget} unit="/100" hint="Total movement reasonable" />
        </div>
      </div>

      {/* Flagged problems */}
      {problems.length > 0 && (
        <div>
          <button
            onClick={() => setShowProblems((v) => !v)}
            style={{
              width: '100%', textAlign: 'left', background: 'transparent',
              border: 'none', padding: '4px 0', cursor: 'pointer',
              fontSize: 12, color: 'var(--text-secondary)',
            }}
          >
            {showProblems ? '▼' : '▶'} {problems.length} framing issue{problems.length === 1 ? '' : 's'} flagged
          </button>
          {showProblems && (
            <div style={{ marginTop: 6, maxHeight: 220, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 }}>
              {problems.slice(0, 100).map((p, i) => (
                <div key={i} style={{
                  display: 'flex', gap: 8, alignItems: 'baseline',
                  fontSize: 11, color: 'var(--text-secondary)',
                  padding: '4px 8px', background: 'var(--bg-base)',
                  borderRadius: 'var(--radius-sm)',
                }}>
                  <span style={{
                    flexShrink: 0, fontSize: 9, fontWeight: 700,
                    color: SEVERITY_COLORS[p.severity] || 'var(--text-muted)',
                    fontFamily: 'var(--font-mono)',
                  }}>
                    {String(p.severity || '').padEnd(4)}
                  </span>
                  <span style={{ flexShrink: 0, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                    {fmtTime(p.time_sec)}
                  </span>
                  <span style={{ minWidth: 0 }}>{String(p.message || '')}</span>
                </div>
              ))}
              {problems.length > 100 && (
                <div style={{ fontSize: 10, color: 'var(--text-muted)', padding: '2px 8px' }}>
                  + {problems.length - 100} more
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
