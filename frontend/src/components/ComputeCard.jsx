import React from 'react';

// Per-stage compute-device badges. Built from JobResult.compute_summary,
// which the pipeline writes at the end of analysis after introspecting
// the engine + perceiver. Lets the user tell at a glance whether the
// GPU was actually used — instead of having to export logs and grep
// "YOLO-World device" / "Frame extraction succeeded (...)" lines.
//
// Schema (defined in backend/services/pipeline.py:_build_compute_summary):
//   compute_summary = {
//     frame_extract: {device: "cuda:0"|"cpu", detail: "ffmpeg strategy: GPU+scene"},
//     yolo_world:    {device: "cuda:0"|"cpu", detail: "YOLO-World v2 ..."},
//     whisper:       {device: "cuda:0"|"cpu", detail: "CTranslate2 float16"},
//   }
//
// Stages not in the dict are simply omitted — those stages didn't run.

const STAGE_LABELS = {
  frame_extract: 'Frame Extract',
  yolo_world: 'YOLO-World',
  whisper: 'Whisper',
};

// Display order. Stages missing from the summary are skipped.
const STAGE_ORDER = ['frame_extract', 'yolo_world', 'whisper'];

function isGpu(device) {
  return typeof device === 'string' && device.toLowerCase().startsWith('cuda');
}

function StageRow({ stage, info }) {
  const gpu = isGpu(info?.device);
  const label = STAGE_LABELS[stage] || stage;
  return (
    <div
      title={info?.detail || ''}
      style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        gap: 12, padding: '6px 0',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <span style={{ fontSize: 12, color: 'var(--text-primary)' }}>{label}</span>
        {info?.detail && (
          <span style={{
            fontSize: 10, color: 'var(--text-muted)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {info.detail}
          </span>
        )}
      </div>
      <span style={{
        padding: '2px 10px', borderRadius: 12,
        fontSize: 10, fontWeight: 700, fontFamily: 'var(--font-mono)',
        background: gpu ? 'rgba(34,197,94,0.15)' : 'rgba(245,158,11,0.15)',
        color: gpu ? 'var(--success, #22c55e)' : 'var(--accent-amber, #f59e0b)',
        border: `1px solid ${gpu ? 'rgba(34,197,94,0.35)' : 'rgba(245,158,11,0.35)'}`,
        whiteSpace: 'nowrap',
      }}>
        {gpu ? `GPU (${info.device})` : 'CPU'}
      </span>
    </div>
  );
}

export default function ComputeCard({ summary }) {
  if (!summary || typeof summary !== 'object') return null;
  const stages = STAGE_ORDER.filter((k) => summary[k]);
  if (stages.length === 0) return null;

  const anyCpu = stages.some((k) => !isGpu(summary[k]?.device));

  return (
    <div style={{
      marginTop: 16, padding: '14px 16px',
      background: 'var(--bg-panel)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)',
    }}>
      <div style={{
        display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
        marginBottom: 8,
      }}>
        <h3 style={{
          fontSize: 13, margin: 0, color: 'var(--text-primary)', fontWeight: 600,
        }}>
          Compute
        </h3>
        {anyCpu && (
          <span style={{ fontSize: 10, color: 'var(--accent-amber, #f59e0b)' }}>
            At least one stage ran on CPU
          </span>
        )}
      </div>
      {stages.map((stage) => (
        <StageRow key={stage} stage={stage} info={summary[stage]} />
      ))}
    </div>
  );
}
