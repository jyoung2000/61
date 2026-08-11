import React, { useEffect, useState } from 'react';

// Detects that the SERVER is serving a newer UI bundle than the one running
// in this tab, and offers a one-click reload. This is the belt to the
// no-store-index suspenders: a tab left open across a container update (or a
// browser that cached the old shell before the header fix) otherwise keeps
// running the old app indefinitely — observed as "the new settings never
// appeared" while the container demonstrably ran the new code.
//
// Detection is bundle-hash comparison: fetch a fresh index.html (no-store),
// read the hashed /assets/index-*.js it references, and compare with the
// script tag this tab actually loaded. No clocks, no versions to maintain.
const CHECK_EVERY_MS = 5 * 60 * 1000;

const bundleOf = (html) => (html.match(/\/assets\/index-[^"']+\.js/) || [null])[0];

export default function UpdateNudge() {
  const [stale, setStale] = useState(false);

  useEffect(() => {
    const current = [...document.scripts]
      .map((s) => s.getAttribute('src') || '')
      .find((src) => /\/assets\/index-[^"']+\.js/.test(src));
    if (!current) return undefined; // dev server — nothing to compare

    let stopped = false;
    const check = async () => {
      try {
        const r = await fetch('/', { cache: 'no-store' });
        if (!r.ok) return;
        const served = bundleOf(await r.text());
        if (!stopped && served && !served.endsWith(current) && !current.endsWith(served)) {
          setStale(true);
        }
      } catch { /* offline — try again later */ }
    };
    const iv = setInterval(check, CHECK_EVERY_MS);
    window.addEventListener('focus', check);
    check();
    return () => { stopped = true; clearInterval(iv); window.removeEventListener('focus', check); };
  }, []);

  if (!stale) return null;
  return (
    <div role="status" style={{
      position: 'fixed', top: 0, left: 0, right: 0, zIndex: 3000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12,
      padding: '8px 16px', background: 'var(--accent-cyan)', color: 'var(--bg-base)',
      fontSize: 13, fontWeight: 600, boxShadow: '0 2px 10px rgba(0,0,0,0.25)',
    }}>
      <span>ClipAI was updated on the server — reload to get the new version.</span>
      <button
        onClick={() => window.location.reload()}
        style={{
          padding: '4px 14px', borderRadius: 6, border: 'none', cursor: 'pointer',
          background: 'var(--bg-base)', color: 'var(--text-primary)', fontSize: 12, fontWeight: 700,
        }}
      >
        Reload now
      </button>
    </div>
  );
}
