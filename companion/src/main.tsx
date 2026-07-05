import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles.css';

try {
  ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
} catch (err) {
  // Make a render-time failure visible instead of leaving a blank window;
  // boot-check.js also has a global handler, this covers the sync throw.
  const el = document.getElementById('root');
  if (el) {
    el.textContent =
      'Failed to start the Companion UI: ' +
      (err instanceof Error ? err.stack || err.message : String(err));
  }
  throw err;
}
