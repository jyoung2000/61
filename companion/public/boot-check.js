// Surfaces frontend / webview failures ON-SCREEN instead of a blank window.
// Loaded as a separate same-origin script (not inline — the CSP forbids
// inline scripts) so it still runs even when the main bundle fails to load
// or execute. Errors render as a dismissable OVERLAY on top of the app —
// a measured failure (a denied openUrl promise) used to REPLACE the whole
// UI with the error page and leave no way back, turning a harmless
// permission bug into a bricked-looking window. "Back to the app" removes
// the overlay (the app is still mounted underneath); "Restart UI" reloads.
(function () {
  function render(title, detail) {
    var old = document.getElementById('cc-err-overlay');
    if (old) old.remove();
    var root = document.getElementById('root');
    var appAlive = !!(root && root.childElementCount > 0);
    var el = document.createElement('div');
    el.id = 'cc-err-overlay';
    el.setAttribute('style',
      'position:fixed;inset:0;overflow:auto;z-index:2147483647;' +
      'background:#fff;color:#c0392b;');
    el.innerHTML =
      '<div style="font:13px/1.6 system-ui,-apple-system,sans-serif;padding:20px;max-width:720px;margin:0 auto">' +
      '<h2 style="margin:0 0 10px;font-size:16px;color:#c0392b">' + title + '</h2>' +
      '<pre style="white-space:pre-wrap;word-break:break-word;background:#fbeeec;border:1px solid #e6b0aa;border-radius:6px;padding:12px;color:#7b241c;font-size:12px">' +
      (detail || '(no details captured)') + '</pre>' +
      '<div style="margin:14px 0">' +
      (appAlive
        ? '<button id="cc-err-back" style="font:13px system-ui;padding:7px 16px;border-radius:6px;border:1px solid #bbb;background:#f6f6f6;color:#222;cursor:pointer;margin-right:8px">← Back to the app</button>'
        : '') +
      '<button id="cc-err-reload" style="font:13px system-ui;padding:7px 16px;border-radius:6px;border:1px solid #bbb;background:#f6f6f6;color:#222;cursor:pointer">Restart UI</button>' +
      '</div>' +
      '<p style="color:#555;margin:12px 0 4px">If this keeps happening, copy the text above, and the full log file, back into the chat:</p>' +
      '<code style="color:#333;background:#eee;padding:2px 6px;border-radius:4px">%LOCALAPPDATA%\\app.clipai.companion\\companion.log</code>' +
      '<p style="color:#888;font-size:11px;margin-top:6px">(macOS: ~/Library/Application Support/app.clipai.companion/companion.log)</p>' +
      '</div>';
    document.body.appendChild(el);
    var back = document.getElementById('cc-err-back');
    if (back) back.onclick = function () { el.remove(); };
    var reload = document.getElementById('cc-err-reload');
    if (reload) reload.onclick = function () { location.reload(); };
  }

  window.addEventListener('error', function (e) {
    var d = (e && e.message ? e.message : '') + '\n' +
      (e && e.error && e.error.stack ? e.error.stack : '') +
      (e && e.filename ? '\n(' + e.filename + ':' + e.lineno + ')' : '');
    render('Companion UI error', d);
  });

  window.addEventListener('unhandledrejection', function (e) {
    var r = e && e.reason;
    render('Companion UI promise rejection', (r && r.stack) ? r.stack : String(r));
  });

  // If React never mounts, say so (and name the usual culprit) rather than
  // leaving a blank white window.
  window.setTimeout(function () {
    var el = document.getElementById('root');
    if (el && el.childElementCount === 0 && !(el.textContent || '').trim()) {
      render('Companion UI did not load',
        'The window opened but the interface did not render within 5 seconds.\n\n' +
        'Most common cause on Windows: the Microsoft Edge WebView2 Runtime is ' +
        'missing. Install the free "Evergreen" runtime, then relaunch:\n' +
        '  https://developer.microsoft.com/microsoft-edge/webview2/\n\n' +
        'If WebView2 is already installed, the log file has the details.');
    }
  }, 5000);
})();
