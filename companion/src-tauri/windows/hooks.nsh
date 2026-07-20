; NSIS installer hooks for the ClipAI GPU Companion (Tauri v2 installerHooks).
;
; Why this file exists — the PURELY-REMOTE update contract:
;
; The remote "push update" flow (ClipAI container → Companion /v1/update) has
; the RUNNING app download the new installer, write a relaunch .cmd script,
; and exit. That script runs the installer with /S and then loops: check
; tasklist for the app's image name FIRST; only if nothing is running does it
; start the OLD exe path as a fallback (a broken update must never leave the
; Companion offline with nobody at the desktop).
;
; Auto-relaunching the freshly-installed exe HERE — inside the installer,
; silent installs only — is what makes that script converge on the NEW
; version even across an install-location change: the v0.2.8→v0.2.9
; transition moved the app from Program Files (perMachine, UAC prompt on
; every update) to %LOCALAPPDATA% (currentUser, no elevation ever). During
; that transition the old script's fallback path still points at Program
; Files; because this hook starts the new exe before the script's first
; tasklist check, the fallback never fires and the stale copy is never
; relaunched. On steady-state per-user updates the hook is simply the
; fastest relaunch, with the script's loop kept as the backstop.
;
; IfSilent guard: attended installs keep the classic finish-page behavior —
; without it the wizard's "Run app" option could start a second instance.

!macro NSIS_HOOK_POSTINSTALL
  IfSilent 0 +2
  Exec '"$INSTDIR\${MAINBINARYNAME}.exe"'
!macroend
