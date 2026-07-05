# Sidecar staging directory

CI (or a local build) drops the platform whisper sidecar here before
`tauri build`:

* Windows: `whisper-server.exe` + its DLLs (PyInstaller output from
  `companion/sidecars/whisper-server`)
* macOS: `whisper-server` (whisper.cpp, Metal) + `ggml-metal.metal`

Everything in this directory is bundled into the installer as the
`sidecar/` resource. Only this README is committed — binaries never
enter git.
