# ClipAI developer targets

.PHONY: parity parity-tests

# ── Pixel-level preview↔export parity check ────────────────────────────
# Compares a client (browser/WebCodecs) export against a server (FFmpeg)
# export of the SAME clip. Run on a machine with FFmpeg (e.g. the Unraid
# host):
#
#   1. In the editor, export the clip once with "Browser export" and
#      once with "Server export" (same quality + aspect + settings).
#   2. make parity CLIENT=client_export.mp4 SERVER=server_export.mp4
#
# Optional: TIMESTAMPS="1.0 2.5 5.0" (defaults below), SSIM=0.90.
# Exits non-zero when any sampled frame falls below the SSIM threshold
# or the subtitle bounding boxes drift. Known acceptable divergence:
# libass vs canvas font metrics (a few px), xfade easing vs canvas
# cubic-bezier mid-transition, varispeed vs atempo pitch (audio only,
# and only when preservePitch differs).
TIMESTAMPS ?= 1.0 2.5 5.0
SSIM ?= 0.90

parity:
	@test -n "$(CLIENT)" -a -n "$(SERVER)" || \
		(echo "usage: make parity CLIENT=client.mp4 SERVER=server.mp4 [TIMESTAMPS=\"1.0 2.5 5.0\"] [SSIM=0.90]" && exit 2)
	python scripts/parity_harness.py "$(CLIENT)" "$(SERVER)" \
		--timestamps $(TIMESTAMPS) --ssim-threshold $(SSIM)

# Feature-level parity gate: the checklist tests + export pipeline tests
parity-tests:
	cd frontend && npx vitest run
	python -m pytest backend/tests/test_export_pipeline.py \
		backend/tests/test_ffmpeg_filter_builder.py \
		backend/tests/test_ass_golden.py \
		backend/tests/test_export_formats.py -q
	node scripts/generate-parity-checklist.mjs
	@git diff --quiet docs/parity-checklist.md || \
		(echo "docs/parity-checklist.md is stale — commit the regenerated file" && exit 1)
