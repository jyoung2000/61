"""Tests for non-blocking browser-preview resolution.

The preview player stalled during offline analysis because the file-serving
endpoint awaited ensure_browser_preview() — re-probing (and sometimes
transcoding) the source on every Range request while the pipeline saturated the
box. The fix serves cached-or-raw bytes immediately and defers generation; these
pin the cache-resolution helper it relies on.
"""

import os
import tempfile
import time

from backend.services import browser_preview as bp


def _mk(path, mtime=None, content=b"x"):
    with open(path, "wb") as fh:
        fh.write(content)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def test_missing_source_resolves_to_itself():
    assert bp.cached_browser_preview("/no/such/file.mp4") == "/no/such/file.mp4"


def test_unresolved_when_no_preview_and_no_marker():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "video.mp4")
        _mk(src)
        assert bp.cached_browser_preview(src) is None


def test_fresh_preview_is_returned():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "video.mkv")
        _mk(src, mtime=time.time() - 100)
        prev = bp._preview_path_for(src)
        _mk(prev, mtime=time.time())  # newer than source, non-empty
        assert bp.cached_browser_preview(src) == prev


def test_stale_preview_is_not_returned():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "video.mkv")
        prev = bp._preview_path_for(src)
        _mk(prev, mtime=time.time() - 100)   # older than source
        _mk(src, mtime=time.time())          # source replaced after preview
        assert bp.cached_browser_preview(src) is None


def test_none_marker_resolves_to_raw_source():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "video.mp4")
        _mk(src, mtime=time.time() - 100)
        bp._touch_none_marker(src)           # "already browser-friendly"
        assert bp.cached_browser_preview(src) == src


def test_stale_none_marker_is_ignored():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "video.mp4")
        marker = bp._none_marker_for(src)
        _mk(marker, mtime=time.time() - 100)  # marker older than source
        _mk(src, mtime=time.time())           # source replaced
        assert bp.cached_browser_preview(src) is None


def test_preview_wins_over_marker_when_both_present():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "video.mkv")
        _mk(src, mtime=time.time() - 100)
        _mk(bp._none_marker_for(src), mtime=time.time())
        prev = bp._preview_path_for(src)
        _mk(prev, mtime=time.time())
        assert bp.cached_browser_preview(src) == prev
