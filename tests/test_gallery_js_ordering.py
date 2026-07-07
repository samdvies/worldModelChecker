"""Guard for failure class F: docs/gallery/index.html rendered fine and
threw zero errors in static validation, but an uncaught TypeError fired
intermittently at page load because loadImages() could synchronously invoke
drawFrame -> updatePlayhead BEFORE drawTraceChart() (which populates
playheadRegistry and the playhead DOM node) had run. Neither `node --check`
nor a plain text read catches this class of ordering/hoisting bug -- only
live execution did (see docs/failure-sweeps.md). This test encodes the
specific pattern narrowly (to avoid false positives) and, since node is
available in this environment, also runs `node --check` as a cheap syntax
floor.
"""
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GALLERY_HTML = REPO_ROOT / "docs" / "gallery" / "index.html"


def _extract_script(text: str) -> str:
    m = re.search(r"<script>(.*)</script>", text, re.DOTALL)
    assert m, "expected exactly one <script>...</script> block in gallery HTML"
    return m.group(1)


def test_node_check_passes_if_node_available():
    node = shutil.which("node")
    if node is None:
        import pytest

        pytest.skip("node not installed in this environment -- static pattern check below still runs")
    script = _extract_script(GALLERY_HTML.read_text(encoding="utf-8"))
    result = subprocess.run(
        [node, "--check", "/dev/stdin"] if not node.lower().endswith(".exe") else [node, "-e", f"new Function({script!r})"],
        input=script,
        text=True,
        capture_output=True,
    )
    # node --check via stdin isn't supported on all builds; the -e Function
    # constructor form only validates syntax (not the ordering bug below),
    # but is a legitimate cheap floor. Fall back tolerantly.
    if result.returncode != 0 and "/dev/stdin" in str(result.args):
        result = subprocess.run([node, "-e", f"new Function({script!r})"], capture_output=True, text=True)
    assert result.returncode == 0, f"node syntax check failed:\n{result.stderr}"


def test_draw_trace_chart_runs_before_load_images_in_init_player():
    """The specific class-F fix: inside initPlayer(), drawTraceChart(...)
    (which populates playheadRegistry[lawName] + the playhead DOM element)
    must textually precede the loadImages() call (whose Image.onload
    handlers can fire synchronously and call updatePlayhead, which reads
    that registry)."""
    script = _extract_script(GALLERY_HTML.read_text(encoding="utf-8"))

    init_player_match = re.search(
        r"function initPlayer\(.*?\n  \}\n", script, re.DOTALL
    )
    assert init_player_match, "expected to find function initPlayer(...) { ... } block"
    body = init_player_match.group(0)

    load_images_call = re.search(r"^\s*loadImages\(\);\s*$", body, re.MULTILINE)
    draw_trace_chart_call = re.search(r"^\s*drawTraceChart\(", body, re.MULTILINE)
    assert load_images_call, "expected a bare loadImages(); call in initPlayer()"
    assert draw_trace_chart_call, "expected a drawTraceChart(...) call in initPlayer()"

    assert draw_trace_chart_call.start() < load_images_call.start(), (
        "drawTraceChart(...) must be called before loadImages() in "
        "initPlayer() -- loadImages()'s Image.onload can fire "
        "synchronously and call updatePlayhead() before playheadRegistry "
        "is populated otherwise (class F, see docs/failure-sweeps.md)"
    )


def test_update_playhead_has_a_defensive_guard():
    """Belt-and-braces: even with correct ordering, updatePlayhead() must
    guard against a missing registry/DOM entry rather than throw -- this is
    what kept the observed bug non-fatal (page recovered) instead of
    breaking the whole tab."""
    script = _extract_script(GALLERY_HTML.read_text(encoding="utf-8"))
    m = re.search(r"function updatePlayhead\(.*?\n  \}\n", script, re.DOTALL)
    assert m, "expected to find function updatePlayhead(...) { ... }"
    assert re.search(r"if\s*\(!reg \|\| !playhead\)\s*return;", m.group(0)), (
        "updatePlayhead() must guard on a missing registry entry or DOM "
        "node and return early instead of throwing"
    )
