"""Guard for failure class I (see docs/failure-sweeps.md): pyproject.toml
used to pin the PyTorch CPU wheel index GLOBALLY via a bare
`[[tool.uv.index]] url = ".../whl/cpu"`, which `uv sync --group gpu` on the
Linux GPU box honoured just as much as local Windows dev -- installing
`torch==X+cpu` on a CUDA-capable box with no error, just silent ~20x
slower encoding (`torch.cuda.is_available()` returns False).

The index must be `explicit = true` (never picked implicitly) and only
consulted for torch/torchvision via a `tool.uv.sources` entry carrying a
`sys_platform == 'win32'` marker, so `uv sync` on Linux resolves torch from
PyPI instead -- whose linux x86_64 wheels are CUDA-enabled.
"""
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

pytestmark = pytest.mark.smoke


def _load() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _pytorch_cpu_index_names(data: dict) -> list[str]:
    indexes = data.get("tool", {}).get("uv", {}).get("index", [])
    return [
        idx["name"]
        for idx in indexes
        if "download.pytorch.org/whl/cpu" in idx.get("url", "")
    ]


def test_pytorch_cpu_index_entries_are_explicit():
    data = _load()
    indexes = data.get("tool", {}).get("uv", {}).get("index", [])
    cpu_indexes = [idx for idx in indexes if "download.pytorch.org/whl/cpu" in idx.get("url", "")]
    assert cpu_indexes, "expected at least one pytorch CPU wheel index entry in [[tool.uv.index]]"
    for idx in cpu_indexes:
        assert idx.get("explicit") is True, (
            f"pytorch CPU index entry {idx!r} must set explicit = true -- otherwise "
            f"`uv sync --group gpu` on the Linux GPU box may silently resolve "
            f"torch/torchvision from it too (class I, see docs/failure-sweeps.md)"
        )


def test_torch_and_torchvision_sources_scoped_to_windows():
    data = _load()
    cpu_index_names = set(_pytorch_cpu_index_names(data))
    assert cpu_index_names, "expected a named pytorch CPU index to cross-check tool.uv.sources against"

    sources = data.get("tool", {}).get("uv", {}).get("sources", {})
    for pkg in ("torch", "torchvision"):
        assert pkg in sources, f"tool.uv.sources.{pkg} must exist to scope the CPU wheel index to Windows"
        entries = sources[pkg]
        assert isinstance(entries, list) and entries, f"tool.uv.sources.{pkg} must be a non-empty list"
        cpu_entries = [e for e in entries if e.get("index") in cpu_index_names]
        assert cpu_entries, f"tool.uv.sources.{pkg} must reference the pytorch CPU index"
        for entry in cpu_entries:
            marker = entry.get("marker", "")
            assert "sys_platform" in marker and "win32" in marker, (
                f"tool.uv.sources.{pkg} entry {entry!r} referencing the pytorch CPU index "
                f"must carry a sys_platform == 'win32' marker -- otherwise the GPU box "
                f"(Linux) resolves torch from the CPU index too (class I regression)"
            )
