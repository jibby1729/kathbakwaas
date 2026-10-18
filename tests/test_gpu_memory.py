"""Admission and recovery behavior, independent of CUDA availability."""
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from app.gpu_memory import GPUMemoryBusy, ensure_headroom


def setup(monkeypatch, free_values):
    cuda = SimpleNamespace(
        mem_get_info=Mock(side_effect=[(n * 2**20, 8 * 2**30) for n in free_values]),
        synchronize=Mock(), empty_cache=Mock())
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    graph = lambda: SimpleNamespace(graphs={1: object()})
    engine = SimpleNamespace(prompt_cache=SimpleNamespace(entries={'page': object()}),
                             fast_execution={k: graph() for k in ['decoder', 'reorder', 'prefill']})
    engine.fast_execution['beam_helpers'] = [graph(), graph()]
    return engine, cuda


def test_enough_headroom_keeps_acceleration(monkeypatch):
    engine, cuda = setup(monkeypatch, [2000])
    ensure_headroom(engine)
    assert engine.fast_execution['decoder'].graphs
    cuda.synchronize.assert_not_called()


def test_pressure_releases_caches_before_new_work(monkeypatch):
    engine, cuda = setup(monkeypatch, [700, 2200])
    ensure_headroom(engine)
    assert not engine.prompt_cache.entries
    assert not engine.fast_execution['prefill'].graphs
    assert engine.fast_execution['decoder'].graphs
    assert all(not g.graphs for g in engine.fast_execution['beam_helpers'])
    cuda.empty_cache.assert_called_once()


def test_stronger_pressure_also_releases_decoder_graphs(monkeypatch):
    engine, cuda = setup(monkeypatch, [700, 1100, 2200])
    ensure_headroom(engine)
    assert not engine.fast_execution['decoder'].graphs
    assert not engine.fast_execution['reorder'].graphs
    assert cuda.empty_cache.call_count == 2


def test_insufficient_memory_defers_inference(monkeypatch):
    engine, cuda = setup(monkeypatch, [600, 900, 900])
    with pytest.raises(GPUMemoryBusy):
        ensure_headroom(engine)
    assert cuda.empty_cache.call_count == 2
