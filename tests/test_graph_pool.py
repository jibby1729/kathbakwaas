from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

from app.graph_pool import capture_pool


def test_recreate_pool_after_cache_clear(monkeypatch):
    allocate = Mock(side_effect=['first', 'replacement'])
    monkeypatch.setattr('app.graph_pool.torch.cuda.graph_pool_handle', allocate)
    owner = SimpleNamespace(pool='retired', graphs={})
    assert capture_pool(owner) == 'first'
    owner.graphs[20] = object()
    assert capture_pool(owner) == 'first'
    owner.graphs.clear()
    assert capture_pool(owner) == 'replacement'
    assert allocate.call_count == 2


def test_lru_preserves_live_pool_but_replaces_after_last_eviction(monkeypatch):
    allocate = Mock(return_value='new')
    monkeypatch.setattr('app.graph_pool.torch.cuda.graph_pool_handle', allocate)
    owner = SimpleNamespace(pool='live', graphs=OrderedDict([(1, object()), (2, object())]))
    owner.graphs.popitem(last=False)
    assert capture_pool(owner) == 'live'
    allocate.assert_not_called()
    owner.graphs.popitem(last=False)
    assert capture_pool(owner) == 'new'
