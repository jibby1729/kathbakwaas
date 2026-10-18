from app.prefix_cache import PromptCache


def test_only_a_complete_identical_prompt_is_reused():
    cache = PromptCache()
    original = {'logits': [3], 'four_beam_cache': [[1], [1], [1], [1]]}
    cache.put('page', [1, 2, 3], original)
    assert cache.fork('page', [1, 2]) is None
    assert cache.fork('page', [1, 2, 3, 4]) is None
    assert cache.fork('page', [1, 20, 3]) is None
    assert cache.fork('page', [9, 2, 3]) is None
    assert cache.fork('page', [1, 2, 3]) == original


def test_generation_cannot_mutate_the_saved_prompt():
    cache = PromptCache()
    original = {'cache': [1, 2, 3]}
    cache.put('page', [1, 2, 3], original)
    copied = cache.fork('page', [1, 2, 3])
    copied['cache'].append(4)
    assert original['cache'] == [1, 2, 3]
    assert cache.fork('page', [1, 2, 3]) == original


def test_page_sessions_are_isolated_and_bounded():
    cache = PromptCache(capacity=2)
    cache.put('first', [1, 2], {'cache': 1})
    cache.put('second', [1, 2], {'cache': 2})
    assert cache.fork('unknown', [1, 2]) is None
    cache.put('third', [1, 2], {'cache': 3})
    assert cache.get('first') is None
    assert len(cache.entries) == 2
    assert cache.fork('second', [1, 2]) == {'cache': 2}


def test_expired_prompts_are_discarded():
    cache = PromptCache(ttl=-1)
    cache.put('page', [1, 2], {'cache': 1})
    assert cache.get('page') is None
