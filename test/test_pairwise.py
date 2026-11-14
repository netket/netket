# Tests of the test-suite helper `common.pairwise`.

import itertools

import pytest

from . import common


class _Callspec:
    def __init__(self, params):
        self.params = params


class _Item:
    """The parts of a pytest item read by `common.select_pairwise`."""

    def __init__(self, marks, params):
        self._marks = marks
        self.callspec = _Callspec(params)

    def iter_markers(self, name):
        assert name == "parametrize"
        return iter(self._marks)


def _items(**axes):
    """The items of the full product of `axes` ({"a, b": [(a0, b0), ...], ...})."""
    marks = [pytest.mark.parametrize(k, v).mark for k, v in axes.items()]
    items = []
    for values in itertools.product(*axes.values()):
        params = {}
        for key, value in zip(axes, values):
            names = [n.strip() for n in key.split(",")]
            params.update(zip(names, value if len(names) > 1 else [value]))
        items.append(_Item(marks, params))
    return items


def _seen(items, *names):
    return {tuple(id(it.callspec.params[n]) for n in names) for it in items}


AXES = {
    "a": [object() for _ in range(3)],
    "b": [object() for _ in range(4)],
    "c, d": [(object(), object()) for _ in range(2)],
    "e": [object() for _ in range(2)],
}


@pytest.mark.parametrize("strength", [1, 2, 3])
def test_covers_all_combinations(strength):
    items = _items(**AXES)
    selected = common.select_pairwise(items, strength=strength)
    assert len(selected) < len(items)
    full = {k: _seen(items, *[n.strip() for n in k.split(",")]) for k in AXES}
    for keys in itertools.combinations(AXES, strength):
        names = [n.strip() for k in keys for n in k.split(",")]
        expected = len(list(itertools.product(*(full[k] for k in keys))))
        assert len(_seen(selected, *names)) == expected


def test_pairwise_size_and_determinism():
    items = _items(a=AXES["a"][:2], b=AXES["b"], e=AXES["e"])
    selected = common.select_pairwise(items)
    # 4 x 2 values: no smaller set covers all the pairs of `b` and `a`
    assert len(selected) == 8
    assert selected == common.select_pairwise(items)
    # kept in collection order
    assert selected == [it for it in items if it in selected]


def test_cross():
    items = _items(**AXES)
    selected = common.select_pairwise(items, cross=("a", "b", "e"))
    assert len(_seen(selected, "a", "b", "e")) == 3 * 4 * 2
    assert len(_seen(selected, "c", "e")) == 2 * 2


def test_exclude():
    items = _items(a=[1, 2, 3], e=AXES["e"])
    selected = common.select_pairwise(items, exclude=lambda p: p["a"] == 2)
    assert {it.callspec.params["a"] for it in selected} == {1, 3}
