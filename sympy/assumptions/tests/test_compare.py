from __future__ import annotations

from sympy import Q, ask, Dummy
from sympy.abc import x, y
from sympy.assumptions import (AssumptionsContext, compare_assumptions,
    global_assumptions)
from sympy.assumptions.assume import assuming


def _by_change(result):
    return {label: props for label, props in result.changes.items() if props}


def test_compare_tri_state_and_changes():
    propositions = [
        Q.positive(x),
        Q.negative(x),
        Q.real(x),
        Q.even(x),
    ]
    result = compare_assumptions(propositions, Q.positive(x), Q.negative(x))
    values = [(c.old, c.new, c.change) for c in result.comparisons]
    assert values == [
        (True, False, 'flipped'),
        (False, True, 'flipped'),
        (True, True, 'unchanged'),
        (None, None, 'unchanged'),
    ]
    assert result.old_valid and result.new_valid
    assert result.problems == ()

    result = compare_assumptions([Q.even(x), Q.positive(x)], True, Q.even(x))
    values = [(c.old, c.new, c.change) for c in result.comparisons]
    assert values == [
        (None, True, 'determined'),
        (None, None, 'unchanged'),
    ]

    result = compare_assumptions([Q.positive(x)], Q.positive(x), True)
    assert (result.comparisons[0].old, result.comparisons[0].new,
        result.comparisons[0].change) == (True, None, 'undetermined')

    assert _by_change(result)['undetermined'] == (Q.positive(x),)


def test_single_boolean_proposition_and_list_assumptions():
    result = compare_assumptions(Q.positive(x),
        [Q.real(x), Q.positive(x)], [Q.real(x)])
    comparison = result.comparisons[0]
    assert comparison.old is True
    assert comparison.new is None
    assert comparison.change == 'undetermined'

    result = compare_assumptions(Q.positive(x), (), ())
    assert result.comparisons[0].old is None
    assert result.old_valid and result.new_valid


def test_environments_are_evaluated_independently():
    result = compare_assumptions(
        [Q.positive(x), Q.even(x)], Q.positive(x),
        Q.positive(x) & Q.negative(x))
    assert result.old_valid is True
    assert result.new_valid is False

    positive, even = result.comparisons
    assert positive.old is True
    assert even.old is None
    for comparison in result.comparisons:
        assert comparison.new is None
        assert comparison.change == 'unavailable'

    problems = [p for p in result.problems
        if p.kind == 'inconsistent_environment']
    assert len(problems) == 1
    assert problems[0].environment == 'new'

    assert ask(Q.positive(x)) is None


def test_contradictory_explicit_assumptions_are_invalid():
    for contradiction in [
            Q.positive(x) & Q.negative(x),
            Q.even(x) & Q.odd(x),
            (Q.positive(x), Q.negative(x)),
            (Q.even(x), Q.odd(x))]:
        result = compare_assumptions([Q.positive(x), Q.real(x)], True,
            contradiction)
        assert result.new_valid is False
        assert result.old_valid is True
        for comparison in result.comparisons:
            assert comparison.new is None
            assert comparison.new is not True
        assert any(p.kind == 'inconsistent_environment' and p.environment == 'new'
            for p in result.problems)


def test_invalid_inputs_are_reported_not_raised():
    result = compare_assumptions([Q.positive(x), 5, 'this is not a bool'],
        Q.positive(x), Q.negative(x))
    good, bad_number, bad_string = result.comparisons
    assert (good.old, good.new, good.change) == (True, False, 'flipped')
    for bad in (bad_number, bad_string):
        assert bad.old is None and bad.new is None
        assert bad.change == 'unavailable'
        assert bad.problems[0].kind == 'invalid_proposition'

    result = compare_assumptions([Q.positive(x)], 5, Q.positive(x))
    assert result.old_valid is False
    assert result.new_valid is True
    assert result.comparisons[0].new is True
    assert any(p.kind == 'invalid_assumptions' and p.environment == 'old'
        for p in result.problems)

    result = compare_assumptions([Q.positive(x)], [Q.positive(x), 5], True)
    assert result.old_valid is False
    assert any(p.kind == 'invalid_assumptions' for p in result.problems)


def test_global_assumptions_are_not_polluted():
    snapshot = global_assumptions.copy()
    compare_assumptions([Q.positive(x)], Q.positive(x), Q.negative(x))
    assert global_assumptions == snapshot

    with assuming(Q.integer(y)):
        result = compare_assumptions([Q.real(y)], True, True)
        assert result.comparisons[0].old is True
        assert global_assumptions == snapshot | {Q.integer(y)}

        with assuming(Q.positive(x)):
            result = compare_assumptions([Q.positive(x)], True, True)
            assert result.comparisons[0].old is True
            compare_assumptions([Q.negative(x)], Q.negative(x), True)
            assert global_assumptions == snapshot | {Q.integer(y),
                Q.positive(x)}

        assert Q.positive(x) not in global_assumptions
        assert global_assumptions == snapshot | {Q.integer(y)}

    assert global_assumptions == snapshot

    with assuming(Q.positive(x)):
        result = compare_assumptions([Q.positive(x)], True, True,
            context=AssumptionsContext())
        assert result.comparisons[0].old is None


def test_nested_calls_are_independent_and_repeatable():
    first = compare_assumptions([Q.positive(x)], Q.positive(x), Q.negative(x))
    second = compare_assumptions([Q.positive(x)], Q.positive(x), Q.negative(x))
    assert [(c.old, c.new) for c in first.comparisons] == \
        [(c.old, c.new) for c in second.comparisons]

    def nested(depth):
        result = compare_assumptions([Q.positive(x)],
            Q.positive(x) if depth % 2 == 0 else Q.negative(x),
            Q.negative(x))
        if depth > 0:
            nested(depth - 1)
        return result, depth

    result, depth = nested(3)
    expected_old = True if depth % 2 == 0 else False
    assert result.comparisons[0].old is expected_old
    assert result.comparisons[0].new is False
    assert global_assumptions == set()


def test_same_name_distinct_identity_symbols_are_not_merged():
    proposition_symbol = Dummy('x')
    old_symbol = Dummy('x')
    new_symbol = Dummy('x')

    result = compare_assumptions([Q.positive(proposition_symbol)],
        [Q.positive(old_symbol)], [Q.positive(new_symbol)])
    comparison = result.comparisons[0]
    assert comparison.old is None
    assert comparison.new is None
    conflicts = [p for p in result.problems
        if p.kind == 'symbol_identity_conflict']
    assert len(conflicts) == 1
    assert 'proposition[0]' in conflicts[0].message
    assert 'old environment' in conflicts[0].message
    assert 'new environment' in conflicts[0].message

    result = compare_assumptions([Q.positive(x)], Q.positive(x), Q.negative(x))
    assert [p for p in result.problems
        if p.kind == 'symbol_identity_conflict'] == []
