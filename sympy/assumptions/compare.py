"""
Batch comparison of proposition truth values under two assumption sets.

This module provides :func:`compare_assumptions`, which evaluates a
collection of propositions independently under an "old" and a "new" set
of assumptions and reports how each conclusion changes.

The comparison is strictly local: assumptions are passed to :func:`~.ask`
per query, so ``global_assumptions`` is never modified. Contradictory
assumptions make an environment *invalid* instead of making every
proposition vacuously true.
"""
from __future__ import annotations

from sympy.core.basic import Basic
from sympy.core.kind import BooleanKind
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError, sympify
from sympy.logic.boolalg import And
from sympy.logic.inference import satisfiable

from .ask import _extract_all_facts, _normalize_expr, ask
from .ask_generated import get_all_known_facts
from .assume import AppliedPredicate, Predicate, global_assumptions
from .cnf import CNF, EncodedCNF

# A Boolean or a single assumption may itself be an iterable of args, so
# only treat plain containers as collections of separate items.
_SEQUENCE_TYPES = (list, tuple, set, frozenset)

_CHANGE_LABELS = (
    'unchanged', 'flipped', 'determined', 'undetermined', 'unavailable',
)


class ComparisonProblem:
    """
    Description of an input problem that prevents judging a proposition
    or an environment.

    Attributes
    ==========

    kind : str
        One of ``'invalid_proposition'``, ``'invalid_assumptions'``,
        ``'inconsistent_environment'``, ``'symbol_identity_conflict'``
        or ``'evaluation_error'``.

    message : str
        Human readable explanation.

    proposition : Boolean or None
        The offending proposition, when the problem is tied to one.

    environment : str or None
        ``'old'`` or ``'new'`` when the problem is tied to one
        environment.

    Examples
    ========

    Problems are exposed on the comparison result rather than raised:

    >>> from sympy import compare_assumptions, Q
    >>> result = compare_assumptions([Q.positive(1)], True, 5)
    >>> result.new_valid
    False
    >>> result.problems[0].kind
    'invalid_assumptions'

    """

    __slots__ = ('kind', 'message', 'proposition', 'environment')

    def __init__(self, kind, message, proposition=None, environment=None):
        self.kind = kind
        self.message = message
        self.proposition = proposition
        self.environment = environment

    def __eq__(self, other):
        return isinstance(other, ComparisonProblem) and all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self.__slots__)

    def __hash__(self):
        return hash((self.kind, self.message))

    def __repr__(self):
        parts = [f"ComparisonProblem({self.kind!r}, {self.message!r}"]
        if self.environment is not None:
            parts.append(f", environment={self.environment!r}")
        parts.append(")")
        return ''.join(parts)


class PropositionComparison:
    """
    Result of evaluating one proposition in both environments.

    Attributes
    ==========

    proposition : Boolean
        The proposition which was evaluated.

    old, new : True, False, or None
        Truth value of the proposition in the old/new environment.
        ``None`` means that the value either could not be determined or
        was not judged (see ``problems`` and the environment validity
        flags on the enclosing result).

    change : str
        How the conclusion changes from old to new:

        ``'unchanged'``
            Same value on both sides.
        ``'flipped'``
            ``True`` to ``False`` or vice versa.
        ``'determined'``
            ``None`` to ``True`` or ``False``.
        ``'undetermined'``
            ``True`` or ``False`` to ``None``.
        ``'unavailable'``
            At least one side was not judged (invalid input or invalid
            environment).

    problems : tuple of ComparisonProblem
        Problems tied to this proposition.

    """

    __slots__ = ('proposition', 'old', 'new', 'change', 'problems')

    def __init__(self, proposition, old, new, change, problems=()):
        self.proposition = proposition
        self.old = old
        self.new = new
        self.change = change
        self.problems = tuple(problems)

    def __repr__(self):
        return (f"PropositionComparison({self.proposition!r}, "
                f"old={self.old!r}, new={self.new!r}, "
                f"change={self.change!r})")


class AssumptionsComparison:
    """
    Result of :func:`compare_assumptions`.

    Attributes
    ==========

    comparisons : tuple of PropositionComparison
        One record per input proposition, in input order.

    old_assumptions, new_assumptions : Boolean
        Normalized assumption expressions of each environment.

    old_valid, new_valid : bool
        Whether the explicit assumptions of the environment are
        consistent. Conclusions under an invalid environment are not
        judged (they are never reported as ``True``).

    problems : tuple of ComparisonProblem
        Environment-level and global input problems, including symbol
        name/identity conflicts and inconsistent environments.

    """

    __slots__ = ('comparisons', 'old_assumptions', 'new_assumptions',
        'old_valid', 'new_valid', 'problems')

    def __init__(self, comparisons, old_assumptions, new_assumptions,
            old_valid, new_valid, problems):
        self.comparisons = tuple(comparisons)
        self.old_assumptions = old_assumptions
        self.new_assumptions = new_assumptions
        self.old_valid = old_valid
        self.new_valid = new_valid
        self.problems = tuple(problems)

    @property
    def changes(self):
        """
        Mapping of change labels to tuples of propositions with that
        change.
        """
        grouped = {label: [] for label in _CHANGE_LABELS}
        for comparison in self.comparisons:
            grouped[comparison.change].append(comparison.proposition)
        return {label: tuple(props) for label, props in grouped.items()}

    def __repr__(self):
        def status(expr, valid):
            mark = '' if valid else ' [INVALID]'
            return f"{expr}{mark}"
        lines = [
            "AssumptionsComparison(",
            f"  old assumptions: {status(self.old_assumptions, self.old_valid)}",
            f"  new assumptions: {status(self.new_assumptions, self.new_valid)}",
        ]
        for comparison in self.comparisons:
            lines.append(f"  {comparison}")
        if self.problems:
            lines.append("  problems:")
            for problem in self.problems:
                lines.append(f"    - [{problem.kind}] {problem.message}")
        lines.append(")")
        return '\n'.join(lines)


def _as_boolean(expr, description):
    try:
        expr = sympify(expr)
    except SympifyError as exc:
        return None, f"{description} could not be parsed: {exc}"
    if isinstance(expr, Predicate) or getattr(expr, 'kind', None) is not BooleanKind:
        return None, (f"{description} must be a Boolean logical expression, "
            f"got {expr!r}")
    return _normalize_expr(expr), None


def _prepare_environment(raw, label):
    """Return (normalized Boolean, problem or None)."""
    description = f"{label} assumptions"
    if isinstance(raw, _SEQUENCE_TYPES):
        terms = []
        for item in raw:
            term, problem = _as_boolean(item, description)
            if problem is not None:
                return None, ComparisonProblem('invalid_assumptions', problem,
                    environment=label)
            terms.append(term)
        expr = And(*terms) if terms else S.true
    else:
        expr, problem = _as_boolean(raw, description)
        if problem is not None:
            return None, ComparisonProblem('invalid_assumptions', problem,
                environment=label)
    return expr, None


def _inconsistency_message(expr):
    """
    Check the satisfiability of the explicit assumptions independently
    of any proposition, using the same predicate-level encoding as
    ``ask``. Return an error message when inconsistent, else None.

    Facts are grouped by expression identity so that distinct symbols
    sharing a name are never conflated.
    """
    cnf = CNF.from_prop(expr)
    exprs = set()
    for predicate in cnf.all_predicates():
        if isinstance(predicate, AppliedPredicate) and len(predicate.arguments) == 1:
            exprs.add(predicate.arguments[0])

    base = EncodedCNF()
    base.from_cnf(CNF(get_all_known_facts()))
    for single_expr in exprs:
        local_facts = _extract_all_facts(cnf, (single_expr,))
        if not local_facts.clauses:
            continue
        encoded = base.copy()
        encoded.add_from_cnf(local_facts)
        if satisfiable(encoded) is False:
            return f"inconsistent assumptions {expr}"
    return None


def _named_symbols(expr):
    if isinstance(expr, Basic):
        return expr.atoms(Symbol)
    return set()


def _identity_conflicts(propositions, old_expr, new_expr):
    """
    Find names that denote distinct symbol objects in different places.

    Symbols that merely share a name but have different identities (for
    example distinct :class:`~.Dummy` instances) must not be treated as
    the same symbol. Such situations are reported as input problems.
    """
    sources = [('proposition[%d]' % i, _named_symbols(p))
        for i, p in enumerate(propositions)]
    if old_expr is not None:
        sources.append(('old environment', _named_symbols(old_expr)))
    if new_expr is not None:
        sources.append(('new environment', _named_symbols(new_expr)))

    # name -> {symbol object: [locations]}
    by_name = {}
    for location, symbols in sources:
        for symbol in symbols:
            entry = by_name.setdefault(symbol.name, {})
            entry.setdefault(symbol, []).append(location)

    problems = []
    for name, identities in by_name.items():
        if len(identities) > 1:
            locations = sorted({loc for locs in identities.values()
                for loc in locs})
            message = (f"symbol name {name!r} denotes distinct objects in: "
                + ', '.join(locations) + "; assumptions are not transferred "
                "between them")
            problems.append(ComparisonProblem('symbol_identity_conflict',
                message))
    return problems


def _classify_change(old, new, old_judged, new_judged):
    if not (old_judged and new_judged):
        return 'unavailable'
    if old is new:
        return 'unchanged'
    if old is None:
        return 'determined'
    if new is None:
        return 'undetermined'
    return 'flipped'


def compare_assumptions(propositions, old_assumptions=True,
        new_assumptions=True, *, context=global_assumptions):
    """
    Evaluate a batch of propositions independently under two assumption
    sets and report how the conclusions change.

    Explanation
    ===========

    Each proposition is evaluated once under *old_assumptions* and once
    under *new_assumptions* with :func:`~.ask`. The two environments are
    never combined with each other. ``global_assumptions`` is only read
    (through *context*) and never modified, so calling this function has
    no effect on the surrounding assumption environment, including when
    nested.

    Contradictory explicit assumptions make the corresponding
    environment invalid: its propositions are reported as not judged
    instead of becoming vacuously true. Invalid input (non-Boolean
    propositions or assumptions, unparseable expressions) is collected
    in the result's ``problems`` rather than aborting the batch.

    Parameters
    ==========

    propositions : Boolean or iterable of Boolean
        Propositions to judge in both environments.

    old_assumptions, new_assumptions : Boolean or iterable of Boolean, optional
        Explicit assumptions of the old and new environment. An
        iterable is combined with :func:`~.And`; the default is no
        assumptions.

    context : AssumptionsContext, optional
        Ambient assumptions applied in addition to each environment,
        by default ``global_assumptions``. It is read only.

    Returns
    =======

    AssumptionsComparison
        Per-proposition tri-state results (``True``/``False``/``None``),
        change labels, environment validity flags and collected input
        problems.

    Examples
    ========

    >>> from sympy import compare_assumptions, Q
    >>> from sympy.abc import x
    >>> result = compare_assumptions(
    ...     [Q.positive(x), Q.even(x)], Q.positive(x), Q.even(x))
    >>> [(c.old, c.new, c.change) for c in result.comparisons]
    [(True, None, 'undetermined'), (None, True, 'determined')]

    A contradictory environment is shown as invalid instead of making
    the proposition true:

    >>> result = compare_assumptions(
    ...     [Q.positive(x)], True, Q.positive(x) & Q.negative(x))
    >>> result.new_valid
    False
    >>> result.comparisons[0].new is None
    True
    >>> result.problems[0].kind
    'inconsistent_environment'

    See Also
    ========

    ask, assuming

    """
    if isinstance(propositions, _SEQUENCE_TYPES):
        raw_propositions = list(propositions)
    else:
        raw_propositions = [propositions]

    old_expr, old_problem = _prepare_environment(old_assumptions, 'old')
    new_expr, new_problem = _prepare_environment(new_assumptions, 'new')

    problems = []
    if old_problem is not None:
        problems.append(old_problem)
    if new_problem is not None:
        problems.append(new_problem)

    old_valid = old_expr is not None
    new_valid = new_expr is not None
    if old_valid:
        message = _inconsistency_message(old_expr)
        if message is not None:
            old_valid = False
            problems.append(ComparisonProblem('inconsistent_environment',
                message, environment='old'))
    if new_valid:
        message = _inconsistency_message(new_expr)
        if message is not None:
            new_valid = False
            problems.append(ComparisonProblem('inconsistent_environment',
                message, environment='new'))

    normalized = []
    judged_propositions = []
    for raw in raw_propositions:
        proposition, message = _as_boolean(raw, 'proposition')
        if message is not None:
            normalized.append((raw, message))
        else:
            normalized.append((proposition, None))
            judged_propositions.append(proposition)

    problems.extend(_identity_conflicts(judged_propositions, old_expr, new_expr))

    def query(proposition, env_expr, label, valid):
        if not valid:
            return None, False, [ComparisonProblem('inconsistent_environment',
                f"proposition not judged because the {label} environment "
                "is invalid", proposition=proposition, environment=label)]
        try:
            result = ask(proposition, assumptions=env_expr, context=context)
        except ValueError as exc:
            return None, False, [ComparisonProblem('inconsistent_environment',
                str(exc), proposition=proposition, environment=label)]
        except (TypeError, SympifyError) as exc:
            return None, False, [ComparisonProblem('evaluation_error',
                f"{type(exc).__name__}: {exc}", proposition=proposition,
                environment=label)]
        return result, True, []

    comparisons = []
    for proposition, error in normalized:
        record_problems = []
        if error is not None:
            record_problems.append(ComparisonProblem('invalid_proposition',
                error, proposition=proposition))
            comparisons.append(PropositionComparison(proposition, None, None,
                'unavailable', record_problems))
            continue

        old_value, old_judged, old_problems = query(proposition, old_expr,
            'old', old_valid)
        new_value, new_judged, new_problems = query(proposition, new_expr,
            'new', new_valid)
        record_problems.extend(old_problems)
        record_problems.extend(new_problems)
        change = _classify_change(old_value, new_value, old_judged, new_judged)
        comparisons.append(PropositionComparison(proposition, old_value,
            new_value, change, record_problems))

    return AssumptionsComparison(comparisons, old_expr, new_expr,
        old_valid, new_valid, problems)
