"""A conservative linear precheck for conflict addresses.

A full race implies its address/activity/template conditions. Replacing
nonlinear integer subterms and array Selects by unconstrained, consistently
shared variables only enlarges that necessary condition. Snapshot guards
and original domains remain, so an inactive array fallback need not pull
an otherwise linear address query into array solving. For ``q*r+d`` we may
also retain injectivity when the ORIGINAL condition explicitly contains
``r > 0`` and ``0 <= d < r``. Then equality of two flattened indices with
the same shared radix entails equality of their quotients and digits.
Only UNSAT of this weaker formula can discard a pair; SAT and unknown are
inconclusive and leave the complete HB/rf/feasibility query unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import z3

from .snapshot_lemmas import snapshot_table_lemmas

# Private switches that turn one optional precheck path off, so tests can
# compare it with the full query; analysis always uses both.
_ENABLE_SNAPSHOT_PRECHECK = True
_ENABLE_RADIX_PRECHECK = True
_PRECHECK_TIMEOUT_MS = 500


def conflict_precheck_features(expr) -> tuple[bool, bool]:
    """Return (symbolic product, Select), cached per lowered event."""
    if not isinstance(expr, z3.ExprRef):
        return False, False
    product, snapshot = False, False
    pending, seen = [expr], set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        snapshot |= _ENABLE_SNAPSHOT_PRECHECK and z3.is_select(node)
        product |= (
            _ENABLE_RADIX_PRECHECK
            and z3.is_mul(node)
            and sum(not z3.is_int_value(c) for c in node.children()) > 1
        )
        if product and snapshot:
            break
        pending.extend(node.children())
    return product, snapshot


def needs_conflict_precheck(expr) -> bool:
    return any(conflict_precheck_features(expr))


def _conjuncts(expr):
    if z3.is_and(expr):
        for child in expr.children():
            yield from _conjuncts(child)
    else:
        yield expr


def _numeral(expr):
    return expr.as_long() if z3.is_int_value(expr) else None


def _explicit_digit_bounds(conditions):
    """Read only conjunctions of asserted bounds, never disjunction arms."""
    nonnegative, positive, upper = set(), set(), set()
    for condition in conditions:
        for atom in _conjuncts(condition):
            if not z3.is_app(atom) or atom.num_args() != 2:
                continue
            kind = atom.decl().kind()
            left, right = atom.children()
            if kind in (z3.Z3_OP_LE, z3.Z3_OP_LT):
                left, right = right, left
                kind = z3.Z3_OP_GE if kind == z3.Z3_OP_LE else z3.Z3_OP_GT
            if kind not in (z3.Z3_OP_GE, z3.Z3_OP_GT):
                continue
            constant = _numeral(right)
            if constant is not None:
                floor = constant + (kind == z3.Z3_OP_GT)
                if floor >= 0:
                    nonnegative.add(left)
                if floor >= 1:
                    positive.add(left)
            # After normalization, left > right means digit < radix.
            if kind == z3.Z3_OP_GT:
                upper.add((right, left))
    return {
        (digit, radix)
        for digit, radix in upper
        if digit in nonnegative and (radix in positive or (_numeral(radix) or 0) > 0)
    }


def _mixed_radix(expr, bounds):
    if not z3.is_add(expr) or expr.num_args() != 2:
        return None
    for product, digit in (expr.children(), tuple(reversed(expr.children()))):
        if not z3.is_mul(product) or product.num_args() != 2:
            continue
        for quotient, radix in (
            product.children(),
            tuple(reversed(product.children())),
        ):
            if (digit, radix) in bounds:
                return quotient, radix, digit
    return None


_LINEAR_OPS = {
    z3.Z3_OP_TRUE,
    z3.Z3_OP_FALSE,
    z3.Z3_OP_AND,
    z3.Z3_OP_OR,
    z3.Z3_OP_NOT,
    z3.Z3_OP_IMPLIES,
    z3.Z3_OP_XOR,
    z3.Z3_OP_EQ,
    z3.Z3_OP_DISTINCT,
    z3.Z3_OP_ITE,
    z3.Z3_OP_LE,
    z3.Z3_OP_LT,
    z3.Z3_OP_GE,
    z3.Z3_OP_GT,
    z3.Z3_OP_ADD,
    z3.Z3_OP_SUB,
    z3.Z3_OP_UMINUS,
}


class _PureSelectExpressionCache:
    """One two-copy solver's assumption-independent expression rewrites.

    Only the simplify-first path uses this cache: it never has radix lemmas
    or uses a pair's bounds/correspondence to rewrite another expression.
    Keys are actual immutable Z3 ASTs, after any cross/intra substitutions.
    Values are expressions, never Solver state or query decisions.
    Every abstracted node records whether its traversal saw a Select, so a
    cache hit retains the same applicability information as a fresh walk.
    """

    def __init__(self):
        self._normalized = {}
        self._abstracted = {}

    def normalize(self, expression):
        cached = self._normalized.get(expression)
        if cached is None:
            cached = z3.simplify(expression)
            self._normalized[expression] = cached
        return cached

    def abstract(self, expression):
        cached = self._abstracted.get(expression)
        if cached is not None:
            return cached
        if z3.is_const(expression) and expression.sort().kind() in (
            z3.Z3_INT_SORT,
            z3.Z3_BOOL_SORT,
        ):
            result, saw_select = expression, False
        else:
            kind = expression.decl().kind() if z3.is_app(expression) else None
            allowed = kind in _LINEAR_OPS or kind in (
                z3.Z3_OP_MUL,
                z3.Z3_OP_IDIV,
                z3.Z3_OP_MOD,
            )
            saw_select = z3.is_select(expression)
            children = None
            if allowed and all(
                child.sort().kind() in (z3.Z3_INT_SORT, z3.Z3_BOOL_SORT)
                for child in expression.children()
            ):
                rewritten = [self.abstract(child) for child in expression.children()]
                children = [child for child, _ in rewritten]
                saw_select |= any(seen for _, seen in rewritten)
                if z3.is_mul(expression):
                    allowed = sum(not z3.is_int_value(child) for child in children) <= 1
                if kind in (z3.Z3_OP_IDIV, z3.Z3_OP_MOD):
                    allowed = len(children) == 2 and (_numeral(children[1]) or 0) != 0
            else:
                allowed = False
            if allowed:
                result = expression.decl()(*children)
            elif expression.sort().kind() == z3.Z3_INT_SORT:
                result = z3.FreshInt("conflict_integer", ctx=expression.ctx)
            elif expression.sort().kind() == z3.Z3_BOOL_SORT:
                result = z3.FreshBool("conflict_boolean", ctx=expression.ctx)
            else:
                raise TypeError("non-integer conflict expression")
        cached = result, saw_select
        self._abstracted[expression] = cached
        return cached

    def relaxation(self, conditions, *, normalized_conditions=None):
        # A diagnostic can disable this factor after a cache was populated.
        # An old memo entry must not silently re-enable the optional path.
        if not _ENABLE_SNAPSHOT_PRECHECK:
            return None
        try:
            normalized = normalized_conditions
            if normalized is None:
                normalized = [self.normalize(condition) for condition in conditions]
            for condition in normalized:
                if z3.is_false(condition):
                    return condition
            rewritten = [self.abstract(condition) for condition in normalized]
            if not any(seen for _, seen in rewritten):
                return None
            return z3.simplify(z3.And(*(expression for expression, _ in rewritten)))
        except (TypeError, z3.Z3Exception):
            return None


def _linear_relaxation(
    conditions,
    correspondence=(),
    *,
    same_instance=False,
    simplify_first=False,
    expression_cache=None,
    normalized_conditions=None,
):
    """Return a sound QF_LIA relaxation, or None outside the cheap pattern.

    ``correspondence`` maps b-copy variables onto matching a-copy variables
    for SHAPE comparison only. It never identifies actual copies in a
    cross-instance condition, and radix matching uses its original AST.
    ``normalized_conditions`` reuses the pure-Select path's preparation;
    it is ignored by the general path, which preserves original radix syntax.
    """
    if simplify_first and expression_cache is not None:
        return expression_cache.relaxation(
            conditions, normalized_conditions=normalized_conditions
        )
    if simplify_first:
        # A pure Select path has no mixed-radix syntax to preserve. Fold
        # constants and trivial conflict predicates in Z3's native rewriter
        # before walking the DAG in Python, once per condition.
        conditions = (
            normalized_conditions
            if normalized_conditions is not None
            else [z3.simplify(condition) for condition in conditions]
        )
        if any(z3.is_false(condition) for condition in conditions):
            return z3.BoolVal(False)
    bounds = set() if simplify_first else _explicit_digit_bounds(conditions)
    memo, flattened, array_reads = {}, [], []
    supported = _LINEAR_OPS

    def abstract(expr):
        if expr in memo:
            return memo[expr]
        match = _mixed_radix(expr, bounds) if _ENABLE_RADIX_PRECHECK else None
        if match is not None and _numeral(match[1]) is None:
            quotient, radix, digit = match
            result = z3.FreshInt("conflict_flat")
            memo[expr] = result
            flattened.append((expr, result, quotient, radix, digit))
            return result
        if z3.is_const(expr) and expr.sort().kind() in (
            z3.Z3_INT_SORT,
            z3.Z3_BOOL_SORT,
        ):
            return expr
        if z3.is_select(expr) and not _ENABLE_SNAPSHOT_PRECHECK:
            raise TypeError("snapshot precheck disabled")
        kind = expr.decl().kind() if z3.is_app(expr) else None
        arithmetic = kind in (z3.Z3_OP_MUL, z3.Z3_OP_IDIV, z3.Z3_OP_MOD)
        allowed = kind in supported or arithmetic
        children = None
        if allowed and all(
            c.sort().kind() in (z3.Z3_INT_SORT, z3.Z3_BOOL_SORT)
            for c in expr.children()
        ):
            # Protect the original mixed-radix nodes above, then fold the
            # remaining arithmetic bottom-up. In particular, truncating
            # division emits If(2 >= 0, 2, -2) as its denominator: deciding
            # linearity before this fold would lose the index/domain link.
            children = [abstract(child) for child in expr.children()]
            if z3.is_mul(expr):
                allowed = sum(not z3.is_int_value(c) for c in children) <= 1
            if kind in (z3.Z3_OP_IDIV, z3.Z3_OP_MOD):
                allowed = len(children) == 2 and (_numeral(children[1]) or 0) != 0
        else:
            allowed = False
        if allowed:
            result = expr.decl()(*children)
            if not simplify_first:
                result = z3.simplify(result)
        elif expr.sort().kind() == z3.Z3_INT_SORT:
            result = z3.FreshInt("conflict_integer")
            if z3.is_select(expr):
                array_reads.append(expr)
        elif expr.sort().kind() == z3.Z3_BOOL_SORT:
            result = z3.FreshBool("conflict_boolean")
            if z3.is_select(expr):
                array_reads.append(expr)
        else:
            raise TypeError("non-integer conflict expression")
        memo[expr] = result
        return result

    try:
        relaxed = [abstract(condition) for condition in conditions]
        groups = defaultdict(list)
        # Nested quotients can expose more flattened nodes during abstraction.
        index = 0
        while index < len(flattened):
            expr, value, quotient, radix, digit = flattened[index]
            index += 1
            shape = z3.substitute(expr, *correspondence) if correspondence else expr
            groups[(radix, shape)].append((value, abstract(quotient), abstract(digit)))
        matched_radix = bool(flattened) and (
            same_instance or any(len(group) > 1 for group in groups.values())
        )
        if not array_reads and not matched_radix:
            return None
        for group in groups.values():
            for i, (va, qa, da) in enumerate(group):
                for vb, qb, db in group[i + 1 :]:
                    relaxed.append(z3.Implies(va == vb, z3.And(qa == qb, da == db)))
        return z3.simplify(z3.And(*relaxed))
    except (TypeError, z3.Z3Exception):
        return None


def conflict_impossible(
    conditions: list[Any],
    correspondence=(),
    *,
    same_instance=False,
    simplify_first=False,
    expression_cache=None,
    snapshot_cache=None,
) -> bool:
    """Cheap UNSAT-only precheck; callers retain their original query budget."""
    normalized = None
    weak_attempted = False
    if simplify_first and _ENABLE_SNAPSHOT_PRECHECK:
        # Pure Select abstraction can already rule out an address pair without
        # inspecting the snapshot cells. Only UNSAT discharges the pair; an
        # inconclusive attempt retains the original guarded-lemma check below.
        normalize = (
            expression_cache.normalize if expression_cache is not None else z3.simplify
        )
        try:
            # Reuse this path's existing normalization in the guarded attempt.
            # Snapshot certificates still receive the ORIGINAL conditions.
            if expression_cache is None:
                normalized = [normalize(condition) for condition in conditions]
            else:
                try:
                    normalized = [normalize(condition) for condition in conditions]
                except (TypeError, z3.Z3Exception):
                    # The cached relaxation has always treated normalization
                    # failure as unsupported rather than throwing to its caller.
                    normalized = None
            relaxed = (
                _linear_relaxation(
                    conditions,
                    correspondence,
                    same_instance=same_instance,
                    simplify_first=True,
                    expression_cache=expression_cache,
                    normalized_conditions=normalized,
                )
                if normalized is not None
                else None
            )
            if relaxed is not None and _relaxation_impossible(relaxed):
                return True
        except (TypeError, z3.Z3Exception):
            # The extra attempt must not prevent an original guarded query
            # from succeeding, including when lemma extraction yields no facts.
            pass
        else:
            weak_attempted = True
    lemmas = ()
    if _ENABLE_SNAPSHOT_PRECHECK:
        # These facts follow from snapshot cells ALREADY in this query.
        # Keep all source conditions and the guards on each derived lemma.
        try:
            lemmas = (
                snapshot_cache.lemmas(conditions)
                if snapshot_cache is not None
                else snapshot_table_lemmas(z3.And(*conditions))
            )
        except (TypeError, z3.Z3Exception):
            lemmas = ()
        if lemmas:
            conditions = [*conditions, *lemmas]
            if normalized is not None:
                try:
                    normalized.extend(normalize(lemma) for lemma in lemmas)
                except (TypeError, z3.Z3Exception):
                    if expression_cache is None:
                        raise
                    return False
    if weak_attempted and not lemmas:
        # With no added facts, the original precheck is exactly the attempted
        # relaxation, including whether it is supported. When applicable, it
        # has already received the original full 500 ms precheck budget.
        return False
    relaxed = _linear_relaxation(
        conditions,
        correspondence,
        same_instance=same_instance,
        simplify_first=simplify_first,
        expression_cache=expression_cache,
        normalized_conditions=normalized,
    )
    if relaxed is None:
        return False
    return _relaxation_impossible(relaxed)


def _relaxation_impossible(relaxed) -> bool:
    if z3.is_false(relaxed):
        return True
    solver = z3.SolverFor("QF_LIA")
    # This is only a speculative shortcut. Expiring it has no effect on
    # the full query's budget, answer, or proof scope. A weak-first attempt
    # never consumes the original guarded-lemma check's independent budget.
    solver.set(timeout=_PRECHECK_TIMEOUT_MS)
    solver.add(relaxed)
    return solver.check() == z3.unsat
