"""Query-local, certified constant folding for variable integer divisors.

Grouped tile addresses often have a variable divisor that is constant under
the access mask. Only original top-level conjuncts may certify this fact.
If Q entails E = (term == value), then Q is equivalent to substitute(Q) AND E.
Keep E so that SAT models still satisfy the original query. No launch bounds,
snapshot values, or other premises are introduced here.
"""

from __future__ import annotations

import time

import z3


_ENABLE_GUARDED_DIVISION = True
_CERTIFY_TIMEOUT_MS = 200


class GuardedDivisionApplicabilityCache:
    """Memoize only whether an immutable AST may need divisor certification.

    One solver can share this cache across conflict queries with common DAGs.
    A negative answer avoids another complete walk of those DAGs. Positive
    answers still run the original query-local certification, with its original
    budget and premises. In particular, neither guards nor proven equalities
    are cached here.
    """

    def __init__(self):
        # Retain contexts and AST wrappers: Z3 may recycle AST identifiers once
        # nodes are released, and identifiers alone do not distinguish contexts.
        self._contexts = {}

    def may_have_variable_divisor(self, expression):
        context_key = id(expression.ctx)
        context_entry = self._contexts.get(context_key)
        if context_entry is None:
            context_entry = (expression.ctx, {})
            self._contexts[context_key] = context_entry
        nodes = context_entry[1]

        # Explicit postorder traversal also handles deeply nested expressions.
        # A cached subtree costs one lookup regardless of its size.
        pending = [(expression, None)]
        while pending:
            node, children = pending.pop()
            node_id = node.get_id()
            if node_id in nodes:
                continue
            if children is None:
                if z3.is_quantifier(node) or not z3.is_app(node):
                    # Do not infer applicability from unsupported syntax. The
                    # original normalization pass will leave the query alone.
                    nodes[node_id] = (node, True)
                    continue
                if node.decl().kind() in (
                    z3.Z3_OP_IDIV,
                    z3.Z3_OP_MOD,
                ) and not z3.is_int_value(node.arg(1)):
                    nodes[node_id] = (node, True)
                    continue
                children = node.children()
                if not children:
                    nodes[node_id] = (node, False)
                    continue
                pending.append((node, children))
                pending.extend((child, None) for child in children)
            else:
                nodes[node_id] = (
                    node,
                    any(nodes[child.get_id()][1] for child in children),
                )
        return nodes[expression.get_id()][1]


def _conjuncts(expression):
    pending = [expression]
    while pending:
        node = pending.pop()
        if z3.is_and(node):
            pending.extend(reversed(node.children()))
        else:
            yield node


def guarded_division_normal_form(expression, *, applicability_cache=None):
    """Keep the original query if the optional certificate machinery fails."""
    try:
        if not _ENABLE_GUARDED_DIVISION:
            return expression
        if (
            applicability_cache is not None
            and not applicability_cache.may_have_variable_divisor(expression)
        ):
            return expression
        return _guarded_division_normal_form(expression)
    except z3.Z3Exception:
        return expression


def _guarded_division_normal_form(expression):
    """Return an equivalent formula, or the original when no fact is proved.

    Candidate discovery is syntactic, but adoption always requires an independent
    UNSAT proof from a subset of Q. Restrict that subset to one local scalar guard
    and its existing variable bounds, avoiding the full two-copy conflict query.
    SAT/unknown in certification simply leaves that term unchanged."""
    if not _ENABLE_GUARDED_DIVISION:
        return expression
    nodes, pending = [], [(expression, False)]
    seen, complete = set(), set()
    while pending:
        node, ready = pending.pop()
        if node in complete:
            continue
        if z3.is_quantifier(node) or not z3.is_app(node):
            return expression
        if not ready:
            if node in seen:
                continue
            seen.add(node)
            pending.append((node, True))
            pending.extend((child, False) for child in node.children())
            continue
        complete.add(node)
        nodes.append(node)

    divisors = [
        node.arg(1)
        for node in nodes
        if node.decl().kind() in (z3.Z3_OP_IDIV, z3.Z3_OP_MOD)
        and not z3.is_int_value(node.arg(1))
    ]
    if not divisors:
        return expression

    variables = {}
    for node in nodes:
        if z3.is_const(node) and node.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            variables[node] = frozenset((node,))
        else:
            variables[node] = frozenset().union(
                *(variables[child] for child in node.children())
            )

    candidates, visited = [], set()
    pending = list(divisors)
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        if node in divisors or (
            node.decl().kind() == z3.Z3_OP_IDIV
            and z3.is_int_value(node.arg(1))
            and node.arg(1).as_long() > 0
        ):
            candidates.append(node)
        pending.extend(node.children())

    atoms = list(_conjuncts(expression))
    # Only scalar integer formulas with no arrays, quantifiers or uninterpreted
    # functions are useful local arithmetic certificates.
    arithmetic = {
        z3.Z3_OP_TRUE,
        z3.Z3_OP_FALSE,
        z3.Z3_OP_ANUM,
        z3.Z3_OP_UNINTERPRETED,
        z3.Z3_OP_AND,
        z3.Z3_OP_OR,
        z3.Z3_OP_NOT,
        z3.Z3_OP_ITE,
        z3.Z3_OP_EQ,
        z3.Z3_OP_DISTINCT,
        z3.Z3_OP_LE,
        z3.Z3_OP_LT,
        z3.Z3_OP_GE,
        z3.Z3_OP_GT,
        z3.Z3_OP_ADD,
        z3.Z3_OP_SUB,
        z3.Z3_OP_UMINUS,
        z3.Z3_OP_MUL,
        z3.Z3_OP_IDIV,
        z3.Z3_OP_MOD,
    }
    scalar = {}
    for node in nodes:
        scalar[node] = (
            node.sort().kind() in (z3.Z3_INT_SORT, z3.Z3_BOOL_SORT)
            and node.decl().kind() in arithmetic
            and (node.decl().kind() != z3.Z3_OP_UNINTERPRETED or z3.is_const(node))
            and all(scalar[child] for child in node.children())
        )
    simple_bounds = []
    for atom in atoms:
        comparison = atom.arg(0) if z3.is_not(atom) else atom
        if (
            comparison.decl().kind()
            in (z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT, z3.Z3_OP_EQ)
            and comparison.num_args() == 2
            and any(z3.is_int_value(child) for child in comparison.children())
            and all(
                z3.is_int_value(child)
                or (z3.is_const(child) and child.sort().kind() == z3.Z3_INT_SORT)
                for child in comparison.children()
            )
        ):
            simple_bounds.append(atom)

    substitutions, equalities = [], []
    deadline = time.monotonic() + _CERTIFY_TIMEOUT_MS / 1000

    def certified_check(solver):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return z3.unknown
        solver.set(timeout=max(1, int(remaining * 1000)))
        return solver.check()

    for term in candidates:
        term_variables = variables[term]
        if not term_variables or not scalar[term]:
            continue
        # Try each single local mask rather than joining unrelated guards.
        for atom in atoms:
            if time.monotonic() >= deadline:
                break
            atom_variables = variables[atom]
            if (
                not scalar[atom]
                or not term_variables <= atom_variables
                or len(atom_variables - term_variables) > 1
                or atom in simple_bounds
            ):
                continue
            # A shared variable alone does not make this a guard for the term.
            if z3.eq(z3.substitute(atom, (term, z3.IntVal(0, ctx=term.ctx))), atom):
                continue
            premises = [atom] + [
                bound for bound in simple_bounds if variables[bound] <= atom_variables
            ]
            solver = z3.Solver(ctx=expression.ctx)
            solver.add(*premises)
            if certified_check(solver) != z3.sat:
                continue
            value = solver.model().eval(term, model_completion=True)
            if not z3.is_int_value(value):
                continue
            equality = term == value
            solver.add(z3.Not(equality))
            if certified_check(solver) == z3.unsat:
                substitutions.append((term, value))
                equalities.append(equality)
                break

    if not substitutions:
        return expression
    return z3.And(z3.simplify(z3.substitute(expression, *substitutions)), *equalities)
