"""Entailed range and injectivity facts from existing finite array equations.

This helper never loads a snapshot or adds launch assumptions. It recognizes
only equations already asserted as top-level conjuncts. A finite, contiguous
integer interval and its values certify the resulting guarded lemmas. Values
outside that interval remain unconstrained, including in any-grid queries.
The original formula must remain present when these lemmas are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import z3


# Avoid generating quadratically many optional facts for an unrelated large
# base query. Declining injectivity leaves both the original formula and the
# linear number of valid range facts intact.
_MAX_INJECTIVITY_READS = 16


@dataclass(eq=False, frozen=True, slots=True)
class _CellFacts:
    """A persistent DAG of original asserted cell equations."""

    children: tuple = ()
    cell: tuple | None = None


@dataclass(frozen=True, slots=True)
class _Summary:
    reads: tuple = ()


class SnapshotLemmaCache:
    """One solver's reusable facts about immutable, original expressions.

    Node summaries do not depend on any assumptions. Only conjunctions lift
    cell equations into the asserted-fact summary; bound reads never leave a
    quantifier. Certificates are keyed by the exact current fact summaries
    for each relevant array. Pair-specific reads reuse a common table without
    reparsing its cells. Removing or changing an asserted equation requires
    another certificate for that array. Other arrays are independent.
    Neither mutable condition lists nor solver decisions are memoized.
    """

    def __init__(self):
        self._context = None
        self._children = {}
        self._cells = {}
        self._candidates = {}
        self._summaries = {}
        self._fact_summaries = {}
        self._fact_joins = {}
        self._certificates = {}
        self._guarded_reads = {}
        self._injectivity = {}

    def _node_children(self, node):
        if node not in self._children:
            self._children[node] = tuple(node.children())
        return self._children[node]

    def _cell_equation(self, node):
        if not z3.is_eq(node):
            return None
        left, right = self._node_children(node)
        for read, value in ((left, right), (right, left)):
            if not (
                z3.is_int_value(value) and z3.is_select(read) and read.num_args() == 2
            ):
                continue
            array, index = self._node_children(read)
            if not z3.is_int_value(index):
                continue
            array_sort = array.sort()
            if (
                array_sort.domain().kind() == z3.Z3_INT_SORT
                and array_sort.range().kind() == z3.Z3_INT_SORT
            ):
                # Decode values only when this array actually needs a
                # certificate. Candidate and read scans need only its identity.
                return array, index, value
        return None

    def _cell(self, node):
        if node not in self._cells:
            self._cells[node] = self._cell_equation(node)
        return self._cells[node]

    def _has_facts(self, expression, arrays=None):
        """Look only through asserted conjunctions, without scanning reads."""
        key = expression, arrays
        if key in self._candidates:
            return self._candidates[key]
        pending, seen = [expression], set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            node_key = node, arrays
            if node_key in self._candidates:
                if self._candidates[node_key]:
                    self._candidates[key] = True
                    return True
                continue
            if z3.is_and(node):
                pending.extend(reversed(self._node_children(node)))
                continue
            cell = self._cell(node)
            candidate = cell is not None and (arrays is None or cell[0] in arrays)
            self._candidates[node_key] = candidate
            if candidate:
                self._candidates[key] = True
                return True
        self._candidates[key] = False
        return False

    def _summary(self, expression):
        # Iterative postorder handles deep expressions without Python recursion.
        pending = [(expression, None)]
        while pending:
            node, children = pending.pop()
            if node in self._summaries:
                continue
            if z3.is_quantifier(node):
                self._summaries[node] = _Summary()
                continue
            if children is None:
                cell = self._cell(node)
                if (
                    cell is not None
                    and z3.is_const(cell[0])
                    and cell[0].decl().kind() == z3.Z3_OP_UNINTERPRETED
                ):
                    # A literal cell of a plain array symbol has no eligible
                    # reads beneath it. Keep its original cell separately, but
                    # avoid summaries for the Select and its literals.
                    # Store/If/function array terms may contain symbolic reads
                    # and must follow the ordinary descendant traversal.
                    self._summaries[node] = _Summary()
                    continue
                children = self._node_children(node)
                if children:
                    pending.append((node, children))
                    pending.extend((child, None) for child in children)
                    continue
            summaries = [self._summaries[child] for child in children]
            # Match the standalone helper's reverse-child DFS read ordering.
            own_read = (
                (node,)
                if z3.is_select(node)
                and len(children) == 2
                and not z3.is_int_value(children[1])
                else ()
            )
            reads = tuple(
                dict.fromkeys(
                    (*own_read, *(r for s in reversed(summaries) for r in s.reads))
                )
            )
            self._summaries[node] = _Summary(reads)
        return self._summaries[expression]

    def _fact_summary(self, expression):
        """Group complete original facts by array in first-occurrence order."""
        pending = [(expression, None)]
        while pending:
            node, children = pending.pop()
            if node in self._fact_summaries:
                continue
            if not z3.is_and(node):
                cell = self._cell(node)
                self._fact_summaries[node] = (
                    ((cell[0], _CellFacts(cell=cell)),) if cell is not None else ()
                )
                continue
            if children is None:
                children = self._node_children(node)
                pending.append((node, children))
                pending.extend((child, None) for child in children)
                continue
            by_array = {}
            for child in children:
                for array, facts in self._fact_summaries[child]:
                    by_array.setdefault(array, []).append(facts)
            facts = []
            for array, array_roots in by_array.items():
                roots = tuple(dict.fromkeys(array_roots))
                if len(roots) == 1:
                    root = roots[0]
                else:
                    if roots not in self._fact_joins:
                        self._fact_joins[roots] = _CellFacts(roots)
                    root = self._fact_joins[roots]
                facts.append((array, root))
            self._fact_summaries[node] = tuple(facts)
        return self._fact_summaries[expression]

    @staticmethod
    def _certify(roots):
        tables, inconsistent = {}, set()
        pending, seen = list(reversed(roots)), set()
        while pending:
            facts = pending.pop()
            if facts in seen:
                continue
            seen.add(facts)
            if facts.cell is None:
                pending.extend(reversed(facts.children))
                continue
            array, index, number = facts.cell
            index, number = index.as_long(), number.as_long()
            table = tables.setdefault(array, {})
            if index in table and table[index] != number:
                inconsistent.add(array)
            table[index] = number
        certificates = {}
        for array, table in tables.items():
            lower, upper = min(table), max(table)
            if array in inconsistent or len(table) != upper - lower + 1:
                continue
            values = tuple(table.values())
            certificates[array] = (
                lower,
                upper,
                min(values),
                max(values),
                len(set(values)) == len(table),
            )
        return certificates

    def lemmas(self, conditions) -> tuple[z3.BoolRef, ...]:
        conditions = tuple(c for c in conditions if not isinstance(c, bool))
        if not conditions or not all(isinstance(c, z3.BoolRef) for c in conditions):
            return ()
        context = conditions[0].ctx
        if any(c.ctx != context for c in conditions):
            raise z3.Z3Exception("snapshot conditions have different contexts")
        if self._context != context:
            # A solver normally uses one context. A reused helper may switch,
            # but equal-looking ASTs from different contexts never share facts.
            self._children.clear()
            self._cells.clear()
            self._candidates.clear()
            self._summaries.clear()
            self._fact_summaries.clear()
            self._fact_joins.clear()
            self._certificates.clear()
            self._guarded_reads.clear()
            self._injectivity.clear()
            self._context = context
        if not any(self._has_facts(condition) for condition in conditions):
            return ()
        summaries = [self._summary(condition) for condition in conditions]
        actual_reads = dict.fromkeys(r for s in reversed(summaries) for r in s.reads)
        if not actual_reads:
            # Certificates cannot produce a lemma without a nonliteral read.
            # Cached read summaries allow this exit before assembling facts.
            return ()
        relevant_arrays = frozenset(self._node_children(r)[0] for r in actual_reads)
        if not any(
            self._has_facts(condition, relevant_arrays) for condition in conditions
        ):
            return ()
        by_array: dict[z3.ArrayRef, list[_CellFacts]] = {}
        for condition in conditions:
            for array, facts in self._fact_summary(condition):
                if array in relevant_arrays:
                    by_array.setdefault(array, []).append(facts)
        certificates = {}
        for array_roots in by_array.values():
            # Each key contains every current asserted cell for this array,
            # including conflicting or incomplete facts. Neither read changes
            # nor unrelated arrays can turn stale premises into a certificate.
            roots = tuple(dict.fromkeys(array_roots))
            if roots not in self._certificates:
                self._certificates[roots] = self._certify(roots)
            certificates.update(self._certificates[roots])
        if not certificates:
            return ()
        reads: dict[z3.ArrayRef, list[z3.ArithRef]] = {
            array: [] for array in certificates
        }
        for read in actual_reads:
            array = self._node_children(read)[0]
            if array in certificates:
                reads[array].append(read)
        lemmas = []
        for array, table_reads in reads.items():
            certificate = certificates[array]
            lower, upper, minimum, maximum, injective = certificate
            guarded = []
            for read in table_reads:
                key = read, certificate
                if key not in self._guarded_reads:
                    index = self._node_children(read)[1]
                    guard = z3.And(index >= lower, index <= upper)
                    lemma = z3.Implies(guard, z3.And(read >= minimum, read <= maximum))
                    self._guarded_reads[key] = guard, lemma
                guard, lemma = self._guarded_reads[key]
                guarded.append((read, guard))
                lemmas.append(lemma)
            if injective and len(guarded) <= _MAX_INJECTIVITY_READS:
                for (read_a, guard_a), (read_b, guard_b) in combinations(guarded, 2):
                    pair_key = read_a, read_b, certificate
                    if pair_key not in self._injectivity:
                        self._injectivity[pair_key] = z3.Implies(
                            z3.And(guard_a, guard_b, read_a == read_b),
                            self._node_children(read_a)[1]
                            == self._node_children(read_b)[1],
                        )
                    lemmas.append(self._injectivity[pair_key])
        return tuple(lemmas)


def _conjuncts(expression):
    pending = [expression]
    while pending:
        node = pending.pop()
        if z3.is_and(node):
            pending.extend(reversed(node.children()))
        else:
            yield node


def _nodes(expression):
    pending, seen = [expression], set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        # A read with a bound index cannot be lifted out of its binder.
        if z3.is_quantifier(node):
            continue
        yield node
        pending.extend(node.children())


def snapshot_table_lemmas(expression) -> tuple[z3.BoolRef, ...]:
    """Return independently entailed lemmas, without solving or new premises.

    For each original table, completeness is checked by comparing the number
    of distinct keys to the inclusive interval's size. Every recorded value
    is checked when deriving the range; injectivity additionally requires all
    those values to differ. Conflicting equations and holes decline the table.
    Duplicate values retain only the valid range facts.
    """
    if not isinstance(expression, z3.BoolRef):
        return ()
    tables: dict[z3.ArrayRef, dict[int, int]] = {}
    inconsistent: set[z3.ArrayRef] = set()
    for atom in _conjuncts(expression):
        if not z3.is_eq(atom):
            continue
        for read, value in (atom.children(), tuple(reversed(atom.children()))):
            if not (
                z3.is_select(read)
                and read.num_args() == 2
                and z3.is_int_value(read.arg(1))
                and z3.is_int_value(value)
            ):
                continue
            array = read.arg(0)
            if (
                array.sort().domain().kind() != z3.Z3_INT_SORT
                or array.sort().range().kind() != z3.Z3_INT_SORT
            ):
                continue
            index, number = read.arg(1).as_long(), value.as_long()
            table = tables.setdefault(array, {})
            if index in table and table[index] != number:
                inconsistent.add(array)
            table[index] = number

    certificates = {}
    for array, table in tables.items():
        lower, upper = min(table), max(table)
        if array in inconsistent or len(table) != upper - lower + 1:
            continue
        values = tuple(table.values())
        certificates[array] = (
            lower,
            upper,
            min(values),
            max(values),
            len(set(values)) == len(table),
        )
    if not certificates:
        return ()

    reads: dict[z3.ArrayRef, list[z3.ArithRef]] = {array: [] for array in certificates}
    for node in _nodes(expression):
        if (
            z3.is_select(node)
            and node.num_args() == 2
            and node.arg(0) in certificates
            and not z3.is_int_value(node.arg(1))
        ):
            reads[node.arg(0)].append(node)

    lemmas: list[z3.BoolRef] = []
    for array, actual_reads in reads.items():
        lower, upper, minimum, maximum, injective = certificates[array]
        guarded = [
            (read, z3.And(read.arg(1) >= lower, read.arg(1) <= upper))
            for read in actual_reads
        ]
        lemmas.extend(
            z3.Implies(guard, z3.And(read >= minimum, read <= maximum))
            for read, guard in guarded
        )
        if injective and len(guarded) <= _MAX_INJECTIVITY_READS:
            lemmas.extend(
                z3.Implies(
                    z3.And(guard_a, guard_b, read_a == read_b),
                    read_a.arg(1) == read_b.arg(1),
                )
                for (read_a, guard_a), (read_b, guard_b) in combinations(guarded, 2)
            )
    return tuple(lemmas)
