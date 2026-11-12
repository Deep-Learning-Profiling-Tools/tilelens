"""Ordered overlapping interval index for exact NKI dependency histories.

Records preserve insertion/split order, matching a linear dependency scan.
Non-finite NaNs fall back to the original list semantics for that storage.
"""
from __future__ import annotations

import math
import random


class _Node:
    __slots__=('entry','rank','uid','priority','left','right','max_hi')
    def __init__(self,entry,rank,uid,priority):
        self.entry=entry;self.rank=rank;self.uid=uid;self.priority=priority
        self.left=None;self.right=None;self.max_hi=entry[1]
    @property
    def key(self):return (self.entry[0],self.uid)


def _update(node):
    if node is not None:
        node.max_hi=max(node.entry[1],node.left.max_hi if node.left else -math.inf,
                        node.right.max_hi if node.right else -math.inf)
    return node


def _rotate_right(node):
    child=node.left;node.left=child.right;child.right=_update(node)
    return _update(child)


def _rotate_left(node):
    child=node.right;node.right=child.left;child.left=_update(node)
    return _update(child)


def _insert(root,node):
    if root is None:return node
    if node.key<root.key:
        root.left=_insert(root.left,node)
        if root.left.priority<root.priority:root=_rotate_right(root)
    else:
        root.right=_insert(root.right,node)
        if root.right.priority<root.priority:root=_rotate_left(root)
    return _update(root)


def _merge(left,right):
    if left is None:return right
    if right is None:return left
    if left.priority<right.priority:
        left.right=_merge(left.right,right)
        return _update(left)
    right.left=_merge(left,right.left)
    return _update(right)


def _delete(root,key):
    if root is None:raise KeyError(key)
    if key==root.key:return _merge(root.left,root.right)
    if key<root.key:root.left=_delete(root.left,key)
    else:root.right=_delete(root.right,key)
    return _update(root)


def _nan(value):
    try:return math.isnan(value)
    except (TypeError,ValueError):return True


class IntervalHistory:
    """Append records, query intersections, and subtract/clear overwritten bytes."""
    def __init__(self):
        self._root=None;self._rng=random.Random(0);self._uid=0;self._serial=0;self._slow=None

    def _add(self,entry,rank):
        node=_Node(entry,rank,self._uid,self._rng.getrandbits(64));self._uid+=1
        self._root=_insert(self._root,node)

    def _all_nodes(self):
        stack=[self._root] if self._root else [];out=[]
        while stack:
            n=stack.pop();out.append(n)
            if n.left:stack.append(n.left)
            if n.right:stack.append(n.right)
        return out

    def _fallback(self):
        if self._slow is None:
            self._slow=sorted(((n.rank,n.entry) for n in self._all_nodes()),key=lambda x:x[0])
            self._root=None

    def append(self,entry):
        rank=(self._serial,);self._serial+=1
        if self._slow is not None or _nan(entry[0]) or _nan(entry[1]) or entry[0] >= entry[1]:
            self._fallback();self._slow.append((rank,entry));return
        self._add(entry,rank)

    def _nodes_overlapping(self,lo,hi):
        stack=[self._root] if self._root else [];out=[]
        while stack:
            node=stack.pop()
            if node.left and node.left.max_hi>lo:stack.append(node.left)
            if lo<node.entry[1] and node.entry[0]<hi:out.append(node)
            if node.right and node.entry[0]<hi and node.right.max_hi>lo:stack.append(node.right)
        out.sort(key=lambda n:n.rank)
        return out

    def overlaps(self,lo,hi):
        if self._slow is not None or _nan(lo) or _nan(hi) or lo >= hi:
            self._fallback()
            return [entry for _,entry in self._slow if lo<entry[1] and entry[0]<hi]
        return [n.entry for n in self._nodes_overlapping(lo,hi)]

    def subtract(self,lo,hi):
        """Split entries around overwritten bytes, preserving original order."""
        if self._slow is not None or _nan(lo) or _nan(hi) or lo >= hi:
            self._fallback();updated=[]
            for rank,entry in self._slow:
                a,b,*rest=entry
                if not (lo<b and a<hi):updated.append((rank,entry));continue
                fragments=[]
                if a<lo:fragments.append((a,min(b,lo),*rest))
                if hi<b:fragments.append((max(a,hi),b,*rest))
                for i,fragment in enumerate(fragments):updated.append((rank+(i,) if len(fragments)>1 else rank,fragment))
            self._slow=updated;return
        for n in self._nodes_overlapping(lo,hi):
            self._root=_delete(self._root,n.key);a,b,*rest=n.entry;fragments=[]
            if a<lo:fragments.append((a,min(b,lo),*rest))
            if hi<b:fragments.append((max(a,hi),b,*rest))
            for i,fragment in enumerate(fragments):self._add(fragment,n.rank+(i,) if len(fragments)>1 else n.rank)

    def discard_contained(self,lo,hi):
        """Discard only records completely covered by a later write."""
        if self._slow is not None or _nan(lo) or _nan(hi) or lo >= hi:
            self._fallback();self._slow=[(r,e) for r,e in self._slow if not (lo<=e[0] and e[1]<=hi)];return
        for n in self._nodes_overlapping(lo,hi):
            if lo<=n.entry[0] and n.entry[1]<=hi:self._root=_delete(self._root,n.key)

    def pop_overlaps(self, lo, hi):
        """Remove and return overlapping records, in their original order."""
        if self._slow is not None or _nan(lo) or _nan(hi) or lo >= hi:
            self._fallback()
            selected = []
            remaining = []
            for rank, entry in self._slow:
                (selected if lo < entry[1] and entry[0] < hi else remaining).append((rank, entry))
            self._slow = remaining
            return [entry for _, entry in selected]
        nodes = self._nodes_overlapping(lo, hi)
        for node in nodes:
            self._root = _delete(self._root, node.key)
        return [node.entry for node in nodes]

    def __bool__(self):
        return bool(self._slow) if self._slow is not None else self._root is not None

    def pop_all(self):
        """Remove and return every remaining record in original order."""
        self._fallback()
        entries = [entry for _, entry in self._slow]
        self._slow = []
        return entries
