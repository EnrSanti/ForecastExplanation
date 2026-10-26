"""
MDL pruning and threshold snapping of the rules of one binary FOLD-RM model,
on its training rows.

A rule is held as a node [head, items, exceptions]: items are CUDatILP
literals (column, op, value), exceptions are nodes. A row is covered by a
node when all its items hold and none of its exceptions covers it, as in
CUDatILP's evaluate.
"""

import math

import numpy as np


class Data:
    """Column-wise view of the rows, with cached literal masks. Empty
    numeric cells ('') fail every comparison, as in CUDatILP."""

    def __init__(self, rows: list[list]):
        self.n = len(rows)
        columns = list(zip(*rows))
        self.raw = columns[:-1]
        self.y = list(columns[-1])
        self.num = []
        for c in self.raw:
            if any(isinstance(v, (int, float)) for v in c):
                self.num.append(
                    np.array([np.nan if isinstance(v, str) else float(v) for v in c])
                )
            else:
                self.num.append(None)
        self.cache = {}
        self.zeros = np.zeros(self.n, dtype=bool)

    def lit(self, item: tuple) -> np.ndarray:
        m = self.cache.get(item)
        if m is not None:
            return m
        j, op, v = item
        if isinstance(v, str):
            if op == "==":
                m = np.array([x == v for x in self.raw[j]], dtype=bool)
            elif op == "!=":
                m = np.array([x != v for x in self.raw[j]], dtype=bool)
            else:
                m = self.zeros
        else:
            a = self.num[j]
            if a is None:
                m = self.zeros
            elif op == "<=":
                m = a <= v
            elif op == ">":
                m = a > v
            else:
                m = self.zeros
        self.cache[item] = m
        return m


def to_node(rule) -> list:
    head, items, ab, _ = rule
    return [head, list(items), [to_node(e) for e in ab]]


def to_rule(node: list) -> tuple:
    head, items, ab = node
    return head, list(items), [to_rule(e) for e in ab], 0


def _mask(node: list, data: Data) -> np.ndarray:
    m = np.ones(data.n, dtype=bool)
    for it in node[1]:
        m &= data.lit(it)
    for e in node[2]:
        if not m.any():
            break
        m &= ~_mask(e, data)
    return m


def _model_mask(nodes: list, data: Data) -> np.ndarray:
    m = data.zeros.copy()
    for r in nodes:
        m |= _mask(r, data)
    return m


def _conditions(node: list) -> int:
    """Feature atoms of the node and its exceptions (two bounds on one
    column are one atom)."""
    return len({it[0] for it in node[1]}) + sum(_conditions(e) for e in node[2])


def _get(node: list, path: tuple) -> list:
    for k in path:
        node = node[2][k]
    return node


def _clone(node: list) -> list:
    return [node[0], list(node[1]), [_clone(e) for e in node[2]]]


def _paths(node: list, path: tuple = ()):
    yield path
    for k, e in enumerate(node[2]):
        yield from _paths(e, path + (k,))


def _moves(nodes: list) -> list[tuple]:
    """Candidate deletions: (kind, rule index, path, arg)."""
    out = []
    for r, root in enumerate(nodes):
        out.append(("rule", r, (), None))
        for p in _paths(root):
            n = _get(root, p)
            if p:
                out.append(("exc", r, p, None))
            cols = {}
            for it in n[1]:
                cols.setdefault(it[0], []).append(it)
            if len(cols) > 1:
                for c in cols:
                    out.append(("atom", r, p, c))
            for c, its in cols.items():
                if len(its) > 1:
                    for it in its:
                        out.append(("item", r, p, it))
    return out


def _apply(root: list, kind: str, path: tuple, arg) -> list:
    root = _clone(root)
    if kind == "exc":
        parent = _get(root, path[:-1])
        del parent[2][path[-1]]
    elif kind == "atom":
        n = _get(root, path)
        n[1] = [it for it in n[1] if it[0] != arg]
    elif kind == "item":
        n = _get(root, path)
        n[1] = [it for it in n[1] if it != arg]
    return root


def _errors(mask: np.ndarray, y: np.ndarray) -> float:
    return float(np.count_nonzero(mask != y))


def mdl_prune(nodes: list, data: Data, y: np.ndarray, lam: float) -> list:
    """Best-improvement deletion of rules, exceptions, atoms and interval
    bounds while J = training errors + lam * conditions does not get worse
    (every move shrinks the program, so ties are accepted)."""
    nodes = [_clone(n) for n in nodes]
    masks = [_mask(n, data) for n in nodes]
    atoms = [_conditions(n) for n in nodes]

    def total(ms):
        m = data.zeros.copy()
        for x in ms:
            m |= x
        return m

    J = _errors(total(masks), y) + lam * sum(atoms)
    while True:
        best = None
        for kind, r, p, arg in _moves(nodes):
            if kind == "rule":
                new_root, nm, na = None, data.zeros, 0
            else:
                new_root = _apply(nodes[r], kind, p, arg)
                nm, na = _mask(new_root, data), _conditions(new_root)
            ms = masks[:r] + [nm] + masks[r + 1 :]
            a = sum(atoms) - atoms[r] + na
            j = _errors(total(ms), y) + lam * a
            if j <= J + 1e-9:
                key = (j, a)
                if best is None or key < best[0]:
                    best = (key, kind, r, new_root, nm, na)
        if best is None:
            return nodes
        (J, _), kind, r, new_root, nm, na = best
        if kind == "rule":
            del nodes[r], masks[r], atoms[r]
        else:
            nodes[r], masks[r], atoms[r] = new_root, nm, na


def snap_step(name: str) -> float:
    """Round step of a feature column, from its name (translator units)."""
    n = name.lower()
    if n.endswith("_hours") or "_hours_" in n:
        return 1.0
    if n.startswith("humidity_fronts_value"):
        return 5.0
    if "area" in n or n.startswith("size_cloud"):
        return 5000.0
    if n.startswith(
        ("humidity", "rh", "region_rh", "cloud", "coverage", "region_cloud")
    ):
        return 5.0
    if n.startswith("temp_fronts_value"):
        return 0.5
    if n.startswith(("temperature", "te850_mean", "lapse")):
        return 1.0
    if n.startswith("wind_speed"):
        return 1.0 if n.endswith(("medium", "high")) else 0.5
    if n == "month":
        return 1.0
    return 0.5


def _round(v: float, step: float) -> float:
    return round(round(v / step) * step, 6)


def snap_thresholds(
    nodes: list, data: Data, y: np.ndarray, attrs: list[str], span: int = 2
) -> list:
    """Each numeric threshold -> the multiple of its feature's step (within
    +-span steps) with the fewest training errors; ties go to the value
    closest to the original."""
    nodes = [_clone(n) for n in nodes]
    for r in range(len(nodes)):
        for p in list(_paths(nodes[r])):
            n = _get(nodes[r], p)
            for k in range(len(n[1])):
                j, op, v = n[1][k]
                if isinstance(v, str) or op not in ("<=", ">"):
                    continue
                step = snap_step(attrs[j])
                base = math.floor(v / step) * step
                cands = sorted(
                    {_round(base + i * step, step) for i in range(-span + 1, span + 1)}
                )
                best = None
                for c in cands:
                    n[1][k] = (j, op, c)
                    key = (_errors(_model_mask(nodes, data), y), abs(c - v))
                    if best is None or key < best[0]:
                        best = (key, c)
                n[1][k] = (j, op, best[1])
    return nodes
