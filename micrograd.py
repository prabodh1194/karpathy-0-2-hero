from __future__ import annotations
from typing import Self
from graphviz import Digraph


def trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    # builds a set of all nodes & edges in a graph
    nodes: set[Value] = set()
    edges: set[tuple[Value, Value]] = set()

    def build(v: Value) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_root(root: Value) -> None:
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label=f"{{ {n.label} | data {n.data:.4f} }}", shape="record")
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    dot.render("graph", view=True, cleanup=True)


class Value:
    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: Self) -> Value:
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other: Self) -> Value:
        return Value(self.data * other.data, (self, other), "*")


if __name__ == "__main__":
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"

    f = Value(-2.0, label="f")

    L = d * f
    L.label = "L"

    print(d, d._prev, d._op)
    draw_root(L)
