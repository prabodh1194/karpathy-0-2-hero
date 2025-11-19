from __future__ import annotations
from typing import Self
from graphviz import Digraph
import math


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


def draw_root(root: Value, render: bool = False) -> Digraph:
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label=f"{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape="record",
        )
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    if render:
        dot.render("graph", view=True, cleanup=True)

    return dot


def build_topo(root: Value) -> list[Value]:
    visited = set()
    ans: list[Value] = []

    def _topo(v: Value) -> None:
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                _topo(child)
            ans.append(v)

    _topo(root)
    return ans


class Value:
    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other: Self) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: Self) -> Value:
        return self + other

    def __mul__(self, _other: Self | int | float) -> Value:
        other = _other if isinstance(_other, Value) else Value(_other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __rmul__(self, other: Self) -> Value:
        return self * other

    def __truediv__(self, other: Self) -> Value:
        return self * other**-1

    def __pow__(self, other: int | float) -> Value:
        assert isinstance(other, (int, float)), "power must be raised to a int or float"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            n = other
            self.grad += n * self.data ** (n - 1) * out.grad

        out._backward = _backward

        return out

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value) -> Value:
        return self + (-other)

    def tanh(self) -> Value:
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def exp(self) -> Value:
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward() -> None:
            self.grad = out.data * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:
        self.grad = 1.0
        topo = build_topo(self)

        for n in reversed(topo):
            n._backward()


def lol() -> None:
    h = 0.001

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

    L1 = L.data

    L.grad = 1.0
    d.grad = -2.0
    f.grad = 4.0
    c.grad = 1.0 * -2.0  # chain rule
    e.grad = 1.0 * -2.0  # chain rule
    a.grad = -2.0 * -3.0  # chain rule
    b.grad = -2.0 * 2.0  # chain rule
    draw_root(L)

    a.data += a.grad * 0.01
    b.data += b.grad * 0.01
    c.data += c.grad * 0.01
    f.data += f.grad * 0.01
    L = (a * b + c) * f
    print(L)

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
    L2 = L.data

    print(L1, L2, (L2 - L1) / h)


if __name__ == "__main__":
    lol()
