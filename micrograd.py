from __future__ import annotations
from typing import Self
import math


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
