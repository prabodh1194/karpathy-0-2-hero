import random
from operator import mul
from typing import Sequence

from micrograd import Value


class Neuron:
    def __init__(self, nin: int) -> None:  # nin = number of inputs to the neuron
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: Sequence[Value | float]) -> Value:
        act = sum(map(mul, self.w, x), self.b)
        out = act.tanh()
        return out

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(
        self, nin: int, nout: int
    ) -> None:  # number of neurons making up a layer; nout is the number of neurons in the layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: Sequence[Value | float]) -> Value | list[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:  # multi layer perceptron
    def __init__(
        self, nin: int, nouts: list[int]
    ):  # nouts is sizes of all the layers in the MLP
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: Sequence[Value | float]) -> Value | list[Value]:
        out: Sequence[Value | float] | Value | list[Value] = x
        for layer in self.layers:
            out = layer(out)  # type: ignore[arg-type]
        return out  # type: ignore[return-value]

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0
