import time
from datetime import datetime

import numpy as np
import numpy.typing as npt

from nn import MLP
from micrograd import Value
from mnist_loader import load_mnist


def prepare_data(
    n_samples: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    X_train, y_train, X_test, y_test = load_mnist()
    X_small = X_train[:n_samples]
    y_small = y_train[:n_samples]
    X_flat = X_small.reshape(-1, 784)
    X_norm = X_flat / 255.0
    return X_norm, y_small


def compute_loss(
    model: MLP,
    X_norm: npt.NDArray[np.float64],
    y_small: npt.NDArray[np.uint8],
    n_samples: int,
) -> tuple[Value, float]:
    inputs = [list(map(Value, xrow)) for xrow in X_norm]

    predictions: list[list[Value]] = []
    for i, x in enumerate(inputs):
        t0 = time.time()
        pred = model(x)
        assert isinstance(pred, list)
        predictions.append(pred)
        t1 = time.time()
        print(f"[{i} - {datetime.now().isoformat()}] Time taken: {t1 - t0:.3f}s")

    losses: list[Value] = []
    for i in range(n_samples):
        pred = predictions[i]
        true_label = int(y_small[i])
        target = [Value(0.0) for _ in range(10)]
        target[true_label] = Value(1.0)
        sample_loss = Value(0.0)
        for j in range(10):
            sample_loss = sample_loss + (pred[j] - target[j]) ** 2
        losses.append(sample_loss)

    total_loss: Value = sum(losses, start=Value(0.0)) * Value(1.0 / n_samples)

    correct = 0.0
    for i in range(n_samples):
        predicted_digit = max(range(10), key=lambda j: predictions[i][j].data)
        if predicted_digit == y_small[i]:
            correct += 1

    return total_loss, correct / n_samples


def train(
    model: MLP,
    X_norm: npt.NDArray[np.float64],
    y_small: npt.NDArray[np.uint8],
    n_samples: int,
    learning_rate: float = 0.01,
    n_steps: int = 5,
) -> MLP:
    for k in range(n_steps):
        total_loss, accuracy = compute_loss(model, X_norm, y_small, n_samples)

        model.zero_grad()
        total_loss.backward()

        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 1 == 0:
            print(f"step {k}: loss {total_loss.data:.4f}, accuracy {accuracy:.2%}")

    return model


def test_model(
    model: MLP,
    X_norm: npt.NDArray[np.float64],
    y_small: npt.NDArray[np.uint8],
    n_samples: int,
) -> None:
    print("\nTesting trained model:\n")

    for i in range(min(5, n_samples)):
        x_test = list(map(Value, X_norm[i]))
        pred = model(x_test)
        assert isinstance(pred, list)
        predicted_digit = max(range(10), key=lambda j: pred[j].data)
        true_digit = y_small[i]
        status = "✓" if predicted_digit == true_digit else "✗"
        print(f"Sample {i}: True={true_digit}, Predicted={predicted_digit} {status}")
        print(f"  Scores: {[f'{pred[j].data:.2f}' for j in range(10)]}")
        print()


def main() -> None:
    n_samples = 10
    learning_rate = 0.01
    n_steps = 5

    X_norm, y_small = prepare_data(n_samples)
    model = MLP(784, [128, 10])

    model = train(model, X_norm, y_small, n_samples, learning_rate, n_steps)
    test_model(model, X_norm, y_small, n_samples)


if __name__ == "__main__":
    main()
