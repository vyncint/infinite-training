"""Train an MNIST classifier in PyTorch until it reaches 98% test accuracy.

The PyTorch counterpart of ``examples/mnist.py``. PyTorch has no ``fit``, so the
training step is written out and handed to the trainer, which calls it once per
round and takes care of the target, the timeout and the checkpointing.

Run it with::

    pip install "infinite_training[torch]" torchvision
    python examples/mnist_torch.py

Stop it at any point with ``Ctrl+C``: the best and most recent weights are
written to ``models_torch/`` and picked up automatically the next time you run
it.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from infinite_training import Target, TorchTrainer

CHECKPOINT_DIR = "models_torch"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_loaders() -> tuple[DataLoader, DataLoader]:
    """MNIST train and test loaders, normalised to the usual mean and stdev."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST("data", train=True, download=True, transform=transform)
    test = datasets.MNIST("data", train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(test, batch_size=1000),
    )


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def main() -> None:
    train_loader, test_loader = build_loaders()

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def step() -> dict[str, float]:
        """One epoch of training followed by an evaluation pass.

        The returned mapping is what the trainer watches, so anything you want
        to target has to appear here.
        """
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                correct += (model(images).argmax(dim=1) == labels).sum().item()

        metrics = {
            "loss": running_loss / len(train_loader.dataset),
            "test_accuracy": correct / len(test_loader.dataset),
        }
        print(f"loss={metrics['loss']:.4f} test_accuracy={metrics['test_accuracy']:.4f}")
        return metrics

    trainer = TorchTrainer(
        model=model,
        target=Target("test_accuracy", smaller_is_better=False, target_value=0.98),
        best_weights_path=f"{CHECKPOINT_DIR}/best_weights.npy",
        last_weights_path=f"{CHECKPOINT_DIR}/last_weights.npy",
        best_value_path=f"{CHECKPOINT_DIR}/best_value.npy",
        value_history_path=f"{CHECKPOINT_DIR}/value_history.npy",
    )

    print(f"Training on {DEVICE}; resuming from round {trainer.rounds_completed}.")
    trainer.train(step)

    print(f"Best test accuracy: {trainer.best_value:.4f}")
    print(f"Rounds completed:   {trainer.rounds_completed}")


if __name__ == "__main__":
    main()
