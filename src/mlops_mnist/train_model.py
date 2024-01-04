import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from models.model import MyNeuralNet


def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model = MyNeuralNet()
    train_set = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train_loader.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Train model and plot training loss progress
    epochs = 10
    steps = 0
    train_losses = []

    for e in tqdm(range(epochs)):
        running_loss = 0
        for images, labels in train_set:
            steps += 1

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_losses.append(running_loss / len(train_set))

    # Save plot of training loss in current folder
    plot_path = os.path.join(os.path.dirname(__file__), "..", "reports", "figures", "training_loss.png")
    plt.plot(train_losses, label="Training loss")
    plt.legend(frameon=False)
    plt.savefig(plot_path)

    # Save model
    torch.save(model, os.path.join(os.path.dirname(__file__), "..", "models", "model.pt"))


if __name__ == "__main__":
    train(1e-3)
