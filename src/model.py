from torch import nn

class MLP(nn.Module):
    def __init__(self, hidden_units=128):
        super().__init__()

        if type(hidden_units) is not int or hidden_units <= 0:
            raise ValueError("hidden_units must be a positive integer")

        self.hidden_units = hidden_units
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(28 * 28, hidden_units)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_units, 10)

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, 1, 28, 28], "
                f"received {tuple(images.shape)}"
            )
        if tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError(
                "Expected each image to have shape [1, 28, 28], "
                f"received {tuple(images.shape[1:])}"
            )

        x = self.flatten(images)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits