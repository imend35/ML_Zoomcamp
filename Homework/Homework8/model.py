
import torch
import torch.nn as nn


class HairModel(nn.Module):
    """A simple CNN for binary hair-type classification."""

    def __init__(self):
        super().__init__()

        # Extract local features from RGB images.
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
        )

        # Reduce the spatial dimensions by half.
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Convert extracted features into a hidden representation.
        self.hidden = nn.Linear(32 * 99 * 99, 64)

        # Produce one logit for binary classification.
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Flatten feature maps while keeping the batch dimension.
        x = torch.flatten(x, start_dim=1)

        x = torch.relu(self.hidden(x))
        x = self.output(x)

        return x


if __name__ == "__main__":
    model = HairModel()

    # Count all trainable model parameters.
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(model)
    print(f"\nTrainable parameters: {total_params:,}")

    # Check the model output using a sample batch.
    sample_input = torch.randn(2, 3, 200, 200)

    with torch.no_grad():
        sample_output = model(sample_input)

    print(f"Input shape: {tuple(sample_input.shape)}")
    print(f"Output shape: {tuple(sample_output.shape)}")
