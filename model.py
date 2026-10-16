"""
model.py — Lightweight CNN for PneumoniaMNIST binary classification.

Architecture:
  Block 1: Conv2d(1→16) → BatchNorm → ReLU → MaxPool(2×2)   [28→14]
  Block 2: Conv2d(16→32) → BatchNorm → ReLU → MaxPool(2×2)  [14→7]
  FC head: Flatten → Linear(32*7*7=1568 → 64) → ReLU → Dropout(0.3) → Linear(64→1)

Output: single raw logit (apply torch.sigmoid externally for probabilities).
Total parameters: ~200 K — suitable for CPU training.

Educational/research use only. Not for clinical diagnosis.
"""

import torch
import torch.nn as nn
import config


class PneumoniaCNN(nn.Module):
    """
    Two-block CNN for binary classification of 28×28 grayscale chest X-rays.

    Output is a single logit per image (shape [B, 1]).
    Use BCEWithLogitsLoss for training, torch.sigmoid for inference probabilities.
    """

    def __init__(self) -> None:
        super().__init__()

        # --- Convolutional blocks ---
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28×28 → 14×14
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 14×14 → 7×7
        )

        # --- Fully-connected head ---
        # After two MaxPool(2×2) on 28×28 input: feature map is 7×7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),                        # single logit for binary clf
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, 1, 28, 28)

        Returns
        -------
        logits : torch.Tensor  shape (B, 1)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


def get_model(seed: int = config.SEED, device: torch.device = config.DEVICE) -> PneumoniaCNN:
    """
    Instantiate PneumoniaCNN, seed weight initialisation, and move to device.

    Parameters
    ----------
    seed   : random seed for reproducible weight initialisation
    device : torch.device

    Returns
    -------
    model : PneumoniaCNN on the specified device
    """
    torch.manual_seed(seed)
    model = PneumoniaCNN()
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = get_model()
    n_params = count_parameters(model)

    print(model)
    print(f"\nDevice          : {config.DEVICE}")
    print(f"Trainable params: {n_params:,}")

    # Forward-pass shape check
    dummy_input = torch.zeros(8, 1, 28, 28, device=config.DEVICE)
    output = model(dummy_input)
    assert output.shape == (8, 1), f"Unexpected output shape: {output.shape}"
    print(f"Output shape    : {tuple(output.shape)}  (expected (8, 1))  [OK]")

    assert n_params < 200_000, f"Model too large: {n_params:,} params (limit 200K)"
    print(f"Parameter count : {n_params:,} < 200,000  [OK]")

    print("\nmodel.py OK")
