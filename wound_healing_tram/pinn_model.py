import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from logging_utils import setup_logger

logger = setup_logger()

LAYER_SIZE = 50
NUM_LAYERS = 3
INPUT_DIM = 3  # (X_norm, Y_norm, T_norm)
OUTPUT_DIM = 1 # (C_norm)
USE_FOURIER_FEATURES = False
NUM_FOURIER_FEATURES = 256 # Number of cosine/sine pairs (256 * 2 = 512 total features)
FOURIER_SCALE = 10.0 # Standard deviation for the B matrix

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple MPS for acceleration.")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA for acceleration.")
    else:
        device = torch.device('cpu')
        logger.info("No GPU available. Using CPU for training.")
    return device

DEVICE = get_device()

class FourierFeatureLayer(nn.Module):
    """
    Feature Embedding/ Input Mapping
    maps (x,y,t) to a higher dimensional space using random Fourier features to help the network learn
    high frequency components.
    """

    def __init__(self, input_dim: int, num_features: int = NUM_FOURIER_FEATURES, scale: float = FOURIER_SCALE):
        super(FourierFeatureLayer, self).__init__()

        # B is a fixed, non-trainable matrix of size (input_dim, num_features)
        self.register_buffer('B', scale * torch.randn(input_dim, num_features))
        self.output_dim = num_features * 2

        logger.info(f"  -> Initialized Fourier Features: {input_dim} -> {self.output_dim} features.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps x using sine and cosine functions.
        The full output includes the original input: [x, cos(2pi B x), sin(2pi B x)]
        """
        # Calculate the projection: x @ B
        x_proj = 2 * np.pi * x @ self.B

        # Concatenate original input with sine and cosine mappings
        return torch.cat([x, torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLP(nn.Module):
    """
    The core approximator network using Tanh activation for smooth derivatives.
    """

    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, layer_size=LAYER_SIZE, num_layers=NUM_LAYERS,
                 use_ffe=USE_FOURIER_FEATURES):
        super(MLP, self).__init__()

        # --- Feature Embedding ---
        self.ffe_layer = None
        if use_ffe:
            self.ffe_layer = FourierFeatureLayer(input_dim, num_features=NUM_FOURIER_FEATURES)
            # Update the input dimension for the first linear layer
            input_dim = input_dim + self.ffe_layer.output_dim

        # --- MLP Backbone Construction ---
        # The input_dim is now potentially much larger due to Fourier Features
        layers = [nn.Linear(input_dim, layer_size)]

        # Hidden layers with Tanh activation
        for _ in range(num_layers - 1):
            layers.append(nn.Tanh())
            layers.append(nn.Linear(layer_size, layer_size))

        # Final layer (no Tanh on final output, but one before)
        layers.append(nn.Tanh())
        layers.append(nn.Linear(layer_size, output_dim))

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier uniform initialization for better convergence
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, x):
        # Apply Fourier Feature Embedding first if enabled
        if self.ffe_layer:
            x = self.ffe_layer(x)
        return self.net(x)


class WoundHealingPINN(nn.Module):
    def __init__(self, use_ffe: bool = USE_FOURIER_FEATURES):
        super(WoundHealingPINN, self).__init__()

        # Core Neural Network
        self.model = MLP(use_ffe=use_ffe)

        # Discovered Physics Parameters (D and rho)
        # These are trainable parameters. Log-space ensures D > 0 and rho > 0.
        self.log_D = nn.Parameter(torch.tensor([-3.0], dtype=torch.float32))  # Initial guess for D
        self.log_rho = nn.Parameter(torch.tensor([-1.0], dtype=torch.float32))  # Initial guess for rho

        logger.info(f" MLP Backbone created: {NUM_LAYERS} layers, {LAYER_SIZE} neurons.")
        logger.info(" Trainable parameters D (Diffusion) and rho (Proliferation) initialized.")

    def forward(self, xyt_normalized: torch.Tensor) -> torch.Tensor:
        """
        Calculates the predicted cell density C_hat from the normalized coordinates.
        """
        return self.model(xyt_normalized)

    @property
    def pde_params(self):
        """Returns the physical parameters D and rho, ensuring they are positive."""
        D = torch.exp(self.log_D)
        rho = torch.exp(self.log_rho)
        return D, rho

    def pde_residual(self, xyt_normalized: torch.Tensor) -> torch.Tensor:
        """
        Placeholder function. Autograd for the PDE residual is handled in the PINNLoss
        class where Jacobian scaling is applied.
        """
        # Ensure input requires gradient for autograd
        xyt_normalized.requires_grad_(True)
        C_hat = self.forward(xyt_normalized)
        return C_hat