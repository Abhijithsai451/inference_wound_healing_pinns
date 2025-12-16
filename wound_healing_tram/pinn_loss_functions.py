from typing import Tuple

import torch
import torch.nn as nn
import torch.autograd as autograd
from sklearn.preprocessing import MinMaxScaler
from pinn_model import WoundHealingPINN, DEVICE
from logging_utils import setup_logger

logger = setup_logger()


def compute_physical_residual(model: WoundHealingPINN, X_collocation: torch.Tensor,
                              dC_scale: float, dX_scale: float, dY_scale: float, dT_scale: float) -> torch.Tensor:
    """
    Computes the physical PDE residual using autograd and Jacobian scaling.
    f = dC/dt - D * (d^2C/dx^2 + d^2C/dy^2) - rho * C * (1 - C)
    """
    xyt_normalized = X_collocation.clone()
    xyt_normalized.requires_grad_(True)
    C_hat = model(xyt_normalized)

    # 1. First temporal derivative (dC/dt)
    C_t = autograd.grad(C_hat, xyt_normalized, torch.ones_like(C_hat), create_graph=True, allow_unused=True)[0][:, 2]

    # 2. Second spatial derivatives (d^2C/dx^2 and d^2C/dy^2)
    C_x_norm = autograd.grad(C_hat, xyt_normalized, torch.ones_like(C_hat), create_graph=True, allow_unused=True)[0][:,
               0].view(-1, 1)
    C_xx_norm = \
    autograd.grad(C_x_norm, xyt_normalized, torch.ones_like(C_x_norm), create_graph=True, allow_unused=True)[0][:, 0]

    C_y_norm = autograd.grad(C_hat, xyt_normalized, torch.ones_like(C_hat), create_graph=True, allow_unused=True)[0][:,
               1].view(-1, 1)
    C_yy_norm = \
    autograd.grad(C_y_norm, xyt_normalized, torch.ones_like(C_y_norm), create_graph=True, allow_unused=True)[0][:, 1]

    # Get discovered parameters
    D, rho = model.discovered_params

    # --- Apply Jacobian Scaling ---
    C_t_phys = C_t * (dC_scale / dT_scale)
    C_xx_phys = C_xx_norm * (dC_scale / (dX_scale ** 2))
    C_yy_phys = C_yy_norm * (dC_scale / (dY_scale ** 2))

    # Calculate physical residual
    f_phys = (C_t_phys -
              D * (C_xx_phys + C_yy_phys) -
              rho * C_hat.squeeze() * (1 - C_hat.squeeze()))

    return f_phys
class PINNLoss:
    """
    Manages the calculation of all loss components (Data, Physics, BC)  and the necessary Jacobian scaling for the PDE.
    """
    def __init__(self, model: WoundHealingPINN, scaler: MinMaxScaler,
                 lambda_data: float = 1.0, lambda_phy: float = 1.0, lambda_bc: float = 1.0):
        """
        Initializes the loss calculator with the model, the data scaler,
        and the weight multipliers (lambdas).
        """
        self.model = model

        # Loss component weights
        self.lambda_data = lambda_data
        self.lambda_phy = lambda_phy
        self.lambda_bc = lambda_bc

        # --- Jacobian Scaling Factors (Step 9) ---
        # The scaler's 'data_max_' and 'data_min_' hold the max/min of [X, Y, T, C]

        self.dX_scale = scaler.data_max_[0] - scaler.data_min_[0]
        self.dY_scale = scaler.data_max_[1] - scaler.data_min_[1]
        self.dT_scale = scaler.data_max_[2] - scaler.data_min_[2]
        self.dC_scale = scaler.data_max_[3] - scaler.data_min_[3]

        logger.info(f"Jacobian Scaling Factors computed:")
        logger.info(f"dX_scale: {self.dX_scale:.2f}, dY_scale: {self.dY_scale:.2f}, dT_scale: {self.dT_scale:.2f}, dC_scale: {self.dC_scale:.2f}")

    def _data_loss(self, X_data, C_data) -> torch.Tensor:
        """
        Calculates the Mean Squared Error (MSE) between PINN prediction and observed data.
        """
        C_hat = self.model(X_data)
        L_data = nn.MSELoss()(C_hat, C_data)
        return L_data

    def _physics_loss(self, X_collocation) -> torch.Tensor:
        """
        Calculates the Mean Squared Error (MSE) of the physical PDE residual (f_phys) at collocation points.
        """
        f_phys = compute_physical_residual(self.model, X_collocation,
                                           self.dC_scale, self.dX_scale, self.dY_scale, self.dT_scale)
        L_phy = nn.MSELoss()(f_phys, torch.zeros_like(f_phys))
        return L_phy

    def _boundary_loss(self, X_init: torch.Tensor, X_spatial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the boundary condition and initial condition loss.
        L_bc = L_init + L_neumann
        """
        c_hat_ic = self.model(X_init)
        c_target_ic = torch.zeros_like(c_hat_ic).to(DEVICE)
        L_ic = nn.MSELoss()(c_hat_ic, c_target_ic)

        X_spatial.requires_grad_(True)
        c_hat_bc = self.model(X_spatial)

        gradients = autograd.grad(c_hat_bc, X_spatial, torch.ones_like(c_hat_bc), create_graph=True, allow_unused=True)[0]
        # X-boundaries
        x_indices = torch.isclose(X_spatial[:, 0], torch.tensor(0.0).to(DEVICE))
        dc_dx_norm = gradients[x_indices, 0]

        # Y-boundaries
        y_indices = torch.isclose(X_spatial[:, 1], torch.tensor(0.0).to(DEVICE))
        dc_dy_norm = gradients[y_indices, 1]

        # target for all the spatial derivatives is zero (zero flux)
        target_zero = torch.zeros_like(dc_dx_norm).to(DEVICE)
        L_neumann_x = nn.MSELoss()(dc_dx_norm, target_zero)

        target_zero = torch.zeros_like(dc_dy_norm).to(DEVICE)
        L_neumann_y = nn.MSELoss()(dc_dy_norm, target_zero)
        L_neumann = L_neumann_x + L_neumann_y

        # Total Boundary Loss
        L_bc = L_ic + L_neumann


        return L_bc, L_ic, L_neumann

    def calculate_total_loss(self, X_data, C_data, X_collocation, X_initial, X_spatial):
        """
        Calculates the total weighted loss: L_Total = L_Data + L_Phy + L_BC.
        """
        L_data = self._data_loss(X_data, C_data)
        L_phy = self._physics_loss(X_collocation)
        L_bc, L_ic, L_neumann = self._boundary_loss(X_initial, X_spatial)

        L_total = self.lambda_data * L_data + self.lambda_phy * L_phy + self.lambda_bc * L_bc

        return L_total, L_data, L_phy, L_bc, L_ic, L_neumann