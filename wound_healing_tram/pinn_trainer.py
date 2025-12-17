from typing import List, Dict

import torch
from pinn_model import WoundHealingPINN
from pinn_loss_functions import PINNLoss
from logging_utils import setup_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
logger = setup_logger()
class PINNTrainer:
    """
    Manages the optimization process of the PINN model.
    """
    def __init__(self, model: WoundHealingPINN, loss: PINNLoss):
        self.model = model
        self.loss = loss

    def train_adam(self, X_data, C_data, X_collocation, X_initial, X_spatial,
                   epochs: int, lr: float) -> List[Dict]:
        """
        Implements the Adam optimizer strategy and the main training loop.
        """

        # Optimizer Strategy (Adam) - Phase 4, Step 1
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=50, verbose=True, min_lr=1e-6)

        loss_history = []

        logger.info(f"\n[ADAM TRAINING] Starting training for {epochs} epochs (LR: {lr})...")

        for epoch in range(1, epochs + 1):

            optimizer.zero_grad()

            # Calculate all loss components by calling the dedicated loss manager
            L_total, L_data, L_phy, L_bc, L_ic, L_n, L_sindy = self.loss.calculate_total_loss(
                X_data, C_data, X_collocation, X_initial, X_spatial
            )

            L_total.backward()
            optimizer.step()
            scheduler.step(L_total)
            # Logging and Checkpointing
            if epoch % 5 == 0 or epoch == 1:
                D, rho = self.model.pde_params
                log_data = {
                    'epoch': epoch,
                    'L_total': L_total.item(),
                    'L_data': L_data.item(),
                    'L_phy': L_phy.item(),
                    'L_bc': L_bc.item(),
                    'D': D.item(),
                    'rho': rho.item()
                }
                loss_history.append(log_data)

                print( f"Epoch {epoch:5d} | L_T: {L_total.item():.5f} | L_D: {L_data.item():.5f} | L_P: {L_phy.item():.5f} "
                       f"| L_BC: {L_bc.item():.5f} | Diffusivity(D): {D.item():.4e} | Proliferation(ρ): {rho.item():.4e}")

        logger.info(f"Adam training complete after {epochs} epochs.")
        return loss_history

    def train_lbfgs(self, X_data, C_data, X_collocation, X_initial, X_spatial,
                   epochs: int, lr: float):
        """
        Implements the L-BFGS-B optimizer strategy and the main training loop.
        """
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=lr,
            max_iter=epochs,
            line_search_fn='strong_wolfe'
        )

        loss_history = []
        current_iter = 0

        logger.info(f"\n[L-BFGS REFINEMENT] Starting L-BFGS optimization (Max Iter: {epochs}, LR: {lr})...")

        # Define the mandatory closure function
        def closure():
            nonlocal current_iter
            current_iter += 1

            # 1. Reset gradients
            optimizer.zero_grad()

            # 2. Calculate loss
            L_total, L_data, L_phy, L_bc, L_ic, L_n, L_sindy = self.loss.calculate_total_loss(
                X_data, C_data, X_collocation, X_initial, X_spatial
            )

            # 3. Backpropagation
            L_total.backward()

            # 4. Logging and Checkpointing
            if current_iter % 5 == 0 or current_iter == 1:
                D, rho = self.model.pde_params
                log_data = {
                    'iter': current_iter,
                    'L_total': L_total.item(),
                    'L_data': L_data.item(),
                    'L_phy': L_phy.item(),
                    'L_bc': L_bc.item(),
                    'D': D.item(),
                    'rho': rho.item()
                }
                loss_history.append(log_data)

                print(
                    f"L-BFGS Iter {current_iter:4d} | Total Loss: {L_total.item():.5f} | Data Loss: {L_data.item():.5f} "
                    f"| Physics Loss: {L_phy.item():.5f} | Bounds Loss: {L_bc.item():.5f} | Diffusion(D): {D.item():.4e}"
                    f" | Proliferation(ρ): {rho.item():.4e}")

            return L_total

        optimizer.step(closure)

        logger.info(f"L-BFGS optimization complete after {current_iter} iterations.")
        return loss_history, self.model.pde_params