from typing import List, Dict

import torch
from pinn_model import WoundHealingPINN
from pinn_loss_functions import PINNLoss
from logging_utils import setup_logger
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

        loss_history = []

        logger.info(f"\n[ADAM TRAINING] Starting training for {epochs} epochs (LR: {lr})...")

        for epoch in range(1, epochs + 1):

            optimizer.zero_grad()

            # Calculate all loss components by calling the dedicated loss manager
            L_total, L_data, L_phy, L_bc, L_ic, L_n = self.loss.calculate_total_loss(
                X_data, C_data, X_collocation, X_initial, X_spatial
            )

            L_total.backward()
            optimizer.step()

            # Logging and Checkpointing
            if epoch % 100 == 0 or epoch == 1:
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

                print( f"\nEpoch {epoch:5d} \nL_T: {L_total.item():.5f} \nL_D: {L_data.item():.5f} \nL_P: {L_phy.item():.5f} "
                       f"\nL_BC: {L_bc.item():.5f} \nDiffusivity(D): {D.item():.4e} \nProliferation(ρ): {rho.item():.4e}")

        logger.info(f"Adam training complete after {epochs} epochs.")
        return loss_history