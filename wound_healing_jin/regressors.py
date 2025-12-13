import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from logging_utils import setup_logger

logger = setup_logger()

class Regressor:
    """
    Implements the core regression algorithms for sparse Identification of Nonlienar Dynamics.
    """
    def __init__(self, threshold: float = 0.05, alpha: float = 1e-5, max_iter: int = 20 ):
        """
        Initializes the SINDy model parameters.

        """
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.coefficients = None
        self.candidate_terms = None
    def fit_least_squares(self, Theta:pd.DataFrame, Y:pd.DataFrame)-> np.ndarray:
        """
        Fits the sparse coefficients (Xi) using standard Least Squares (l2-norm)
        Args:
            Theta: Design Matrix
            Y: Target Vector
        """
        logger.info("Fitting model using standard Least Squares (numpy.linalg.lstq).")

        Theta_np = Theta.values
        Y_np = Y.values

        Xi, residuals, rank, s = np.linalg.lstsq(Theta_np, Y_np, rcond=None)
        self.candidate_terms = Theta.columns.tolist()
        logger.info(f"Least Squares fit completed. Rank: {rank}, Coefficients Shape: {Xi.shape}.")

        return Xi

    def fit_lasso(self, Theta:pd.DataFrame, Y:pd.DataFrame)-> np.ndarray:
        """
        Fits the sparse coefficients (Xi) using L1-regularized least squares (lasso)
        This directly enforces sparsity (driving some coefficents to exactly zero).
        Args:
            Theta: Design Matrix
            Y: Target Vector
        """
        logger.info(f"Fitting model using Lasso (L1 regularization) with alpha={self.alpha}.")

        # Lasso implementation from scikit-learn. fit_intercept=False is standard for SINDy.
        model = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10000)

        # Scikit-learn expects 1D target array for single output regression
        Y_1d = Y.values.ravel()

        model.fit(Theta.values, Y_1d)

        # Coefficients are stored as a 1D array, reshape to column vector for consistency
        Xi = model.coef_.reshape(-1, 1)

        self.coefficients = Xi
        self.candidate_terms = Theta.columns.tolist()
        logger.info(f"Lasso finished. Non-zero terms found: {np.sum(np.abs(Xi) > 1e-10)}")

        return Xi

    def fit_stridge(self, Theta: pd.DataFrame, Y: pd.DataFrame) -> np.ndarray:
        """
        Implements the Sequential Threshold Ridge Regression (STRidge) algorithm
        for robust sparsity enforcement and coefficient discovery.

        This method iteratively prunes terms whose coefficients fall below the threshold.
        """
        logger.info(f"Fitting model using STRidge (Iterative Thresholding).")
        logger.info(f"Params: Threshold={self.threshold}, Alpha={self.alpha}, Max Iterations={self.max_iter}")

        # Initialize: Start with all terms included (mask = True)
        n_terms = Theta.shape[1]
        mask = np.ones(n_terms, dtype=bool)
        Xi = np.zeros((n_terms, 1))

        Theta_np = Theta.values
        Y_np = Y.values

        # Iterate to prune terms
        for iteration in range(self.max_iter):
            # 1. Fit the model using Ridge regression on the currently active terms (mask)

            # Use sklearn's Ridge for stable L2 regularization
            model = Ridge(alpha=self.alpha, fit_intercept=False, max_iter=10000)

            # Fit only on the active columns of Theta
            model.fit(Theta_np[:, mask], Y_np.ravel())

            # Update the full coefficient vector
            Xi[mask] = model.coef_.reshape(-1, 1)

            # Check for convergence: if no terms are removed, we stop
            terms_removed_in_this_iter = 0

            # 2. Identify terms whose coefficients are below the threshold
            for i in range(n_terms):
                if mask[i] and np.abs(Xi[i]) < self.threshold:
                    mask[i] = False  # Prune the term
                    Xi[i] = 0.0  # Set coefficient to zero
                    terms_removed_in_this_iter += 1

            if terms_removed_in_this_iter == 0:
                logger.info(f"STRidge converged at iteration {iteration}. No more terms pruned.")
                break

        self.coefficients = Xi
        self.candidate_terms = Theta.columns.tolist()
        logger.info(f"STRidge finished. Final active terms: {np.sum(mask)}.")
        return Xi

    def extract_equation(self, decimals: int = 4) -> str:
        """
        Translates the final sparse coefficients into a human-readable PDE string.

        Args:
            decimals (int): Number of decimal places for coefficient rounding.

        Returns:
            str: The discovered governing PDE equation.
        """
        if self.coefficients is None or self.candidate_terms is None:
            return "Error: Model must be fit before extracting the equation."

        equation_parts = []

        # Mapping common feature column names to mathematical notation (C = density_gaussian)
        mapping = {
            'density_gaussian': 'C',  # Zeroth order term
            'grad_C': '\\nabla C',  # First spatial derivative
            'laplacian_C': '\\nabla^2 C',  # Second spatial derivative (Diffusion)
            'C_logistic': 'C(1 - C/K)',  # Logistic growth
            'C_pow2': 'C^2',  # Polynomial
            'C_pow3': 'C^3',
            'C_gradC': 'C\\nabla C',
            'C_laplacian_C': 'C\\nabla^2 C',
            # Add other candidate terms and their notation here
        }

        # Iterate through the coefficients and build the string
        for i, (term_name, coef) in enumerate(zip(self.candidate_terms, self.coefficients.flatten())):
            if np.abs(coef) > 1e-10:  # Only include non-zero coefficients
                rounded_coef = round(coef, decimals)

                # Determine sign and prefix
                if not equation_parts:  # First term (no leading '+')
                    sign_prefix = ""
                elif rounded_coef > 0:
                    sign_prefix = " + "
                else:  # Coefficient is negative, sign is already included
                    sign_prefix = " "

                    # Get the mathematical symbol for the term
                term_symbol = mapping.get(term_name, term_name)  # Use column name if mapping fails

                # Format the term: [+/-] [coef] * [term_symbol]
                equation_part = f"{sign_prefix}{rounded_coef} {term_symbol}"
                equation_parts.append(equation_part)

        if not equation_parts:
            return "Equation: \\frac{\\partial C}{\\partial t} \\approx 0 (No active terms found)"

        # Final assembled equation
        equation = "\\frac{\\partial C}{\\partial t} \\approx " + "".join(equation_parts)
        return equation
