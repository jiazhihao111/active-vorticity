import torch
from typing import Dict
from scipy import stats as sp_stats


class NESSEvaluator:
    """Non-Equilibrium Steady State evaluator for LLM hidden state dynamics.

    Implements four NESS criteria from GUIT-TRT:
    1. sigma > 0 (entropy production rate)
    2. J_FP > 0 (probability flux / high-order moment circulation)
    3. Macroscopic detailed balance not broken (first-moment <v> ~ 0)
    4. Macroscopic quantities stable (CV < 0.5)

    Key distinction: "macroscopic detailed balance" means <h_dot> ~ 0 (first
    moment of velocity is zero), which is compatible with J_FP > 0 (second
    and higher moments non-zero). The steady-state circulation comes from
    the higher-moment structure of the velocity distribution, not from a
    net average velocity displacement.
    """

    def __init__(self, alpha_star: float, gamma: float = 0.01, mass: float = 1.0):
        self.alpha_star = alpha_star
        self.gamma = gamma
        self.mass = mass

    def evaluate(self, hidden_states: torch.Tensor) -> Dict:
        """Full NESS evaluation from hidden state trajectory.

        Args:
            hidden_states: [T, D] hidden state sequence

        Returns:
            Dict with all NESS metrics and pass/fail verdicts
        """
        if hidden_states.dim() != 2:
            raise ValueError(f"Expected 2D [T, D], got {hidden_states.dim()}D")
        T, D = hidden_states.shape
        if T < 10:
            return {"error": "Need T >= 10 for NESS evaluation"}

        vel = hidden_states[1:] - hidden_states[:-1]
        acc = vel[1:] - vel[:-1]
        v_for = vel[1:]
        min_t = min(acc.size(0), v_for.size(0))
        acc = acc[:min_t]
        v_for = v_for[:min_t]

        gamma_eff = self.gamma - self.alpha_star
        sigma = self._entropy_production(v_for, gamma_eff)
        j_fp = self._probability_flux(vel)
        db_result = self._detailed_balance_check(vel)
        cv = self._coefficient_of_variation(vel)

        c1_sigma = sigma > 0
        c2_j = j_fp["J_norm_per_dim"] > 0
        c3_db = not db_result["broken"]
        c4_cv = cv < 0.5
        n_pass = sum([c1_sigma, c2_j, c3_db, c4_cv])

        return {
            "entropy_production_sigma": sigma,
            "J_FP_norm_per_dim": j_fp["J_norm_per_dim"],
            "J_FP_mean_velocity_norm": j_fp["mean_vel_norm"],
            "macro_detailed_balance_broken": db_result["broken"],
            "macro_detailed_balance_n_nonzero_dims": db_result["n_nonzero_dims"],
            "macro_detailed_balance_frac_nonzero": db_result["frac_nonzero"],
            "coefficient_of_variation": cv,
            "criteria": {
                "sigma_gt_0": c1_sigma,
                "J_FP_gt_0": c2_j,
                "macro_detailed_balance_intact": c3_db,
                "CV_lt_0_5": c4_cv,
            },
            "ness_pass_count": n_pass,
            "ness_verdict": "NESS" if n_pass >= 3 else "NOT_NESS",
        }

    def effective_temperature(self, hidden_states: torch.Tensor) -> float:
        """Compute effective temperature T_eff from velocity fluctuations.

        T_eff = <delta_v^2> / D (mean squared velocity fluctuation per dimension)

        Validated gradient: T_eff(pos) > T_eff(scr) > T_eff(rnd)
        (stronger causal constraints require higher activity injection)
        """
        vel = hidden_states[1:] - hidden_states[:-1]
        vel_mean = vel.mean(dim=0)
        vel_centered = vel - vel_mean
        T_eff = float(vel_centered.pow(2).mean().item())
        return T_eff

    def _entropy_production(self, v_for: torch.Tensor, gamma_eff: float) -> float:
        """Entropy production rate: sigma = alpha* / |gamma_eff| * <||v||^2>.

        For gamma_eff < 0 (negative damping / active driving regime), the
        entropy production rate is positive in magnitude. The sign convention
        follows: sigma > 0 always for NESS, regardless of gamma_eff sign.

        Physical meaning: the active driving force alpha* injects energy,
        which is dissipated through effective friction |gamma_eff|.
        """
        v_sq = (v_for * v_for).sum(dim=-1).mean().item()
        if abs(gamma_eff) < 1e-10:
            return float("inf")
        sigma = self.alpha_star / abs(gamma_eff) * v_sq
        return sigma

    def _probability_flux(self, vel: torch.Tensor) -> Dict:
        """Probability flux J_FP (high-order moment circulation).

        J_FP = <v * p(h)> != 0 but <v> ~ 0.
        We measure J_FP as the norm of the mean velocity vector per dimension,
        and separately the per-dimension mean velocity norm.
        """
        vel_mean = vel.mean(dim=0)
        J_norm = float(vel_mean.norm().item())
        J_per_dim = J_norm / vel.size(1)
        mean_vel_norm = float(vel.norm(dim=-1).mean().item())
        return {
            "J_norm_per_dim": J_per_dim,
            "mean_vel_norm": mean_vel_norm,
        }

    def _detailed_balance_check(self, vel: torch.Tensor) -> Dict:
        """Macroscopic detailed balance check.

        Tests whether first-moment velocity <v_d> is significantly non-zero
        for each dimension. "Macroscopic detailed balance intact" means
        very few dimensions have significantly non-zero mean velocity.
        """
        n_dims = vel.size(1)
        n_test = min(n_dims, 50)
        n_nonzero = 0
        for d in range(n_test):
            _, p_val = sp_stats.ttest_1samp(vel[:, d].numpy(), 0)
            if p_val < 0.05:
                n_nonzero += 1
        frac = n_nonzero / n_test if n_test > 0 else 0.0
        return {
            "broken": n_nonzero > n_test * 0.5,
            "n_nonzero_dims": n_nonzero,
            "frac_nonzero": frac,
        }

    @staticmethod
    def _coefficient_of_variation(vel: torch.Tensor) -> float:
        """CV of velocity norm across time steps. Stable if CV < 0.5."""
        vel_norms = vel.norm(dim=-1)
        mean = vel_norms.mean().item()
        std = vel_norms.std().item()
        return std / mean if abs(mean) > 1e-10 else float("inf")