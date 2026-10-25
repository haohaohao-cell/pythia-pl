"""Dixon-Coles (1997) bivariate Poisson model with time decay.

The model treats each match as two Poisson counts whose rates are

    lambda_home = exp(attack_home - defence_away + home_advantage)
    lambda_away = exp(attack_away - defence_home)

with a low-score correction ``rho`` for the 0-0, 1-0, 0-1 and 1-1 cells.  Fitted
by maximum likelihood with an exponential time-decay weight, so a match from
three years ago counts less than one from last month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, xlogy

from ..config import DC_HALF_LIFE_DAYS, DC_MAX_GOALS


def _poisson_pmf(lam: np.ndarray, k: int) -> np.ndarray:
    """Poisson probability for counts 0..k-1, broadcasting over ``lam``."""
    kk = np.arange(k)
    log_p = xlogy(kk, lam[..., None]) - lam[..., None] - gammaln(kk + 1)
    return np.exp(log_p)


def _tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles correction for the four low-scoring cells."""
    tau = np.ones_like(lam)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return np.maximum(tau, 1e-10)


class DixonColes:
    name = "dixon_coles"

    def __init__(
        self,
        half_life_days: float = DC_HALF_LIFE_DAYS,
        max_goals: int = DC_MAX_GOALS,
        ridge: float = 1e-3,
    ) -> None:
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.ridge = ridge
        self.teams_: list[str] = []
        self.index_: dict[str, int] = {}
        self.attack_: np.ndarray | None = None
        self.defence_: np.ndarray | None = None
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0

    # ------------------------------------------------------------------ fit
    def _unpack(self, params: np.ndarray, n: int):
        attack = params[:n] - params[:n].mean()  # sum-to-zero identifies the model
        defence = params[n : 2 * n]
        return attack, defence, params[2 * n], params[2 * n + 1]

    def _nll(self, params, home_idx, away_idx, goals_h, goals_a, weights, n):
        attack, defence, home_adv, rho = self._unpack(params, n)
        lam = np.exp(attack[home_idx] - defence[away_idx] + home_adv)
        mu = np.exp(attack[away_idx] - defence[home_idx])
        ll = (
            xlogy(goals_h, lam) - lam + xlogy(goals_a, mu) - mu
            + np.log(_tau(goals_h, goals_a, lam, mu, rho))
        )
        penalty = self.ridge * (np.sum(attack**2) + np.sum(defence**2) + home_adv**2)
        return -np.sum(weights * ll) / weights.sum() + penalty

    def fit(self, train: pd.DataFrame) -> "DixonColes":
        data = train.dropna(subset=["fthg", "ftag"]).sort_values("date")
        self.teams_ = sorted(set(data["home"]) | set(data["away"]))
        self.index_ = {t: i for i, t in enumerate(self.teams_)}
        n = len(self.teams_)

        home_idx = data["home"].map(self.index_).to_numpy()
        away_idx = data["away"].map(self.index_).to_numpy()
        goals_h = data["fthg"].to_numpy(dtype=float)
        goals_a = data["ftag"].to_numpy(dtype=float)

        decay = np.log(2.0) / self.half_life_days
        age = (data["date"].max() - data["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
        weights = np.exp(-decay * age)

        x0 = np.concatenate([np.zeros(2 * n), [0.2, 0.0]])
        bounds = (
            [(-3.0, 3.0)] * (2 * n) + [(-1.0, 2.0), (-0.2, 0.2)]
        )
        res = minimize(
            self._nll,
            x0,
            args=(home_idx, away_idx, goals_h, goals_a, weights, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-9},
        )
        attack, defence, home_adv, rho = self._unpack(res.x, n)
        self.attack_, self.defence_ = attack, defence
        self.home_adv_, self.rho_ = float(home_adv), float(rho)
        self.nll_ = float(res.fun)
        return self

    # -------------------------------------------------------------- predict
    def _rates(self, home: str, away: str) -> tuple[float, float]:
        assert self.attack_ is not None and self.defence_ is not None, "fit() first"
        ih = self.index_.get(home)
        ia = self.index_.get(away)
        att_h = self.attack_[ih] if ih is not None else 0.0
        def_h = self.defence_[ih] if ih is not None else 0.0
        att_a = self.attack_[ia] if ia is not None else 0.0
        def_a = self.defence_[ia] if ia is not None else 0.0
        lam = float(np.exp(att_h - def_a + self.home_adv_))
        mu = float(np.exp(att_a - def_h))
        return lam, mu

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """Joint distribution of the scoreline, shape ``(max_goals, max_goals)``."""
        lam, mu = self._rates(home, away)
        ph = _poisson_pmf(np.array([lam]), self.max_goals)[0]
        pa = _poisson_pmf(np.array([mu]), self.max_goals)[0]
        matrix = np.outer(ph, pa)
        matrix[0, 0] *= 1.0 - lam * mu * self.rho_
        matrix[0, 1] *= 1.0 + lam * self.rho_
        matrix[1, 0] *= 1.0 + mu * self.rho_
        matrix[1, 1] *= 1.0 - self.rho_
        return np.clip(matrix, 0, None) / np.clip(matrix, 0, None).sum()

    def outcome_probs(self, home: str, away: str) -> np.ndarray:
        matrix = self.score_matrix(home, away)
        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()
        probs = np.array([home_win, draw, away_win])
        return probs / probs.sum()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.vstack([self.outcome_probs(h, a) for h, a in zip(frame["home"], frame["away"])])

    def strength_table(self) -> pd.DataFrame:
        """Attack/defence parameters, handy for the explainability plots."""
        assert self.attack_ is not None and self.defence_ is not None, "fit() first"
        return (
            pd.DataFrame({"team": self.teams_, "attack": self.attack_, "defence": self.defence_})
            .sort_values("attack", ascending=False)
            .reset_index(drop=True)
        )
