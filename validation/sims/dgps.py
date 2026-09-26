"""Data-generating processes for the statistical performance study.

VALIDATION_PLAN.md, "Statistical performance study" > "Data-generating
processes". Covariates are X ~ Unif[0, 1]^p, independent, p = 10 unless
stated (D7: p = 50); D8 uses a Gaussian copula with latent correlation 0.6.
The outcome is Y = f(X) + sigma * eps for the regression DGPs, with sigma set
from the population R^2 (0.8 "lo" or 0.3 "hi") and sd(f) from the diagnostic
draw (``diagnostics.json``, written by ``diagnostics.py``). D3-bin and D4-bin
are Bernoulli with mu(X) = expit(lambda * (f(X) - E f(X))), lambda from the
same diagnostic draw. D6 is pure noise with sigma = 1 fixed (one noise level).

Each ``Dgp`` knows its own covariate sampler and which covariates are
relevant (appear in ``f``); the rest are irrelevant by construction, which is
what the sparsity claim's "use of irrelevant covariates" measure checks
against.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit, ndtr

DIAGNOSTICS_PATH = Path(__file__).with_name("diagnostics.json")

N_COVARIATES = 10
N_COVARIATES_D7 = 50
COPULA_LATENT_CORR = 0.6  # D8's latent Gaussian correlation

# The two regression noise levels, named by the population R^2 they target.
R_SQUARED: dict[str, float] = {"lo": 0.8, "hi": 0.3}
NOISE_LEVELS: tuple[str, ...] = tuple(R_SQUARED)
SAMPLE_SIZES: tuple[int, ...] = (200, 1_000, 5_000)


def _hinge(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def f_d1(X: np.ndarray) -> np.ndarray:
    """D1 linear: x1 + 2 x2 - x3."""
    return X[:, 0] + 2 * X[:, 1] - X[:, 2]


def f_d2(X: np.ndarray) -> np.ndarray:
    """D2 additive hinges: 2(x1-0.3)+ - 3(x1-0.7)+ + 2(0.5-x2)+."""
    return (
        2 * _hinge(X[:, 0] - 0.3)
        - 3 * _hinge(X[:, 0] - 0.7)
        + 2 * _hinge(0.5 - X[:, 1])
    )


def f_d3(X: np.ndarray) -> np.ndarray:
    """D3 additive smooth: sin(2 pi x1) + 2(x2-0.5)^2 + exp(x3)."""
    return np.sin(2 * np.pi * X[:, 0]) + 2 * (X[:, 1] - 0.5) ** 2 + np.exp(X[:, 2])


def f_d4(X: np.ndarray) -> np.ndarray:
    """D4 Friedman #1: 10 sin(pi x1 x2) + 20(x3-0.5)^2 + 10 x4 + 5 x5."""
    return (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )


def f_d5(X: np.ndarray) -> np.ndarray:
    """D5 hinge interaction: 4(x1-0.4)+(x2-0.5)+ + x3."""
    return 4 * _hinge(X[:, 0] - 0.4) * _hinge(X[:, 1] - 0.5) + X[:, 2]


def f_d6(X: np.ndarray) -> np.ndarray:
    """D6 pure noise: f = 0 everywhere."""
    return np.zeros(X.shape[0])


# D7 is D3 with p = 50 (47 irrelevant covariates); D8 is D4 with correlated
# covariates. Both reuse the smaller DGP's functional form on the first few
# columns of a wider or correlated covariate draw.
f_d7 = f_d3
f_d8 = f_d4


def sample_independent_uniform(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(n, p))


def sample_copula_uniform(
    rng: np.random.Generator, n: int, p: int, corr: float = COPULA_LATENT_CORR
) -> np.ndarray:
    """Uniform margins from a Gaussian copula with one latent correlation
    ``corr`` between every pair of the ``p`` latent normals (compound
    symmetry). The Pearson correlation between two transformed margins equals
    the Spearman correlation of the latent normals, (6/pi) asin(corr/2): about
    0.58 at corr = 0.6 (VALIDATION_PLAN.md).
    """
    cov = np.full((p, p), corr, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal(size=(n, p)) @ chol.T
    return ndtr(z)  # the standard normal CDF


@dataclass(frozen=True)
class Dgp:
    """One data-generating process.

    ``f`` is the regression function (the raw one for a binary DGP too: its
    mu(X) is expit(lambda * (f(X) - mean_f)), with lambda and mean_f from the
    diagnostic draw). ``levels`` are the noise-level names this DGP runs at:
    both "lo" and "hi" for the two-level regression DGPs, a single ``(None,)``
    for D6 (fixed sigma = 1) and for the binary DGPs (no additive noise).
    """

    name: str
    p: int
    f: Callable[[np.ndarray], np.ndarray]
    relevant: tuple[int, ...]
    sample_x: Callable[[np.random.Generator, int], np.ndarray]
    levels: tuple[str | None, ...]
    binary: bool = False

    def sample_covariates(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.sample_x(rng, n)


def _iid(p: int) -> Callable[[np.random.Generator, int], np.ndarray]:
    return lambda rng, n: sample_independent_uniform(rng, n, p)


REGISTRY: dict[str, Dgp] = {
    "D1": Dgp("D1", N_COVARIATES, f_d1, (0, 1, 2), _iid(N_COVARIATES), NOISE_LEVELS),
    "D2": Dgp("D2", N_COVARIATES, f_d2, (0, 1), _iid(N_COVARIATES), NOISE_LEVELS),
    "D3": Dgp("D3", N_COVARIATES, f_d3, (0, 1, 2), _iid(N_COVARIATES), NOISE_LEVELS),
    "D4": Dgp(
        "D4", N_COVARIATES, f_d4, (0, 1, 2, 3, 4), _iid(N_COVARIATES), NOISE_LEVELS
    ),
    "D5": Dgp("D5", N_COVARIATES, f_d5, (0, 1, 2), _iid(N_COVARIATES), NOISE_LEVELS),
    "D6": Dgp("D6", N_COVARIATES, f_d6, (), _iid(N_COVARIATES), (None,)),
    "D7": Dgp(
        "D7", N_COVARIATES_D7, f_d7, (0, 1, 2), _iid(N_COVARIATES_D7), NOISE_LEVELS
    ),
    "D8": Dgp(
        "D8",
        N_COVARIATES,
        f_d8,
        (0, 1, 2, 3, 4),
        lambda rng, n: sample_copula_uniform(rng, n, N_COVARIATES),
        NOISE_LEVELS,
    ),
    "D3-bin": Dgp(
        "D3-bin",
        N_COVARIATES,
        f_d3,
        (0, 1, 2),
        _iid(N_COVARIATES),
        (None,),
        binary=True,
    ),
    "D4-bin": Dgp(
        "D4-bin",
        N_COVARIATES,
        f_d4,
        (0, 1, 2, 3, 4),
        _iid(N_COVARIATES),
        (None,),
        binary=True,
    ),
}

REGRESSION_DGPS: tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8")
BINARY_DGPS: tuple[str, ...] = ("D3-bin", "D4-bin")


@dataclass(frozen=True)
class Cell:
    """One (DGP, sample size, noise level) combination. ``name`` is the
    stable string that seeds every dataset drawn for it (``seeds.py``) and
    that names its result and cache files.
    """

    dgp: str
    n: int
    noise: str | None = None

    @property
    def name(self) -> str:
        parts = [self.dgp, f"n{self.n:05d}"]
        if self.noise is not None:
            parts.append(self.noise)
        return "_".join(parts)

    def __str__(self) -> str:
        return self.name


def all_cells() -> list[Cell]:
    """The full factorial grid: 7 regression DGPs at 3 sizes and 2 noise
    levels (42), D6 at 3 sizes and its one level (3), and 2 binary DGPs at 3
    sizes (6): 51 cells (VALIDATION_PLAN.md, "Data-generating processes").
    """
    cells = []
    for name, dgp in REGISTRY.items():
        for n in SAMPLE_SIZES:
            for level in dgp.levels:
                cells.append(Cell(name, n, level))
    return cells


def load_diagnostics(path: Path = DIAGNOSTICS_PATH) -> dict:
    import json

    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; run `python -m validation.sims.diagnostics` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def true_values(dgp: Dgp, X: np.ndarray, diagnostics: dict) -> np.ndarray:
    """f(X) for a regression DGP, or mu(X) for a binary DGP."""
    raw = dgp.f(X)
    if not dgp.binary:
        return raw
    entry = diagnostics[dgp.name]
    return expit(entry["lambda"] * (raw - entry["mean_f"]))


def generate(
    dgp: Dgp,
    rng: np.random.Generator,
    n: int,
    noise: str | None,
    diagnostics: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw one (X, y, truth) dataset of size ``n``.

    ``truth`` is f(X) for a regression DGP (what the excess-risk measure
    compares fits against) or mu(X) for a binary DGP. ``noise`` must be one of
    ``dgp.levels``: "lo"/"hi" for the two-level regression DGPs, else ``None``.
    """
    if noise not in dgp.levels:
        raise ValueError(f"{dgp.name} does not run at noise level {noise!r}")
    X = dgp.sample_covariates(rng, n)
    truth = true_values(dgp, X, diagnostics)
    if dgp.binary:
        y = rng.binomial(1, truth).astype(np.float64)
    elif dgp.name == "D6":
        sigma = diagnostics["D6"]["sigma"]
        y = truth + sigma * rng.standard_normal(n)
    else:
        sigma = diagnostics[dgp.name][f"sigma_{noise}"]
        y = truth + sigma * rng.standard_normal(n)
    return X, y, truth
