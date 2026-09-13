"""Tests for the large-N formulation, checked against the paper's own numbers."""

import pathlib
import sys

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from marriage.analytic import (numeric_steps, p_infinity, p_infinity_mean,
                               p_infinity_mean_paper, p_infinity_uniform_closed_form,
                               sigma as sig_sym, u as u_sym, uniform_steps)
from marriage.model import simulate

STEPS = uniform_steps(2)


def test_reproduces_b1_r1():
    assert STEPS[1]["b"] == sp.Rational(1, 4)
    assert STEPS[1]["r"] == sp.Rational(3, 4)


def test_reproduces_b2_r2():
    """Eq. (48) of the paper."""
    assert STEPS[2]["b"] == sp.Rational(1861, 5184)
    assert STEPS[2]["r"] == sp.Rational(3323, 5184)


def test_reproduces_pc1_ps1_p1():
    """Eqs. (43)-(45): p_c1 = (2s+u)/(2s) p_0 etc., with p_0 = 1/(4 sigma)."""
    p0 = 1 / (4 * sig_sym)
    assert sp.simplify(STEPS[1]["pc"] - (2 * sig_sym + u_sym) / (2 * sig_sym) * p0) == 0
    assert sp.simplify(STEPS[1]["ps"] - (6 * sig_sym + u_sym) / (6 * sig_sym) * p0) == 0
    assert sp.simplify(STEPS[1]["p"] - (4 * sig_sym + u_sym) / (4 * sig_sym) * p0) == 0


def test_reproduces_pc2_ps2():
    """Eq. (49)."""
    s = sig_sym
    pc2 = 3 * (2 * s + u_sym) * (2258 * s + 335 * u_sym) / (7444 * (2 * s) ** 3)
    ps2 = (3233 * (2 * s) ** 2 + 3672 * s * u_sym + 270 * u_sym ** 2) / (6646 * (2 * s) ** 3)
    assert sp.simplify(STEPS[2]["pc"] - pc2) == 0
    assert sp.simplify(STEPS[2]["ps"] - ps2) == 0


def test_densities_are_normalised():
    for d in STEPS:
        mass = sp.integrate(sp.expand(d["p"].subs(sig_sym, 1)), (u_sym, -2, 2))
        assert sp.simplify(mass - 1) == 0


def test_mean_utility_after_one_step():
    """E[U_1] = sigma/(2 sqrt(pi)) Gaussian, sigma/3 Uniform (Eq. 46)."""
    assert numeric_steps(1, "N")["mean"][1] == pytest.approx(1 / (2 * np.sqrt(np.pi)), abs=1e-3)
    assert numeric_steps(1, "U")["mean"][1] == pytest.approx(1 / 3, abs=1e-3)


def test_numeric_recursion_matches_symbolic():
    num = numeric_steps(2, "U")
    assert num["b"][1] == pytest.approx(0.25, abs=1e-3)
    assert num["b"][2] == pytest.approx(float(sp.Rational(1861, 5184)), abs=1e-3)


def test_meanfield_is_exact_at_the_first_step_only():
    """The equations give b_1, b_2 and u_1 exactly, but u_2 is already biased.

    Run at N = 200,000 so the residual is the theory's, not the sample's.
    """
    mf = numeric_steps(3, "N")
    sim = simulate(n=200_000, steps=3, dist="N", lam=np.inf, seed=13)
    assert sim.coupled_share[1] == pytest.approx(mf["b"][1], rel=0.01)
    assert sim.coupled_share[2] == pytest.approx(mf["b"][2], rel=0.01)
    assert sim.mean_utility[1] == pytest.approx(mf["mean"][1], rel=0.01)
    # u_2 is off by ~3.5%, well outside sampling error at this N
    assert abs(sim.mean_utility[2] - mf["mean"][2]) / mf["mean"][2] > 0.02


def test_p_infinity_matches_the_uniform_closed_form():
    """Eq. (61) evaluated with the log identity equals Eq. (62) exactly."""
    grid = np.linspace(-2, 2 - 1e-9, 50_001)
    for lam in (-2.0, -1.0, 0.0, 1.0, 1.5):
        a = p_infinity(grid, lam, "U", sig=1.0)
        b = p_infinity_uniform_closed_form(grid, lam, sig=1.0)
        assert np.allclose(a, b, atol=1e-9)
        assert np.trapezoid(a, grid) == pytest.approx(1.0, abs=1e-3)


def test_corrected_asymptotic_mean():
    """E[U_inf] = L^2/(16 s) + L/4 + 5 s/4, not the paper's Eq. (64).

    Checked by integrating the density, and by a long simulation.
    """
    grid = np.linspace(-2, 2 - 1e-9, 400_001)
    for lam in (-2.0, -1.0, 0.0, 1.0, 1.5):
        by_integral = np.trapezoid(grid * p_infinity(grid, lam, "U", sig=1.0), grid)
        assert by_integral == pytest.approx(p_infinity_mean(lam), abs=2e-3)

    # the two formulas agree only at Lambda = -2 sigma
    assert p_infinity_mean(-2.0) == pytest.approx(p_infinity_mean_paper(-2.0))
    assert p_infinity_mean(1.0) == pytest.approx(1.5625)
    assert p_infinity_mean_paper(1.0) == pytest.approx(2.6875)

    res = simulate(n=20_000, steps=800, dist="U", lam=1.0, seed=14, track_utility=True)
    assert res.utility[-1][res.married].mean() == pytest.approx(1.5625, abs=0.02)
