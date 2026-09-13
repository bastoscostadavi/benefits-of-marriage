"""Large-N (mean-field) formulation, section 3 of the paper.

As ``N -> inf`` the society becomes homogeneous and the state of the system is
just a pair of densities: ``p_ct(u)`` for coupled agents and ``p_st(u)`` for
singles, carrying weights ``b_t`` and ``r_t``.  Writing ``F_A`` for the CDF of
the off-diagonal distribution and ``Gbar_A = 1 - F_A``, one step is

    alpha_t = E_{L~U_t}[Gbar_A(L)]      (chance a random agent accepts you)
    gamma_t = E_{M~C_t}[Gbar_A(M)]      (same, restricted to coupled agents)
    beta_t  = 1 - alpha_t * gamma_t     (chance your partner is *not* poached)

    b_{t+1} p_{c,t+1}(u) = alpha_t p_A(u) [ b_t F_{C_t}(u) + r_t F_{S_t}(u) ]
                         + beta_t b_t p_{c,t}(u) [ alpha_t F_A(u) + 1 - alpha_t ]

    r_{t+1} p_{s,t+1}(u) = alpha_t gamma_t beta_t b_t p_0(u)
                         + r_t p_{s,t}(u) [ alpha_t F_A(u) + 1 - alpha_t ]

which is the density form of Eqs. (33)-(40) of the paper.  The first line says
an agent ends up coupled either by out-bidding whoever it met (new couple, new
utility drawn from ``p_A`` above its old one) or by surviving the step; the
second says a single stays single, plus the inflow of agents whose partner left.

Two implementations:

``uniform_steps``
    Exact, symbolic (sympy), for the Uniform model, where every density stays
    a polynomial.  Reproduces ``b_1 = 1/4``, ``p_c1``, ``p_s1``, ``p_2`` and
    ``b_2 = 1861/5184`` from the paper.
``numeric_steps``
    Grid-based, works for any distribution including the Gaussian one, for
    which the paper reports no closed form.

``p_infinity`` is the asymptotic (= married-couple) density of section 3.3.
Note that its mean does **not** match Eq. (64) of the paper; see
:func:`p_infinity_mean`.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from .affinity import Distribution, _as_dist

__all__ = ["uniform_steps", "numeric_steps", "p_infinity",
           "p_infinity_uniform_closed_form", "p_infinity_mean", "p_infinity_mean_paper"]

u, sigma, Lam = sp.symbols("u sigma Lambda", real=True)


# --------------------------------------------------------------------------
# exact symbolic recursion, Uniform model
# --------------------------------------------------------------------------

def uniform_steps(n_steps=4, sig=None):
    """Exact densities ``p_t(u)`` for the Uniform model, ``t = 0..n_steps``.

    Returns a list of dicts with keys ``b, r, pc, ps, p`` (sympy expressions in
    ``u`` and ``sigma``).  Everything is rational; nothing is evaluated
    numerically along the way.
    """
    s = sigma if sig is None else sp.nsimplify(sig)
    lo, hi = -2 * s, 2 * s
    p0 = 1 / (4 * s)                       # density of U(-2 sigma, 2 sigma)
    F_A = (u + 2 * s) / (4 * s)            # its CDF, on the support
    Gbar = 1 - F_A

    def integrate(expr):
        return sp.simplify(sp.integrate(sp.expand(expr), (u, lo, hi)))

    def cdf_of(dens):
        """CDF of a density supported on [-2s, 2s], as a polynomial in u."""
        x = sp.Symbol("x", real=True)
        return sp.simplify(sp.integrate(sp.expand(dens.subs(u, x)), (x, lo, u)))

    out = [dict(b=sp.Integer(0), r=sp.Integer(1), pc=sp.Integer(0), ps=p0, p=p0)]

    b, r, pc, ps = sp.Integer(0), sp.Integer(1), sp.Integer(0), p0
    for _ in range(n_steps):
        p_t = sp.simplify(b * pc + r * ps)
        alpha = integrate(p_t * Gbar)
        gamma = integrate(pc * Gbar) if b != 0 else sp.Integer(0)
        beta = 1 - alpha * gamma

        F_C = cdf_of(pc) if b != 0 else sp.Integer(0)
        F_S = cdf_of(ps)
        survive = alpha * F_A + (1 - alpha)

        c_next = sp.expand(alpha * p0 * (b * F_C + r * F_S) + beta * b * pc * survive)
        s_next = sp.expand(alpha * gamma * beta * b * p0 + r * ps * survive)

        b_next = integrate(c_next)
        r_next = integrate(s_next)
        pc = sp.simplify(c_next / b_next) if b_next != 0 else sp.Integer(0)
        ps = sp.simplify(s_next / r_next)
        b, r = sp.nsimplify(b_next), sp.nsimplify(r_next)
        out.append(dict(b=b, r=r, pc=pc, ps=ps, p=sp.simplify(b * pc + r * ps)))

    return out


# --------------------------------------------------------------------------
# numeric recursion, any distribution
# --------------------------------------------------------------------------

def numeric_steps(n_steps=100, dist="N", mu=0.0, sig=1.0, mu_s=None, sigma_s=None,
                  grid=4001, tail=6.0):
    """Same recursion on a grid; works for the Gaussian model too.

    Returns ``dict(u=grid, b=..., r=..., pc=..., ps=..., p=..., mean=...)``
    with the time axis first.
    """
    dist = _as_dist(dist)
    mu_s = mu if mu_s is None else mu_s
    sigma_s = sig if sigma_s is None else sigma_s

    lo_a, hi_a = dist.support(mu, sig, tail)
    lo_s, hi_s = dist.support(mu_s, sigma_s, tail) if sigma_s > 0 else (mu_s, mu_s)
    lo, hi = min(lo_a, lo_s) - 1e-9, max(hi_a, hi_s) + 1e-9
    x = np.linspace(lo, hi, grid)
    dx = x[1] - x[0]

    p_A = dist.pdf(x, mu, sig)
    F_A = dist.cdf(x, mu, sig)
    Gbar = 1.0 - F_A

    if sigma_s > 0:
        p_0 = dist.pdf(x, mu_s, sigma_s)
    else:  # A_kk = mu_s exactly: a spike, represented on the grid
        p_0 = np.zeros_like(x)
        p_0[np.argmin(np.abs(x - mu_s))] = 1.0 / dx
    p_0 = p_0 / np.trapezoid(p_0, x)

    def cdf_of(dens):
        c = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * dx)))
        return c

    bs, rs, pcs, pss, ps_, means = [0.0], [1.0], [np.zeros_like(x)], [p_0], [p_0], []
    b, r, pc, ps = 0.0, 1.0, np.zeros_like(x), p_0.copy()
    means.append(float(np.trapezoid(x * p_0, x)))

    for _ in range(n_steps):
        p_t = b * pc + r * ps
        alpha = float(np.trapezoid(p_t * Gbar, x))
        gamma = float(np.trapezoid(pc * Gbar, x)) if b > 0 else 0.0
        beta = 1.0 - alpha * gamma
        survive = alpha * F_A + (1.0 - alpha)

        c_next = alpha * p_A * (b * cdf_of(pc) + r * cdf_of(ps)) + beta * b * pc * survive
        s_next = alpha * gamma * beta * b * p_0 + r * ps * survive

        b = float(np.trapezoid(c_next, x))
        r = float(np.trapezoid(s_next, x))
        pc = c_next / b if b > 0 else np.zeros_like(x)
        ps = s_next / r
        bs.append(b); rs.append(r); pcs.append(pc.copy()); pss.append(ps.copy())
        p = b * pc + r * ps
        ps_.append(p)
        means.append(float(np.trapezoid(x * p, x)))

    return dict(u=x, b=np.array(bs), r=np.array(rs), pc=np.array(pcs),
                ps=np.array(pss), p=np.array(ps_), mean=np.array(means))


# --------------------------------------------------------------------------
# asymptotic / married-couple distribution, section 3.3
# --------------------------------------------------------------------------

def p_infinity(x, lam, dist="N", mu=0.0, sig=1.0):
    """Eq. (61): the utility density of married couples, identical at every step.

    The paper writes the first term as an integral,
    ``int_Lambda^u p_0(x) / Gbar(x) dx``.  That integral is elementary for any
    distribution, since ``p_0 = -dGbar/dx``:

        int_Lambda^u p_0(s)/Gbar(s) ds = ln( Gbar(Lambda) / Gbar(u) ),

    so

        p_inf(u) = Theta(u - Lambda) p_0(u) [ ln(Gbar(Lambda)/Gbar(u))
                                              + F_0(Lambda)/Gbar(Lambda) ].

    That is exactly the paper's Uniform-model closed form Eq. (62), but it now
    holds for the Gaussian model too, where the paper reports none.  It also
    avoids the numerical blow-up: the density still diverges logarithmically at
    the top of a bounded support, but integrably, and the log form is stable.

    An agent sitting below Lambda marries the first partner it likes enough, so
    its final utility is ``p_0`` conditioned on exceeding Lambda -- that is the
    second term, carrying weight ``F_0(Lambda)``.  An agent already at
    ``x > Lambda`` has to clear ``x`` instead, which is the log term.
    """
    dist = _as_dist(dist)
    x = np.asarray(x, dtype=float)
    p0 = dist.pdf(x, mu, sig)
    Gbar = np.clip(1.0 - dist.cdf(x, mu, sig), 1e-300, None)
    F_lam = float(np.atleast_1d(dist.cdf(np.array([lam]), mu, sig))[0])
    Gbar_lam = max(1.0 - F_lam, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = p0 * (np.log(Gbar_lam / Gbar) + F_lam / Gbar_lam)
    return np.where(x >= lam, dens, 0.0)


def p_infinity_uniform_closed_form(x, lam, sig=1.0):
    """Eq. (62): the Uniform-model closed form, for checking ``p_infinity``."""
    x = np.asarray(x, dtype=float)
    p0 = np.where((x >= -2 * sig) & (x <= 2 * sig), 1.0 / (4 * sig), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.log((2 * sig - lam) / np.clip(2 * sig - x, 1e-300, None))
    return np.where(x >= lam, p0 * (log_term + (lam + 2 * sig) / (2 * sig - lam)), 0.0)


def p_infinity_mean(lam, sig=1.0):
    """``E[U_inf]`` for the Uniform model -- **corrected**.

    Integrating ``u p_inf(u)`` over ``[Lambda, 2 sigma]`` with ``p_inf`` from
    Eq. (61) gives

        E[U_inf] = Lambda^2 / (16 sigma) + Lambda / 4 + 5 sigma / 4,

    which is maximal at ``Lambda = 2 sigma``, where it equals ``2 sigma``.

    The paper's Eq. (64) instead reports
    ``3 Lambda^2/(16 sigma) + 3 Lambda/4 + 7 sigma/4``, with a maximum of
    ``4 sigma``.  The two agree at ``Lambda = -2 sigma`` (both give ``sigma``,
    the check the paper performs) and nowhere else: at ``Lambda = sigma`` the
    paper gives 2.6875 against the correct 1.5625, and a long simulation lands
    on 1.5625.  See ``scripts/fig12_p_infinity.py`` for the numerical check and
    ``tests/test_analytic.py`` for the assertion.

    The qualitative reading changes with it.  The paper takes the 4 sigma
    maximum as showing that a very high threshold would be wonderful if only
    one could wait for it.  The true ceiling, 2 sigma, is much closer to what
    the dynamics reach by t = 100 (~1.5 sigma at Lambda = sigma), so the
    asymptotic regime is less of a lost paradise than it looks.
    """
    return lam ** 2 / (16 * sig) + lam / 4 + 5 * sig / 4


def p_infinity_mean_paper(lam, sig=1.0):
    """The paper's Eq. (64) at n = 1, kept for the side-by-side comparison."""
    return 3 * lam ** 2 / (16 * sig) + 3 * lam / 4 + 7 * sig / 4
