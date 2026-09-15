# Findings

12 societies of `N = 10,000` per point, `Lambda` on a `0.05` grid refined by
parabolic interpolation, errors bootstrapped over societies.

## `Lambda*` grows like the log of the horizon

**Gaussian model.** Over `T` in `[20, 10,000]`:

    Lambda*(T) = -0.10 + 0.237 ln T

| `T` | 10 | 100 | 1000 | 10,000 |
| --- | --- | --- | --- | --- |
| `Lambda*(T)` | 0.40 (clipped) | `0.998 +/- 0.016` | `1.576 +/- 0.005` | `1.981 +/- 0.004` |

There is no sign of saturation: `Lambda*` rises by `+0.405` over the last decade
of `T`. Gaussian affinities are unbounded, so however picky an agent already is,
there is always a better tail draw worth waiting for.

## This is what fixes `T = 100`

Solving `Lambda*(T) = sigma` gives

    T = 105

So `Lambda = sigma` is not advice for an arbitrary horizon. It is advice for an
agent who expects about a hundred serious encounters -- and `T = 100` is not an
arbitrary choice of the paper's but the horizon at which its own rule of thumb is
exactly right. The measured value at `T = 100` is `0.998 +/- 0.016`; the fit
gives `0.991`.

This is the motivation the paper's choice of `T` was missing, and it should
replace "we ran the simulation for `T = 100` steps" wherever that appears.

## The Uniform model saturates, as it must

    Lambda*(T) = 0.37 + 0.181 ln T

| `T` | 100 | 1000 | 10,000 |
| --- | --- | --- | --- |
| `Lambda*(T)` | `1.246` | `1.705` | `1.835 +/- 0.001` |

The rise over the last decade is `+0.129` against the Gaussian's `+0.405`, and
the curve is bending towards `2 sigma` -- the top of the support of
`U(-2 sigma, 2 sigma)`.

That limit is exactly what the corrected asymptotic formula predicts. With

    E[U_inf] = Lambda^2 / (16 sigma) + Lambda / 4 + 5 sigma / 4

monotone increasing on the support (derivative `Lambda/(8 sigma) + 1/4 > 0`),
the supremum sits at `Lambda = 2 sigma` and equals `2 sigma`. It is a supremum
and not a maximum: at `Lambda = 2 sigma` exactly no uniform-model pair clears the
threshold, nobody marries, and the system reverts to never committing. The time
to approach it diverges, which is the concrete sense in which `T -> infinity` is
not the regime of interest.

## What the fixed convention gives up

Holding `Lambda = sigma` while the horizon grows costs more and more. At
`T = 100` it is optimal by construction; by `T = 1000` it yields `1.610` against
`1.935` at `Lambda*(T)`, and by `T = 10,000` `1.611` against `2.199`. The
convention saturates while the optimum keeps climbing.

## Caveat

All of the above is the *social* optimum, computed for a uniform convention. The
equilibrium threshold of `nash-equilibrium/` is measured only at `T = 100`, so
the gap `Lambda_eq - Lambda*` is known at one horizon. Whether the 3.9% price of
anarchy grows, shrinks or reverses as `T` grows is open, and needs the
best-response iteration rerun at several horizons.
