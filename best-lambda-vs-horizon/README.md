# What horizon is `Lambda = sigma` advice for?

The paper reports that the welfare-optimal threshold is `Lambda* = sigma`, but
measures it at `T = 100`. That horizon is arbitrary, and Fig. 3 of the paper
already shows the curves for different `Lambda` crossing, so `Lambda*` cannot be
horizon-independent. This folder measures `Lambda*(T)` directly.

See **[FINDINGS.md](FINDINGS.md)** for the result.

```bash
python best-lambda-vs-horizon/sweep.py --dist N --seeds 12    # ~5 min, 13 workers
python best-lambda-vs-horizon/sweep.py --dist U --seeds 12
```

## Method

One run at a given `Lambda` produces `u_t` for every `t` along the way, so a
single sweep to `T_MAX = 10,000` yields the whole family of horizons at once.
We sweep `Lambda` over `[0.40, 3.50]` in steps of `0.05`, average over 12
independent societies of `N = 10,000`, and read off the argmax at each `T`.

*Sub-grid resolution.* The argmax on a `0.05` grid is too coarse to see the
trend cleanly, so the peak is refined by a parabola through the top three
points. At `T = 100` this gives `0.998` against a grid answer of `1.00`.

*Error bars.* Bootstrapped over societies, 400 resamples. The unit of
replication is the society, never the agent.

*Where the fit is taken.* Below `T ~ 20` the optimum is at or below the bottom
of the `Lambda` grid and the argmax is clipped, so the log-linear fit starts
at `T = 20`. Points below that are plotted but excluded.
