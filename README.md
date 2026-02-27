# Bayesian Option Pricing Framework

## ⚠️ Project Under Review

This repository contains the **specification** for a Bayesian option pricing framework. The project is currently in the **proposal stage** and awaiting approval before implementation begins.

**Full specification:** See [`PROJECT_SPEC.md`](PROJECT_SPEC.md)

## Quick Summary

A production-quality Python framework for pricing options under **full parameter uncertainty** by treating volatility, drift, jump intensity, and regime states as random variables with posterior distributions (rather than point estimates).

### The Problem We're Solving

1. **Volatility is uncertain**, not fixed
2. **Markets have multiple regimes** with different return distributions
3. **Jump risk** is real and often ignored
4. **Greeks are reported as false confidence** (single numbers that are actually highly uncertain)

### The Solution

Use **Bayesian inference** (NUTS sampling) to:
- Learn return model parameters from data
- Quantify uncertainty in all estimates
- Price options as distributions (not point estimates)
- Report Greeks with credible intervals
- Condition on regime shifts

## Project Status

| Phase | Status | Timeline |
|-------|--------|----------|
| **Specification** | ✅ Complete (awaiting review) | 2026-02-27 |
| **Phase 1** (Return models + inference) | ⏳ Pending approval | Weeks 1–2 |
| **Phase 2** (Option pricing) | ⏳ Pending approval | Weeks 3–4 |
| **Phase 3** (Documentation + notebooks) | ⏳ Pending approval | Weeks 5–6 |

## Key Features (Planned)

- ✅ **Multiple return models:** Gaussian, Student-t (fat tails), Merton jump-diffusion
- ✅ **Bayesian inference:** NUTS sampler via PyMC
- ✅ **Option pricing:** Black-Scholes + Merton formulas
- ✅ **Greeks distributions:** Not point estimates, but credible intervals
- ✅ **Regime-conditional:** Greeks and prices change with market regime
- ✅ **Scenario analysis:** Stress test how prices respond to regime shifts
- ✅ **Production code:** Type hints, comprehensive docstrings, 100+ unit tests

## Mathematics at a Glance

### Return Model
```
Log-returns: y_t = μ + σ ε_t + jumps
Where:
  - μ = regime-conditional drift
  - σ = regime-conditional volatility
  - ε_t ~ Student-t (fat tails)
  - Jumps ~ Poisson(λ) with lognormal size
  - All parameters learned via Bayesian inference
```

### Option Pricing
```
Classical Black-Scholes: C = fixed price (single number)

Bayesian pricing: C ~ P(C | data)
  - Draw parameter sample θ from posterior
  - Compute price C(θ) using BS or Merton
  - Repeat for all posterior samples
  - Result: price distribution with credible intervals
```

### Greeks with Uncertainty
```
Classical: Vega = 0.45 (single number)

Bayesian: Vega ~ P(Vega | data)
  Result: Vega = 0.45 ± 0.12 (90% CI: [0.24, 0.68])
```

## Documentation

- **[`PROJECT_SPEC.md`](PROJECT_SPEC.md)** — Full technical specification (math, implementation plan, decisions)
- **[`docs/`](docs/)** — (Will be populated after approval)

## Getting Involved

This project is **ready for technical review**. If you're interested in:
- Reviewing the mathematical specification
- Providing feedback on the approach
- Suggesting improvements or extensions

Please open an issue or contact the maintainers.

## License

MIT License (pending final approval)

---

**Note:** No code has been written yet. This repository contains the specification only, pending review and approval.
