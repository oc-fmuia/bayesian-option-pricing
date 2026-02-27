# Bayesian Option Pricing Framework — Project Specification

**Status:** Proposal (awaiting approval before implementation)  
**Date Created:** 2026-02-27  
**Target:** Independent GitHub repo (separate from regime-switching project)

---

## Executive Summary

Build a standalone **Bayesian option pricing framework** that extends classical option pricing (Black-Scholes-Merton) by treating all uncertain parameters (volatility, drift, jump intensity) as random variables with posterior distributions.

**Core innovation:** Option prices are not point estimates, but distributions that account for:
1. **Volatility uncertainty** (σ is estimated from data, not known)
2. **Regime-switching volatility** (σ changes with market regimes)
3. **Jump risk** (market crashes, earnings gaps have regime-dependent intensity)
4. **Full parameter uncertainty** (all parameters jointly inferred via Bayesian inference)

**Deliverable:** Production-quality Python library for pricing vanilla/exotic options under Bayesian uncertainty, with complete mathematical documentation and educational notebooks.

---

## Problem Statement

### Current Approach (Industry Standard)

```
Black-Scholes pricing:

Input:
  • S = spot price (observed)
  • K = strike price (contract terms)
  • T = time to maturity (contract terms)
  • σ = volatility (ESTIMATED from historical returns, then treated as FIXED)
  • r = risk-free rate (observable)
  • q = dividend yield (contract terms)

Output:
  • C = call price (single deterministic value)
  • Greeks (delta, gamma, vega, theta, rho) — single point estimates

Used for:
  • Pricing derivatives
  • Risk management (Greeks)
  • Implied volatility extraction
  • Volatility arbitrage trading
```

### What's Wrong

1. **Volatility is uncertain**
   - Estimated σ from historical data has estimation error (20–40% common)
   - σ changes over time (volatility clustering, regime shifts)
   - But uncertainty is ignored in pricing and Greeks
   - Result: Pricing error of 5–20% typical in turbulent markets

2. **Single-regime assumption fails**
   - Black-Scholes assumes constant μ and σ
   - Markets have multiple regimes (growth, crisis, recovery)
   - Option Greeks (especially vega, gamma) are regime-dependent
   - Crisis regime: high vol, non-normal tails, jump risk
   - Result: Hedging fails when regimes shift (worst time to be wrong)

3. **Jump risk ignored**
   - Gaps from earnings announcements, policy surprises, market crashes
   - Classical option pricing has no jump component
   - Results: OTM options systematically misprice (empirical fact)
   - Tail risk underestimated in crisis scenarios

4. **Greeks are false confidence**
   - Reported as single numbers (e.g., vega = 0.45)
   - But vega is highly uncertain and regime-dependent
   - Trader thinks a 1% vol move = $450 PnL change
   - In crisis: actual change might be 10× this
   - Result: Risk management fails under stress

### Use Cases Enabled by Bayesian Approach

| User | Current Problem | Our Solution |
|------|-----------------|--------------|
| **Derivatives trader** | Prices 100 similar options; can't detect systematic mispricing | Bayesian credible intervals show if market price is outside our distribution |
| **Risk manager** | Greeks don't account for regime shifts; hedges break in crises | Regime-conditional Greeks; alerts if regime probability changes |
| **Hedge fund** | Long OTM calls; underestimated tail risk using BS | Full posterior lets us see probability of 10× payout scenarios |
| **Market maker** | Sets bid-ask spreads based on BS volatility; gets picked off | Uncertainty bounds show when spread is too tight for the uncertainty we face |
| **Volatility analyst** | Estimates implied vol surface; ignores regime structure | Surface changes with regime probability; can trade the surface shift |

---

## Solution Architecture

### High-Level Data Flow

```
Historical returns (daily OHLC + trades)
         │
         ▼
┌──────────────────────────────────────┐
│  Bayesian Parameter Inference        │
│  (Fit returns model to data)         │
│  • Regime identification (Markov)    │
│  • Volatility (regime-conditional)   │
│  • Jump intensity (Poisson, regime)  │
│  • Jump distribution (student-t)     │
│  • Drift (regime-conditional)        │
└──────────────────────────────────────┘
         │
         ▼ Posterior samples
┌──────────────────────────────────────┐
│  Option Pricing Module               │
│  (For each parameter sample)         │
│  • Black-Scholes (no jumps)         │
│  • Merton Jump-Diffusion (with jumps)│
│  • Monte Carlo (exotic options)      │
└──────────────────────────────────────┘
         │
         ▼ Price samples
┌──────────────────────────────────────┐
│  Aggregation & Reporting             │
│  • Price distribution                │
│  • Greeks (delta, gamma, vega, etc.) │
│  • Credible intervals & tail metrics │
│  • Regime-conditional Greeks         │
│  • Stress test scenarios             │
└──────────────────────────────────────┘
         │
         ▼
Output:
  • Fair price (posterior median)
  • Price credible intervals (50%, 90%)
  • Greeks by regime
  • Tail risk metrics
  • Scenario analysis
```

---

## Mathematical Specification

### Part 1: Return Model (Underlying Asset Dynamics)

#### Model Specification

We model log-returns as:

```
y_t = log(S_t / S_{t-1})

Case 1: Gaussian model (baseline)
  y_t = μ + σ ε_t,  ε_t ~ N(0, 1)

Case 2: Student-t model (fat tails, recommended)
  y_t = μ + σ ε_t,  ε_t ~ T_ν(0, 1)
  where ν = degrees of freedom (tail thickness)

Case 3: Jump-diffusion (Merton, for crisis data)
  y_t = (μ - λ κ) dt + σ dW_t + dJ_t
  where:
    λ = Poisson jump intensity (jumps per year)
    κ = E[e^J - 1] = expected jump impact
    J_t ~ N(μ_J, σ_J²) = jump size distribution
```

#### Regime-Switching Extension

```
Add regime state s_t ∈ {1, 2, ..., K}:

  y_t | s_t, λ_{s_t} ~ N(μ_{s_t}, σ_{s_t}²)   [or Student-t]
  
  With regime-conditional parameters:
    μ_k = drift in regime k
    σ_k = volatility in regime k
    λ_k = jump intensity in regime k
    ν_k = tail index in regime k
    
  Regime dynamics:
    P(s_t = j | s_{t-1} = i) = P_ij  (Markov transition)
```

#### Prior Specification

```
For each regime k:

  μ_k ~ Normal(0, 0.1)              # Drift prior
  σ_k ~ HalfNormal(0, 1)            # Vol prior (positive)
  ν_k ~ Exponential(0.1)            # Tail index (encourages fat tails)
  λ_k ~ Gamma(2, 0.5)               # Jump intensity (≥0, rare by default)
  μ_J ~ Normal(−0.03, 0.05)         # Jump mean (negative = crash bias)
  σ_J ~ HalfNormal(0, 0.1)          # Jump vol
  
Transition matrix:
  P_k ~ Dirichlet(α)  where α encodes persistence
```

#### Likelihood

```
For each observation y_t:

Case 1 (Gaussian):
  L(y_t | μ, σ) = (1 / (σ√(2π))) exp(−(y_t − μ)² / (2σ²))

Case 2 (Student-t):
  L(y_t | μ, σ, ν) ∝ [1 + ((y_t − μ) / σ)² / ν]^{−(ν+1)/2}

Case 3 (Jump-diffusion):
  L(y_t | μ, σ, λ, J) = mixture of:
    • P(no jump) × Gaussian likelihood
    • P(1 jump) × Jump likelihood
    • P(2+ jumps) ~ negligible
```

### Part 2: Bayesian Inference

#### Posterior Distribution

```
Joint posterior:

  P(μ, σ, λ, ν, s, P | {y_1, ..., y_T})
  
  ∝ [∏_t L(y_t | μ, σ, λ, ν, s_t)] × P(μ) P(σ) P(λ) P(ν) P(P)

Interpretation:
  • (μ, σ, λ, ν) = regime-conditional return parameters
  • s = hidden regime path
  • P = transition probabilities
  • Posterior = full distribution over all unknowns, given observed returns
```

#### Inference Algorithm

```
Use NUTS (No-U-Turn Sampler) for posterior inference:

  1. Specify priors (as above)
  2. Feed observed returns to likelihood
  3. Run NUTS (via PyMC) to sample from posterior
     • Output: 2000 draws of (μ, σ, λ, ν, s, P)
  4. Diagnose convergence (Rhat, ESS)
  5. Extract posterior samples for pricing

Alternative: Particle filter (if regime inference is primary goal)
  • Used by practitioners when fast regime tracking is needed
  • Less flexible than NUTS, but computationally cheaper
```

### Part 3: Option Pricing Under Parameter Uncertainty

#### Black-Scholes Price (No Jumps)

```
Classical formula:

  C(S, K, T, σ, r, q) = S e^{−qT} N(d_1) − K e^{−rT} N(d_2)
  
  where:
    d_1 = [log(S/K) + (r − q + σ²/2)T] / (σ √T)
    d_2 = d_1 − σ √T
    N(·) = CDF of standard normal

Greeks:
  Δ = ∂C/∂S = e^{−qT} N(d_1)                    (delta)
  Γ = ∂²C/∂S² = n(d_1) / (S σ √T)               (gamma)
  ν = ∂C/∂σ = S e^{−qT} n(d_1) √T / 100        (vega, per 1% change)
  Θ = ∂C/∂T = [−S e^{−qT} n(d_1) σ / (2√T)] − rK e^{−rT} N(d_2) (theta)
  ρ = ∂C/∂r = K T e^{−rT} N(d_2)               (rho)
```

#### Merton Jump-Diffusion Price

```
Merton (1976) adds jump risk:

  dS/S = (μ − λ κ) dt + σ dW + dJ

  where J_t = jump process with Poisson intensity λ and lognormal jump sizes

Closed-form pricing:

  C_Merton(S, K, T, σ, r, q, λ, J_dist) 
    = Σ_{n=0}^∞ [e^{−λT} (λT)^n / n!] × C_BS(S, K, T, σ_n, r_n, q)
  
  where:
    σ_n² = σ² + n Var(jump) / T  (conditional vol given n jumps)
    r_n = adjusted drift accounting for n expected jumps

Interpretation:
  • Sum weighted by probability of exactly n jumps
  • Each term is Black-Scholes with adjusted volatility
  • Tail adjustment: OTM options get premium from jump risk
```

#### Bayesian Pricing: Full Posterior Distribution

```
Output: not a single price, but P(C | data)

Algorithm:

  1. Draw parameter sample θ = (μ, σ, λ, ν) from posterior
  2. Compute price using appropriate formula:
       if λ ≈ 0: C = Black-Scholes(S, K, T, σ, r)
       if λ > 0: C = Merton(S, K, T, σ, λ, J_dist)
  3. Repeat for all posterior samples → {C_1, C_2, ..., C_M}
  4. Aggregate into price distribution:
       - Posterior mean: E[C] = mean of samples
       - Credible interval: quantiles of samples (50%, 95%, etc.)
       - Posterior std: uncertainty in price

Output for trading:
  • Fair value = posterior median C
  • Lower 5%ile / upper 95%ile = credible interval
  • Uncertainty = posterior std dev
  
  Interpretation: "95% confident true price is in [C_lower, C_upper]"
```

#### Greeks Under Parameter Uncertainty

```
For each Greek (delta, gamma, vega, etc.):

  1. Compute Greek for each posterior parameter sample
  2. Aggregate: Δ ~ P(Δ | data)

Example (Vega):

  For sample i with σ_i:
    Vega_i = S e^{−qT} n(d_1^i) √T
    
  Aggregate:
    E[Vega] = mean of {Vega_1, ..., Vega_M}
    SD[Vega] = std of {Vega_1, ..., Vega_M}
    
  Output: "Vega is 0.45 ± 0.12 (90% CI: [0.24, 0.68])"
  
  Interpretation:
    • Baseline: 1% vol increase → $0.45 PnL
    • But uncertainty: could be $0.24 to $0.68
    • Especially large when regime uncertainty is high
```

#### Regime-Conditional Greeks

```
Group posterior samples by most-likely regime:

  For each regime k:
    • Filter posterior samples where s_T = k (high posterior probability)
    • Compute Greeks on this subset
    
  Output:
    Greeks_growth = {Δ, Γ, ν, Θ, ρ} in low-vol regime
    Greeks_crisis = {Δ, Γ, ν, Θ, ρ} in high-vol regime
    
  Example:
    Growth:  Vega = 0.25
    Crisis:  Vega = 1.80
    
  Interpretation: "In normal times, vol changes don't matter much"
                  "In crisis, vega explodes (convexity risk)"
```

### Part 4: Output & Scenario Analysis

#### Price Reporting

```
For vanilla option (e.g., 6-month ATM call):

Standard output:
  Fair value:        $5.00 (posterior median)
  90% CI:            [$4.50, $5.75]
  Posterior SD:      $0.35
  Regime probability:
    - Growth (prob 75%): $4.30
    - Crisis (prob 25%): $7.50

Interpretation: "True price most likely around $5.00, but could reasonably 
                be $4.50–$5.75 depending on parameter realizations"
```

#### Greeks Reporting

```
Standard Greeks (volatility-weighted):

  Delta:   0.65 ± 0.05      (stock exposure)
  Gamma:   0.042 ± 0.008    (convexity)
  Vega:    0.45 ± 0.12      (vol exposure)
  Theta:   −0.018 ± 0.003   (time decay)
  Rho:     0.25 ± 0.02      (rate exposure)

Regime-conditional Greeks:

  Growth regime (75% prob):
    Vega_growth = 0.25
    
  Crisis regime (25% prob):
    Vega_crisis = 1.80
    
  Interpretation: "In crisis, option becomes much more sensitive to vol"
```

#### Stress Testing

```
Scenario 1: "Regime shift from growth to crisis"
  Current regime prob: [growth: 85%, crisis: 15%]
  New regime prob:     [growth: 10%, crisis: 90%]
  
  Price change: $5.00 → $7.50 (+50%)
  Vega change: 0.25 → 1.80 (7× increase)
  
  → "If crisis hits, option worth 50% more but exposed to much more vol"

Scenario 2: "Volatility jump"
  Current σ distribution: [5th%ile: 20%, median: 35%, 95th%ile: 55%]
  Shock: σ → 60%
  
  Price change: $5.00 → $6.20
  Greeks impact: All Greeks increase, especially Vega
  
Scenario 3: "Jump event"
  If jump intensity λ increases (crisis signal):
    OTM call price increases more than BS predicts
    Gamma/Vega become more important
    Portfolio PnL changes non-linearly with spot moves
```

#### Implied Volatility Under Regime Shift

```
Standard approach: One IV per strike

Bayesian approach: IV is uncertain and regime-dependent

Example:
  Market call price: $5.20
  Our credible interval: [$4.50, $5.75]
  
  Implied vol (using our median price $5.00): 35%
  Implied vol (if market underprices): 37%
  Implied vol (if market overprices): 33%
  
  → Detect if market price is too high/low relative to our beliefs
  → Trade if misprice is large relative to uncertainty
```

---

## Implementation Plan

### Scope (What Will Be Built)

#### Phase 1: Core Infrastructure (Weeks 1–2)

1. **Return Model Module** (`src/returns/`)
   - `DiffusionModel` (base class)
   - `GaussianModel` (simplest, for baseline)
   - `StudentTModel` (fat tails)
   - `MertonJumpModel` (jumps)
   - Tests: Likelihood computation, sampling, parameter validation

2. **Bayesian Inference** (`src/inference/`)
   - `ReturnModelBuilder` (PyMC model assembly)
   - `ReturnSampler` (NUTS sampling for return models)
   - `ReturnDiagnostics` (Rhat, ESS for return models)
   - Tests: Model specification, posterior validation

3. **Utilities**
   - `utils/data_loading.py` — OHLC data ingestion
   - `utils/returns.py` — Log-return computation
   - `tests/` — Comprehensive unit tests

**Deliverable:** Standalone module for Bayesian inference on returns (independent of pricing)

#### Phase 2: Option Pricing (Weeks 3–4)

1. **Pricing Module** (`src/pricing/`)
   - `BlackScholesPrice` (classical, using scipy.stats)
   - `MertonJumpPrice` (Merton formula)
   - `OptionGreeks` (delta, gamma, vega, theta, rho)
   - Tests: Validation against known benchmarks, Greeks numerical verification

2. **Bayesian Pricing** (`src/pricing/bayesian_pricer.py`)
   - `BayesianOptionPrice` (applies posterior to pricing)
   - `PriceDistribution` (aggregates price samples)
   - `GreeksDistribution` (aggregates Greeks)
   - Tests: Posterior propagation, regime-conditional outputs

3. **Reporting** (`src/reporting/`)
   - `PriceReport` (fair value, CI, regime breakdown)
   - `GreeksReport` (Greeks distribution, regime-conditional)
   - `StressTestReport` (scenario analysis)

**Deliverable:** End-to-end pipeline: returns → posterior → prices → reporting

#### Phase 3: Documentation & Notebooks (Weeks 5–6)

1. **Mathematical Documentation**
   - `docs/returns_models.md` — Gaussian, Student-t, Jump models
   - `docs/bayesian_inference.md` — NUTS for returns, priors, diagnostics
   - `docs/option_pricing.md` — Black-Scholes, Merton, Greeks
   - `docs/bayesian_pricing.md` — Posterior propagation, scenario analysis

2. **Educational Notebooks**
   - `notebooks/01_return_model_inference.ipynb` — Fit models to real data
   - `notebooks/02_option_pricing_basics.ipynb` — BS vs. Merton pricing
   - `notebooks/03_bayesian_option_pricing.ipynb` — Full pipeline
   - `notebooks/04_interactive_trader.ipynb` — Interactive tool

3. **Example Cases**
   - Real S&P 500 options pricing under 2008 crisis regime
   - VIX options with jump component
   - Implied volatility surface under regime shift

**Deliverable:** Production-ready documentation + educational materials

### Scope Clarification: What We're NOT Building

1. **Real-time inference** — Focuses on batch analysis, not live updating
2. **Exotic option pricing** — Starts with vanilla (European) options
3. **Market impact / transaction costs** — Ignores microstructure
4. **Portfolio optimization** — Focuses on single-option pricing, not portfolio-level optimization
5. **Data ingestion pipeline** — Assumes data is already available as clean OHLC
6. **Web UI / trading system** — Pure Python library, not production trading platform

(These can be Phase 2 extensions if there's interest)

---

## Technical Decisions to Confirm

### Decision 1: Return Model Choice

**Options:**
- A: Gaussian (simple, baseline, unrealistic tails)
- B: Student-t (fat tails, better empirically, recommended)
- C: Jump-diffusion (includes crash risk, more complex)
- D: All three (support all models, user selects)

**Recommendation:** D (all three)
- Gaussian useful for comparison
- Student-t is standard in practice
- Merton useful for crisis data
- Code handles all via polymorphism

**Question for Francesco:** Approve this approach?

---

### Decision 2: Regime-Switching Complexity

**Options:**
- A: No regime switching (single-regime only)
- B: Optional regime switching (user can enable/disable)
- C: Regime switching as default (always fit regimes)

**Recommendation:** B (optional)
- Simpler to start without regimes
- Can add regime component later
- Cleaner separation of concerns

**Question for Francesco:** Should we include regime switching in Phase 1, or defer to Phase 2?

---

### Decision 3: Exotic Options

**Options:**
- A: Vanilla only (European calls/puts)
- B: American options (early exercise)
- C: Barrier options (knock-in/out)
- D: Exotic smile (smile-aware pricing)

**Recommendation:** A (vanilla only in Phase 1)
- American options require binomial/finite-diff (different algorithm)
- Barrier/exotic need Monte Carlo
- Vanilla options are 80% of use cases
- Can extend later

**Question for Francesco:** Proceed with vanilla-only Phase 1?

---

### Decision 4: Inference Speed vs. Accuracy

**Options:**
- A: Full NUTS sampling (slow, ~5–10 min per symbol, high accuracy)
- B: Variational inference (fast, ~10 sec per symbol, lower accuracy)
- C: Both (user selects)

**Recommendation:** A (full NUTS)
- Variational inference is approximate and less well-studied for price distributions
- Practitioners can afford 5–10 min for daily analysis
- Can add VI as Phase 2 option

**Question for Francesco:** OK with ~5–10 min inference time?

---

### Decision 5: Greeks Computation

**Options:**
- A: Analytical (closed-form, fast, limited to Black-Scholes)
- B: Numerical (finite differences, applies to any pricing model)
- C: Both

**Recommendation:** C (both)
- Analytical for BS (speed)
- Numerical for Merton (required)
- Good validation check

**Question for Francesco:** Approved?

---

## GitHub Repository Structure

```
bayesian-option-pricing/
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE (MIT)
│
├── src/
│   ├── __init__.py
│   ├── returns/
│   │   ├── __init__.py
│   │   ├── base.py           # DiffusionModel base class
│   │   ├── gaussian.py       # Gaussian model
│   │   ├── student_t.py      # Student-t model
│   │   ├── jump_diffusion.py # Merton model
│   │   └── test_*.py         # Unit tests
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── builder.py        # PyMC model assembly
│   │   ├── sampler.py        # NUTS sampling
│   │   ├── diagnostics.py    # Rhat, ESS, etc.
│   │   └── test_*.py         # Unit tests
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── black_scholes.py  # BS formula + Greeks
│   │   ├── merton.py         # Merton jump model
│   │   ├── bayesian.py       # Posterior propagation
│   │   └── test_*.py         # Unit tests
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── price_report.py   # Price + CI output
│   │   ├── greeks_report.py  # Greeks output
│   │   ├── scenario_report.py # Stress tests
│   │   └── test_*.py         # Unit tests
│   └── utils/
│       ├── __init__.py
│       ├── data.py           # Data loading
│       ├── returns.py        # Log-returns
│       └── validation.py     # Input validation
│
├── notebooks/
│   ├── 01_return_model_inference.ipynb
│   ├── 02_option_pricing_basics.ipynb
│   ├── 03_bayesian_option_pricing.ipynb
│   └── 04_interactive_trader.ipynb
│
├── docs/
│   ├── returns_models.md     # Mathematical spec
│   ├── bayesian_inference.md # Inference details
│   ├── option_pricing.md     # BS + Merton formulas
│   ├── bayesian_pricing.md   # Full Bayesian approach
│   ├── api_reference.md      # Function documentation
│   └── examples.md           # Use case walkthroughs
│
├── tests/
│   ├── __init__.py
│   ├── test_returns.py       # All return models
│   ├── test_inference.py     # PyMC builder + sampling
│   ├── test_pricing.py       # BS + Merton pricing
│   ├── test_bayesian.py      # Posterior propagation
│   └── integration_test.py   # End-to-end pipeline
│
└── examples/
    ├── sp500_crisis_pricing.py    # 2008 crisis analysis
    ├── vix_options_pricing.py     # Volatility products
    └── implied_vol_surface.py     # IV surface under regimes
```

---

## What Happens Next: Approval & Iteration

### Step 1: You Review This Document
- Does the mathematical spec match what you want?
- Are the technical decisions aligned with your vision?
- Any missing components or incorrect assumptions?
- Propose changes before we code

### Step 2: We Agree on Answers to Key Questions
1. Return models: Gaussian/Student-t/Jump? (recommend all three)
2. Regime switching: Include in Phase 1? (recommend Phase 2)
3. Vanilla-only pricing? (recommend yes)
4. NUTS sampling speed acceptable? (5–10 min per symbol)
5. Analytics + numerical Greeks? (recommend both)

### Step 3: Create GitHub Repo
Once approved, create: `github.com/oc-fmuia/bayesian-option-pricing`

### Step 4: Start Implementation
- Phase 1: Return model + inference (2 weeks)
- Phase 2: Pricing module (2 weeks)
- Phase 3: Documentation + notebooks (2 weeks)
- Each phase committed to GitHub with clean commits

### Step 5: Code Review Before Merging
- You review code against this spec
- If implementation diverges from spec, we pause and discuss
- **No silent assumptions, no hardcoding workarounds**

---

## Success Criteria

✅ **Phase 1 Complete When:**
- Return models (Gaussian, Student-t, Merton) correctly implemented
- NUTS inference working on real data (S&P 500, VIX, etc.)
- Diagnostics (Rhat, ESS) showing convergence
- 50+ unit tests, all passing
- Mathematical documentation written

✅ **Phase 2 Complete When:**
- Black-Scholes pricing matches scipy.stats
- Merton pricing validated against benchmarks
- Greeks (analytical + numerical) correct
- Posterior propagation to prices working
- Regime-conditional Greeks computed correctly
- 50+ unit tests, all passing

✅ **Phase 3 Complete When:**
- 4 educational notebooks executable + clear
- Full mathematical documentation with examples
- API reference with examples for all functions
- Type hints on all functions
- Real-world examples (2008, VIX, etc.) working

---

## Questions for Francesco

**Before we start coding, please approve/clarify:**

1. ✓ **Scope:** Does the above match what you envision?
2. ✓ **Return models:** Gaussian/Student-t/Jump or subset?
3. ✓ **Regime switching:** Phase 1 or Phase 2?
4. ✓ **Vanilla-only pricing:** Agreed?
5. ✓ **Inference speed:** 5–10 min per symbol acceptable?
6. ✓ **Greeks approach:** Analytical + numerical?
7. ✓ **Repository:** Ready to create `bayesian-option-pricing` repo?
8. ✓ **Timeline:** 6 weeks reasonable?
9. ✓ **Any missing components?**

Once you approve, I will:
- Create GitHub repo
- Commit this spec as `PROJECT_SPEC.md`
- Begin Phase 1 implementation
- **Report any technical difficulties immediately, not silently**

---

**Document Status:** Ready for review  
**Last Updated:** 2026-02-27
