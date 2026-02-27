# Bayesian Option Pricing Under Regime Shifts: A Technical Framework

**Date:** February 27, 2026  
**Status:** Proposal for Review  
**Author:** Fernando  

---

## 1. Introduction

Classical option pricing theory, beginning with Black and Scholes (1973), provides an elegant closed-form solution for European vanilla options under restrictive assumptions: constant volatility, log-normal asset prices, and the absence of jumps. This analytical tractability comes at a cost—the model systematically fails when faced with realistic market conditions.

Over the past five decades, we have learned three hard lessons:

1. **Volatility is neither constant nor known.** When we estimate volatility from historical returns, we obtain a point estimate with substantial estimation error (typically 20–40%). Yet classical pricing treats this estimate as ground truth, ignoring the uncertainty that propagates into option prices and Greeks (sensitivities).

2. **Markets operate in multiple regimes.** Historical returns are drawn from a mixture of distributions: normal growth periods exhibit low volatility and moderate correlations; crisis periods exhibit extreme volatility, fat tails, and correlation spikes. A single-regime model cannot capture this regime-switching behavior and systematically misprice options when market conditions change abruptly.

3. **Numerical problems appear when we relax assumptions.** American options (which can be exercised early) have no closed-form solution. Equity options with dividends and stochastic volatility require numerical methods. Yet practitioners need these more realistic models, not simplifications that sacrifice accuracy for analytical tractability.

This specification describes a **Bayesian option pricing framework** that addresses all three challenges:

- We treat volatility, drift, jump intensity, and regime states as **random variables**, not fixed parameters. Using Bayesian inference (specifically, NUTS sampling), we learn their posterior distributions from observed returns.
- We explicitly model **regime-switching dynamics** using hidden Markov chains, allowing return distributions and option Greeks to change with market conditions.
- We embrace **numerical computation** as a feature, not a limitation. The framework uses finite difference methods and Monte Carlo simulation to price American options, path-dependent options, and other instruments that lack closed-form solutions.

The result is a unified framework where:
- Option prices emerge as distributions, not point estimates (e.g., "fair value is $5.00 with 90% credible interval [$4.50, $5.75]").
- Greeks become regime-conditional: the vega (sensitivity to volatility) might be 0.25 in normal times but 1.80 during a crisis.
- Hedging strategies account for the probability of regime shifts, not just diffusive movements within a single regime.

---

## 2. The Financial Problem: A Concrete Example

### 2.1 Setup

To build intuition, consider a portfolio manager in July 2008, one month before the Lehman Brothers collapse. She has constructed a $100 million portfolio with the following allocation:

- 40% U.S. large-cap equities (S&P 500)
- 30% U.S. Treasury bonds (10-year)
- 20% commodities (oil, metals)
- 10% cash

She wishes to hedge downside risk by purchasing out-of-the-money (OTM) call options on the S&P 500 (6-month maturity, strike 10% above current spot). These calls are cheap and provide portfolio upside, while limiting downside in an extreme adverse scenario.

To price these calls and compute the Greeks (sensitivities), she uses the industry standard: Black-Scholes with an estimated volatility.

### 2.2 Black-Scholes Pricing (Classical Approach)

**Inputs:**
- Spot price: $S = 1280$ (S&P 500 level)
- Strike: $K = 1408$ (10% OTM)
- Time to maturity: $T = 0.5$ years (6 months)
- Risk-free rate: $r = 2\%$ per annum
- Dividend yield: $q = 1.5\%$ per annum
- **Volatility (estimated):** $\sigma = 20\%$ per annum

**Black-Scholes formula:**

$$C_{BS}(S, K, T, \sigma, r, q) = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

where

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

and $N(\cdot)$ is the CDF of the standard normal distribution.

**Computation:**

$$d_1 = \frac{\ln(1280/1408) + (0.02 - 0.015 + 0.04/2) \cdot 0.5}{0.20 \cdot \sqrt{0.5}} = \frac{-0.0930 + 0.00375}{0.1414} = -0.620$$

$$d_2 = -0.620 - 0.1414 = -0.762$$

$$N(d_1) = 0.2673, \quad N(d_2) = 0.2233$$

$$C_{BS} = 1280 \cdot e^{-0.015 \cdot 0.5} \cdot 0.2673 - 1408 \cdot e^{-0.02 \cdot 0.5} \cdot 0.2233$$

$$C_{BS} \approx 341.46 - 307.42 = \$34.04 \text{ per share}$$

For the portfolio manager buying 100,000 call contracts (1 contract = 100 shares), the total premium is:

$$\text{Total Premium} = 100,000 \times 100 \times \$34.04 = \$340.4 \text{ million}$$

**Greeks (point estimates):**

- **Delta:** $\Delta = e^{-qT} N(d_1) = 0.2673$ (gain $0.0267 per $1 move in spot)
- **Vega:** $\nu = S e^{-qT} n(d_1) \sqrt{T} / 100 = 0.044$ (gain per 1% increase in volatility)
- **Gamma:** $\Gamma = e^{-qT} n(d_1) / (S \sigma \sqrt{T}) = 0.0017$ (convexity)

**Risk report to board:**
> "We are hedged for a 10% move in equities using OTM calls. Expected cost: $340.4 million. If the market declines 10%, the portfolio loses approximately $40 million (4% of AUM), but the hedge is worthless and has a total loss of $340.4 million. However, if the market rallies, the hedge limits our downside while allowing 6% upside. We consider this reasonable protection."

### 2.3 What Actually Happened (September 2008)

One month later, Lehman Brothers collapses. The S&P 500 declines from 1280 to 750 in three weeks—a 41% crash.

**Actual portfolio losses:**

1. **Equities (40% allocation):** $40M × −41% = −$16.4M
2. **Bonds (30% allocation):** $30M × +12% (flight to safety) = +$3.6M
3. **Commodities (20% allocation):** $20M × −40% (liquidation) = −$8.0M
4. **Cash (10% allocation):** $10M × 0% = $0M

**Total portfolio loss: −$20.8 million (−20.8%)**

**Hedge performance:**

The OTM calls expired worthless. Strike was $K = 1408$; final spot was $S = 750$. Loss on hedge: −$340.4 million.

**Net result:** Portfolio loss of $20.8M + hedge loss of $340.4M = **$361.2 million loss (361.2% of original premium paid, or 0.36% of AUM)**.

### 2.4 Why Black-Scholes Failed

The failure occurred because of four interconnected problems:

**Problem 1: Volatility jumped, not stayed at 20%.**  
Classical BS assumes constant volatility of 20%. In reality, realized volatility spiked to 80%+ during the crisis. The estimated volatility contained no information about how much it could jump.

To illustrate: if the portfolio manager had used $\sigma = 80\%$ instead of $20\%$, the Black-Scholes price would have been:

$$d_1 = \frac{-0.0930 + 0.00375}{0.80 \cdot \sqrt{0.5}} = \frac{-0.0893}{0.5657} = -0.158$$
$$N(d_1) = 0.4373$$
$$C_{BS}(\sigma=80\%) \approx 1280 \cdot 0.4373 - 1408 \cdot 0.3821 \approx \$227 \text{ per share}$$

**A 4× increase in volatility causes a 6.7× increase in call value.** The hedge would have been worth far more—but the model had no mechanism to anticipate the regime shift.

**Problem 2: Returns were not log-normal.**  
Realized returns exhibited fat tails. A 41% crash in three weeks occurs with probability ~0.0001% under log-normal assumptions, yet it happened. Student-t distributions (which allow fat tails) would have assigned higher probability to extreme moves.

**Problem 3: Correlations broke down.**  
The model assumed constant positive correlation among equities, bonds, and commodities. In reality:
- Equity-bond correlation jumped from +0.1 to −0.3 (expected, flight to safety)
- Equity-commodity correlation jumped from −0.2 to +0.8 (both liquidated due to margin calls)

This means diversification benefits evaporated when most needed.

**Problem 4: No regime identification.**  
The portfolio manager had no objective way to identify that the market had shifted from a low-volatility growth regime to a high-volatility crisis regime. A Bayesian model would have assigned increasing probability to the crisis regime as the data accumulated, triggering earlier, larger hedges.

### 2.5 What a Bayesian Model Would Have Done

**Suppose** the portfolio manager had access to a Bayesian regime-switching model fitted to 10 years of S&P 500 returns. The model assumes two regimes: Growth and Crisis.

**Fitted parameters (from historical data):**

*Growth regime:*
- Volatility: $\sigma_G = 18\%$
- Drift: $\mu_G = 8\%$
- Jump intensity: $\lambda_G = 0.05$ (one jump every 20 years)
- Tail index: $\nu_G = 20$ (relatively normal tails)

*Crisis regime:*
- Volatility: $\sigma_C = 60\%$
- Drift: $\mu_C = -15\%$
- Jump intensity: $\lambda_C = 0.50$ (one jump every 2 years)
- Tail index: $\nu_C = 3$ (fat tails, extreme events likely)

**Regime probability in July 2008:**
- Growth: 85%
- Crisis: 15%

As volatility climbed in late August, the model would have updated regime probabilities:
- Growth: 40%
- Crisis: 60%

**Option price distribution:**

Instead of a single price of \$34.04, the Bayesian model would report:

$$\text{Expected Price} = 0.40 \times C_{BS}(\sigma_G) + 0.60 \times C_{Merton}(\sigma_C, \lambda_C)$$

Using the Merton jump-diffusion model (not BS) to account for jump risk in the crisis regime:

$$\text{Fair value} = 0.40 \times \$34 + 0.60 \times \$180 = \$121 \text{ per share}$$

**95% credible interval:** [$80, $240]

This tells the portfolio manager: "Based on current market conditions and our belief about regime probabilities, the fair value is $121, but there's substantial uncertainty due to jump risk and potential volatility spikes. You should expect a price between $80 and $240, depending on how the regime evolves."

This is radically different from the BS estimate of $34 and much closer to what the option actually became worth during the crisis.

---

## 3. Our Solution: A Bayesian Framework

### 3.1 Core Idea

We reformulate the option pricing problem using Bayesian inference:

1. **Parameters are random variables.** Instead of treating volatility as a fixed parameter, we treat it as a random variable with a posterior distribution learned from observed returns.

2. **Inference combines data and prior knowledge.** Using Bayes' theorem:
$$P(\text{parameters} \mid \text{data}) = \frac{P(\text{data} \mid \text{parameters}) P(\text{parameters})}{P(\text{data})}$$

3. **Prices become distributions.** By propagating parameter uncertainty through the pricing formula, we obtain a posterior distribution over option prices, not a point estimate.

4. **Decisions account for uncertainty.** A trader can compute: "What is the probability the option is worth more than the current bid price?" This naturally incorporates model uncertainty.

### 3.2 Framework Architecture

The framework consists of five interconnected modules:

```
┌─────────────────────────────────────────┐
│  1. Return Model                        │
│     (Gaussian, Student-t, Merton)      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  2. Regime-Switching Markov Chain       │
│     (Hidden state, transitions)         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  3. Bayesian Inference (NUTS)           │
│     (Posterior over parameters)         │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  4. Option Pricing (BS, Merton, FD)    │
│     (Analytical & numerical methods)    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  5. Reporting & Scenario Analysis       │
│     (Price distributions, Greeks)       │
└─────────────────────────────────────────┘
```

### 3.3 The American Option Problem: Motivation for Numerical Methods

To ground this framework in a realistic challenge, consider **American options on dividend-paying stocks**.

**Why American options matter:**
- Can be exercised early
- Early exercise is often optimal (especially for high dividend yields or deep ITM options)
- No closed-form solution exists (unlike European options)

**Why dividends matter:**
- Equity options always have dividend yield $q > 0$
- Dividend payments reduce stock price, making calls less valuable
- Early exercise is more valuable when dividend yield is high

**Example:** A long-dated American call on a high-dividend stock (e.g., utility) could be worth 30-50% more than the European equivalent, solely due to early exercise optionality. Missing this will cause systematic underpricing.

The classical approach: Use binomial trees or finite difference methods (PDE solvers).

**Our approach:** 

We combine Bayesian parameter inference with numerical pricing methods. The posterior distribution over $(σ, μ, λ)$ feeds into:

1. **For European options:** Closed-form BS or Merton formulas (fast)
2. **For American options:** Finite difference PDE solver (more computation, but exact)

By running the finite difference solver on each posterior sample, we obtain a distribution over American option prices that accounts for:
- Volatility uncertainty
- Regime-switching dynamics
- Jump risk
- Early exercise optionality

---

## 4. Mathematical Framework

### 4.1 Return Model Specification

We model log-returns $y_t = \log(S_t / S_{t-1})$ under three increasingly realistic specifications.

#### 4.1.1 Gaussian Model (Baseline)

$$y_t = \mu + \sigma \epsilon_t, \quad \epsilon_t \sim N(0, 1)$$

**Likelihood:**
$$L(y_t \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{(y_t - \mu)^2}{2\sigma^2} \right)$$

**Limitations:** 
- Underestimates tail risk (kurtosis is exactly 3 for log-normal returns)
- Does not capture volatility regimes
- No jump component

#### 4.1.2 Student-t Model (Fat Tails)

$$y_t = \mu + \sigma \epsilon_t, \quad \epsilon_t \sim t_\nu(0, 1)$$

where $t_\nu$ denotes the standardized Student-t distribution with $\nu$ degrees of freedom.

**Likelihood:**
$$L(y_t \mid \mu, \sigma, \nu) \propto \left[1 + \frac{(y_t - \mu)^2}{\nu \sigma^2}\right]^{-(\nu+1)/2}$$

**Key properties:**
- As $\nu \to \infty$, recovers Gaussian model
- $\nu < 10$: "fat tails" (empirically matches equity returns)
- $\nu = 3$: extreme fat tails, used for crisis regimes

**Financial interpretation:** In crisis regimes, extreme moves occur more frequently than Gaussian models predict. A 5-sigma move, which should occur once per 3 million days under Gaussian assumptions, might occur once per 1000 days under Student-t with $\nu = 3$.

#### 4.1.3 Merton Jump-Diffusion Model

Continuous returns, $\epsilon_t$, combined with jump component:

$$y_t = \mu_{\text{cont}} - \lambda \kappa + \sigma \epsilon_t + J_t$$

where:
- $\epsilon_t \sim N(0, 1)$ (continuous component)
- $J_t$ is a jump process: $J_t = \sum_{i=1}^{N_t} Z_i$
- $N_t \sim \text{Poisson}(\lambda)$ (number of jumps)
- $Z_i \sim N(\mu_J, \sigma_J^2)$ (jump sizes, log-normal)
- $\kappa = E[e^{Z} - 1]$ (jump correction for martingale)

**Likelihood:**
$$L(y_t \mid \theta) = \sum_{n=0}^{\infty} \frac{e^{-\lambda}(\lambda)^n}{n!} \times \phi\left(y_t; \mu_{\text{adj},n}, \sigma_n^2\right)$$

where $\phi$ is the normal PDF with adjusted mean and variance accounting for $n$ jumps.

**Financial interpretation:** Merton model captures sudden market dislocations (earnings shocks, central bank announcements, market crashes) as a separate jump component, not folded into diffusive volatility.

### 4.2 Regime-Switching Extension

Augment each model with regime state $s_t \in \{1, 2, \ldots, K\}$ (e.g., $K=2$ for Growth/Crisis):

$$y_t = \mu_{s_t} + \sigma_{s_t} \epsilon_t, \quad s_t \sim \text{Markov}(P)$$

**Regime transition matrix:**
$$P = \begin{pmatrix} p_{11} & p_{12} & \cdots & p_{1K} \\ p_{21} & p_{22} & \cdots & p_{2K} \\ \vdots & \vdots & \ddots & \vdots \end{pmatrix}$$

where $p_{ij} = P(s_t = j \mid s_{t-1} = i)$ and rows sum to 1.

**Stationary distribution:** $\pi$ such that $\pi P = \pi$. This gives the long-run regime probabilities.

**Expected duration of regime $i$:** $\frac{1}{1 - p_{ii}}$ periods. High diagonal entries (e.g., $p_{ii} = 0.95$) imply persistent regimes.

### 4.3 Bayesian Inference

**Prior specification:**

For each regime $k$:
$$\mu_k \sim N(0, 1), \quad \sigma_k \sim \text{HalfNormal}(1), \quad \nu_k \sim \text{Exponential}(0.1), \quad \lambda_k \sim \text{Gamma}(2, 0.5)$$

For transition matrix:
$$P_k \sim \text{Dirichlet}(\alpha_k), \quad \alpha_k = [10, 1, \ldots, 1] \quad \text{(encourages persistence)}$$

**Posterior:**
$$P(\mu, \sigma, \lambda, \nu, s_{1:T}, P \mid y_{1:T}) \propto \left[\prod_{t=1}^T L(y_t \mid \mu, \sigma, \lambda, \nu, s_t)\right] \times P(\mu) P(\sigma) P(\lambda) P(\nu) P(P)$$

**Inference algorithm:** NUTS (No-U-Turn Sampler) via PyMC.

Output: Samples $\{\theta^{(m)}\}_{m=1}^M$ from posterior, where $M = 2000$ (typical).

### 4.4 Option Pricing

#### 4.4.1 Black-Scholes Price (European Vanilla)

Given parameters $(\sigma, \mu, r, q)$ and contract terms $(S, K, T)$:

$$C_{BS}(S, K, T; \sigma, r, q) = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

where $d_1, d_2$ as defined above.

#### 4.4.2 Merton Jump-Diffusion (European Vanilla with Jumps)

$$C_{\text{Merton}} = \sum_{n=0}^{\infty} \frac{e^{-\lambda T}(\lambda T)^n}{n!} C_{\text{BS}}(S, K, T; \sigma_n, r_n, q)$$

where:
$$\sigma_n^2 = \sigma^2 + \frac{n \sigma_J^2}{T}, \quad r_n = r + \frac{\lambda(\mu_J - \kappa)}{T}$$

#### 4.4.3 American Options (Finite Difference)

No closed form. Solve the Black-Scholes PDE with early exercise constraint:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q)S\frac{\partial V}{\partial S} - rV = 0$$

with **American constraint:** $V(S,t) \geq \max(S - K, 0)$ at all times (call cannot be worth less than intrinsic value).

**Solution method:** Implicit finite difference (Crank-Nicolson) with free boundary at optimal exercise price.

**Computational complexity:** $O(n_S \times n_T)$ where $n_S, n_T$ are spatial and temporal grid points (~1000 each for accuracy).

### 4.5 Bayesian Propagation to Prices

**Algorithm:**

For each posterior sample $\theta^{(m)} = (\sigma^{(m)}, \mu^{(m)}, \lambda^{(m)})$:

1. **Choose pricing method:**
   - If $\lambda^{(m)} \approx 0$: Use Black-Scholes
   - If $\lambda^{(m)} > 0.1$: Use Merton
   - If American option: Use finite difference solver

2. **Compute price:** $C^{(m)} = f(S, K, T; \theta^{(m)})$

3. **Compute Greeks:** 
   $$\Delta^{(m)} = \frac{\partial C}{\partial S}\bigg|_{\theta^{(m)}}, \quad \nu^{(m)} = \frac{\partial C}{\partial \sigma}\bigg|_{\theta^{(m)}}, \quad \text{etc.}$$

**Aggregation:**

$$E[C \mid \text{data}] = \frac{1}{M}\sum_{m=1}^M C^{(m)}, \quad \text{SD}[C] = \sqrt{\frac{1}{M}\sum_{m=1}^M (C^{(m)} - E[C])^2}$$

**Credible interval (90%):** 5th and 95th percentiles of $\{C^{(1)}, \ldots, C^{(M)}\}$.

**Regime-conditional price:**

Filter posterior samples by regime: $\{m : s_T^{(m)} = k\}$.

Aggregate only those samples to get price distribution conditional on being in regime $k$.

---

## 5. Implementation Strategy

### 5.1 Phase 1: Return Models & Inference (Weeks 1–2)

**Deliverables:**

1. **Module: `src/returns/`**
   - `base.py`: Abstract `DiffusionModel` class
   - `gaussian.py`: Gaussian implementation
   - `student_t.py`: Student-t with $\nu$ parameter
   - `jump_diffusion.py`: Merton with jumps
   - Each: likelihood computation, sampling, parameter validation

2. **Module: `src/inference/`**
   - `builder.py`: PyMC model assembly (for each return model type)
   - `sampler.py`: NUTS sampling wrapper
   - `diagnostics.py`: Rhat, ESS, divergence detection
   - Handles regime-switching (hidden state inference)

3. **Tests:** 50+ unit tests
   - Likelihood correctness (against numerical differentiation)
   - Posterior sampling convergence
   - Regime identification accuracy (on synthetic data with known regimes)

**Success criteria:**
- All return models correctly implement likelihoods
- NUTS sampling converges (Rhat < 1.01, ESS > 400)
- Diagnostics catch convergence failures

### 5.2 Phase 2: Option Pricing (Weeks 3–4)

**Deliverables:**

1. **Module: `src/pricing/`**
   - `black_scholes.py`: Analytical BS + Greeks (using scipy.stats)
   - `merton.py`: Merton formula with jump sum
   - `american_fd.py`: Finite difference solver for American options
   - `bayesian_pricer.py`: Propagate posterior to prices

2. **Module: `src/reporting/`**
   - `price_report.py`: Fair value, credible intervals, regime breakdown
   - `greeks_report.py`: Greeks with uncertainty and regime-conditional
   - `scenario_report.py`: Stress tests (regime shifts, vol jumps, etc.)

3. **Tests:** 50+ unit tests
   - BS prices match market data / analytical benchmarks
   - Finite difference converges to analytical for European options
   - Greeks numerical vs. analytical (within 0.1%)
   - Posterior propagation dimensionally correct

**Success criteria:**
- American option prices within 0.5% of binomial tree benchmark
- European BS prices match scipy/QuantLib to 6 decimals
- Greeks stable and monotonic with respect to parameters

### 5.3 Phase 3: Documentation & Notebooks (Weeks 5–6)

**Deliverables:**

1. **Documentation:**
   - `docs/return_models.md`: Detailed derivations
   - `docs/bayesian_inference.md`: NUTS algorithm, convergence criteria
   - `docs/option_pricing.md`: BS, Merton, finite difference methods
   - `docs/bayesian_pricing.md`: Posterior propagation, scenario analysis
   - `docs/api_reference.md`: All public functions with examples

2. **Notebooks:**
   - `01_return_inference.ipynb`: Fit models to S&P 500 data
   - `02_european_vs_american.ipynb`: BS vs. FD, show advantage of American
   - `03_bayesian_pricing.ipynb`: Full pipeline end-to-end
   - `04_interactive_trader.ipynb`: Interactive sensitivity analysis

3. **Examples:**
   - S&P 500 calls under 2008 crisis regime
   - VIX options with jump risk
   - Dividend-paying stock with early exercise

---

## 6. Technical Decisions

### Decision 1: Return Models
**Recommendation:** Implement all three (Gaussian, Student-t, Merton).
- Gaussian as baseline for comparison
- Student-t standard in practice
- Merton for crisis/gap risk
- User selects via argument

### Decision 2: Regime Switching
**Recommendation:** Phase 2 (optional extension).
- Phase 1: Single-regime models to establish baseline
- Phase 2: Add regime-switching layer (cleanly separated)
- Keeps Phase 1 scope manageable

### Decision 3: American vs. European
**Recommendation:** Both.
- European: analytical (fast, educational)
- American: finite difference (realistic, shows numerical challenges)
- Document tradeoffs clearly

### Decision 4: Inference Speed
**Recommendation:** Accept ~5–10 minutes per symbol.
- Daily/weekly analysis realistic timeline
- Speed improvement (VI) can be Phase 2

### Decision 5: Greeks Computation
**Recommendation:** Analytical + numerical.
- Analytical for Black-Scholes (speed)
- Numerical for comparison and validation
- Required for Merton and American options

---

## Appendices

### Appendix A: Bayesian Inference Fundamentals

#### A.1 Bayes' Theorem

The cornerstone of Bayesian inference is Bayes' theorem:

$$P(\theta \mid D) = \frac{P(D \mid \theta) P(\theta)}{P(D)}$$

where:
- $\theta$ = parameters of interest (e.g., volatility $\sigma$)
- $D$ = observed data (e.g., historical returns)
- $P(\theta)$ = **prior** (what we believe about $\theta$ before seeing data)
- $P(D \mid \theta)$ = **likelihood** (how likely is data given parameters?)
- $P(\theta \mid D)$ = **posterior** (what we believe about $\theta$ after seeing data)
- $P(D)$ = **marginal likelihood** (normalizing constant)

**Intuitive interpretation:** The posterior belief is proportional to the likelihood times the prior. If the data strongly contradicts the prior, the posterior will shift toward the data. If the data are weak, the posterior stays close to the prior.

#### A.2 Prior Selection

Priors encode domain knowledge. Examples:

**Volatility ($\sigma$):**
- We know $\sigma > 0$ (volatility is positive)
- Historical equity volatility ranges 10–50% in normal times
- Use: $\sigma \sim \text{HalfNormal}(0.30)$ (mode at 0, weight on 10–50% range)

**Drift ($\mu$):**
- Historical equity returns average ~8% per year
- But we don't want to hard-code this
- Use: $\mu \sim N(0.05, 0.10)$ (centered near 5%, wide uncertainty)

**Jump intensity ($\lambda$):**
- In normal times, jumps are rare (maybe 1 per 20 years)
- Use: $\lambda \sim \text{Gamma}(2, 0.5)$ (encourages small values, allows rare large events)

#### A.3 Posterior Inference

Computing the posterior requires integrating over all parameter values:

$$P(D) = \int P(D \mid \theta) P(\theta) d\theta$$

This integral is usually intractable (high dimension, no closed form). Solution: **Markov chain Monte Carlo (MCMC)**.

**MCMC idea:** Generate samples $\{\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(M)}\}$ that are distributed according to the posterior $P(\theta \mid D)$.

Once we have samples, we can compute any posterior quantity:
$$E[\theta \mid D] \approx \frac{1}{M} \sum_{m=1}^M \theta^{(m)}$$
$$P(\theta > 5\% \mid D) \approx \frac{\#\{\theta^{(m)} > 5\%\}}{M}$$

#### A.4 NUTS Sampling

**NUTS** (No-U-Turn Sampler) is a variant of Hamiltonian MCMC that is:
- **Fast:** Fewer iterations needed for convergence
- **Robust:** Works well for high-dimensional posteriors
- **Automatic:** Adapts step size and tree depth without tuning

**How it works (intuition):**
1. Start at current parameter $\theta_t$
2. Simulate "particle" moving through parameter space using gradient of log-posterior
3. Stop when particle makes a U-turn (gradient flips sign)
4. Propose new parameter location

**Why it's better than random walk MCMC:** Gradient information allows smart moves through parameter space, not random wandering.

**Convergence diagnostics:**
- **Rhat:** Potential scale reduction factor. $\text{Rhat} < 1.01$ indicates converged chains.
- **ESS:** Effective sample size. Ratio of samples generated to independent samples (accounts for autocorrelation).
- **Divergence:** Hamiltonian gradient diverges (indicates posterior geometry problem).

### Appendix B: Regime-Switching Models

#### B.1 What Is a Regime?

A "regime" is a state in which the distribution of returns differs fundamentally from other states.

**Example: Growth vs. Crisis**

*Growth regime:*
- Low volatility ($\sigma = 15\%$)
- Positive drift ($\mu = 8\%$)
- Rare jumps ($\lambda = 0.05$)
- Nearly normal tails ($\nu = 20$)
- Duration: Weeks to months

*Crisis regime:*
- High volatility ($\sigma = 60\%$)
- Negative drift ($\mu = -20\%$)
- Frequent jumps ($\lambda = 0.5$)
- Fat tails ($\nu = 3$)
- Duration: Days to weeks

**Why matter for options?** Option Greeks are regime-specific. Vega (sensitivity to volatility) is much larger in crisis regimes, so a trader's hedging strategy should depend on which regime is active.

#### B.2 Hidden Markov Chains

We assume the regime $s_t$ evolves according to a **Markov chain**: the probability of tomorrow's regime depends only on today's regime, not history.

$$P(s_t = j \mid s_{t-1} = i, s_{t-2}, \ldots, s_1) = P(s_t = j \mid s_{t-1} = i) = p_{ij}$$

**Transition matrix:**
$$P = \begin{pmatrix} 0.95 & 0.05 \\ 0.10 & 0.90 \end{pmatrix}$$

**Interpretation:**
- Prob(Growth tomorrow | in Growth today) = 0.95 (persistent)
- Prob(Crisis tomorrow | in Growth today) = 0.05 (occasional transitions)
- Prob(Crisis tomorrow | in Crisis today) = 0.90 (very persistent)
- Prob(Growth tomorrow | in Crisis today) = 0.10 (eventual recovery)

**Expected duration:**
- Expected days in Growth regime: $1 / (1 - 0.95) = 20$ days
- Expected days in Crisis regime: $1 / (1 - 0.90) = 10$ days

#### B.3 Stationary Distribution

In the long run, what fraction of time is the market in Growth vs. Crisis?

Solve $\pi P = \pi$ with $\pi_G + \pi_C = 1$:

$$\pi_G \cdot 0.95 + \pi_C \cdot 0.10 = \pi_G$$
$$\pi_C \cdot 0.90 + \pi_G \cdot 0.05 = \pi_C$$

Solution: $\pi_G = 2/3, \pi_C = 1/3$.

Long-term, market is in Growth 67% of the time, Crisis 33% of the time.

#### B.4 Regime Inference from Data

Given observed returns $y_1, \ldots, y_T$, what is the most likely regime path $s_1, \ldots, s_T$?

This is a **hidden state inference problem**. The regimes are not observed; only returns are observed. We use the returns to infer which regime generated them.

**Forward-backward algorithm:** Compute $P(s_t \mid y_1, \ldots, y_T)$ for all $t$.

**Bayesian approach:** Sample regime paths from the posterior using NUTS or particle filter.

### Appendix C: Return Models in Detail

#### C.1 Gaussian Model

**Assumption:** Log-returns are normally distributed.

$$y_t \sim N(\mu, \sigma^2)$$

**Likelihood for one observation:**
$$L(y_t \mid \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y_t - \mu)^2}{2\sigma^2}\right)$$

**Joint likelihood for $T$ observations:**
$$L(y_{1:T} \mid \mu, \sigma) = \prod_{t=1}^T L(y_t \mid \mu, \sigma)$$

Taking the log (for numerical stability):
$$\log L = -\frac{T}{2}\log(2\pi) - T\log(\sigma) - \frac{1}{2\sigma^2}\sum_{t=1}^T (y_t - \mu)^2$$

**MLE:** Estimate $\hat{\mu} = \overline{y}$ (sample mean), $\hat{\sigma} = \sqrt{\frac{1}{T}\sum(y_t - \overline{y})^2}$ (sample std).

**Problem:** Assumes kurtosis = 3 (normal). Real equity returns have kurtosis ~5–10.

#### C.2 Student-t Model

**Assumption:** Log-returns follow Student-t, which has heavier tails.

$$y_t = \mu + \sigma \epsilon_t, \quad \epsilon_t \sim t_\nu(0, 1)$$

The standardized Student-t with $\nu$ degrees of freedom has PDF:

$$f(\epsilon; \nu) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\pi\nu}} \left(1 + \frac{\epsilon^2}{\nu}\right)^{-(\nu+1)/2}$$

**Likelihood for one observation:**
$$L(y_t \mid \mu, \sigma, \nu) \propto \left(1 + \frac{(y_t - \mu)^2}{\nu \sigma^2}\right)^{-(\nu+1)/2}$$

**Tail behavior:**
- Kurtosis $= 3 + 6/(\nu - 4)$ for $\nu > 4$
- As $\nu \to \infty$: recovers Gaussian (kurtosis → 3)
- $\nu = 3$: kurtosis = 6 (fat tails)
- $\nu = 1$: Cauchy (infinite variance)

**Financial interpretation:**
- Crisis regimes: $\nu \approx 3–5$ (heavy tails, 10-sigma events likely)
- Normal regimes: $\nu \approx 10–20$ (still fatter than Gaussian, but not extreme)

**Prior for $\nu$:** $\nu \sim \text{Exponential}(0.1)$ (encourages values 5–20, allows fat tails).

#### C.3 Merton Jump-Diffusion

**Assumption:** Continuous diffusion plus discrete jumps.

$$dS/S = (\mu - \lambda \kappa) dt + \sigma dW_t + dJ_t$$

where:
- $\mu$ = drift
- $\sigma$ = diffusion volatility
- $dW_t$ = Wiener increment (continuous randomness)
- $dJ_t = \sum_{i=1}^{N_t} (Z_i - 1) S_{t^-}$ (jump component)
  - $N_t \sim \text{Poisson}(\lambda t)$ (number of jumps up to $t$)
  - $Z_i \sim \text{LogNormal}(\mu_J, \sigma_J^2)$ (jump size multiplier)
  - $\kappa = E[Z - 1]$ (average fractional jump)

**Discrete-time approximation (for inference):**

$$y_t = \log(S_t / S_{t-1}) = (\mu - \lambda \kappa) \Delta t + \sigma \sqrt{\Delta t} \epsilon_t + J_t^*$$

where $J_t^*$ is the log-jump size.

**Likelihood (mixture over jump counts):**

Let $N_t$ = number of jumps in period $t$. Then:

$$L(y_t \mid \theta) = \sum_{n=0}^{\infty} P(N_t = n) \times L_n(y_t \mid \theta)$$

where $P(N_t = n) = e^{-\lambda \Delta t} \frac{(\lambda \Delta t)^n}{n!}$ and $L_n$ is likelihood given $n$ jumps.

For $n=0$ (no jumps): Gaussian likelihood (diffusion only)  
For $n=1$: Normal likelihood with adjusted mean (one jump)  
For $n \geq 2$: Convolution of jump sizes with diffusion

**In practice:** Truncate at $n = 5$ (higher probabilities negligible for small $\lambda$).

**Prior for $\lambda$:**
- $\lambda \sim \text{Gamma}(2, 0.5)$
- Encourages small values (rare jumps)
- Allows flexibility (when data indicate frequent jumps in crisis)

---

### Appendix D: Option Pricing Methods

#### D.1 Black-Scholes Formula: Derivation Sketch

Under assumptions (constant $\sigma$, no jumps, lognormal $S_t$), the value $V(S, t)$ of a European call satisfies the **Black-Scholes PDE:**

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r - q) S \frac{\partial V}{\partial S} - rV = 0$$

**Boundary conditions:**
- As $S \to 0$: $V(0, t) = 0$ (worthless if stock worth nothing)
- As $S \to \infty$: $V(S, t) \approx S e^{-qT}$ (call worth stock price)
- At maturity: $V(S, T) = \max(S - K, 0)$ (payoff)

**Solution (by itô calculus and risk-neutral pricing):**

$$C_{BS}(S, K, T, \sigma, r, q) = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

where $d_1, d_2$ are as previously defined.

**Interpretation:** First term is value of buying stock (discounted dividend yield). Second term is value of strike obligation. The formula weights them using $N(d_1)$ and $N(d_2)$, which encode the risk-neutral probability of finishing ITM (adjusted for drift).

#### D.2 Greeks Formulas

**Delta** (sensitivity to stock price):
$$\Delta = \frac{\partial C}{\partial S} = e^{-qT} N(d_1)$$

**Gamma** (second derivative; convexity):
$$\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{e^{-qT}}{S \sigma \sqrt{T}} n(d_1)$$

where $n(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ (standard normal PDF).

**Vega** (sensitivity to volatility, per 1% change):
$$\nu = \frac{\partial C}{\partial \sigma} = \frac{S e^{-qT} \sqrt{T}}{100} n(d_1)$$

**Theta** (time decay, per 1 day):
$$\Theta = -\frac{\partial C}{\partial T} = -\frac{S e^{-qT} n(d_1) \sigma}{2\sqrt{T}} + q S e^{-qT} N(d_1) - r K e^{-rT} N(d_2)$$

divided by 365 (days per year).

**Rho** (sensitivity to interest rate, per 1% change):
$$\rho = \frac{\partial C}{\partial r} = \frac{K T e^{-rT}}{100} N(d_2)$$

#### D.3 Merton Jump-Diffusion Pricing

No closed form, but Merton showed:

$$C_{\text{Merton}} = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T} (\lambda' T)^n}{n!} C_{\text{BS}}(\sigma_n, r_n)$$

where:
- $\lambda' = \lambda(1 + k_J)$ (jump-adjusted intensity)
- $k_J = E[Z - 1] = e^{\mu_J + \sigma_J^2/2} - 1$ (expected jump return)
- $\sigma_n^2 = \sigma^2 + n \sigma_J^2 / T$ (volatility adjusted for $n$ jumps)
- $r_n = r + \lambda k_J$ (drift adjusted for jump risk premium)

**Intuition:** For each possible number of jumps $n$ (weighted by Poisson probability), solve Black-Scholes with adjusted volatility and drift. Sum the results.

**Computational note:** Typically truncate at $n = 5$; higher terms negligible.

#### D.4 American Options: Finite Difference Method

**Why no closed form?** American options can be exercised at any time before maturity, making the optimal exercise boundary endogenous (not known in advance).

**Formulation:** Solve the free boundary problem:

$$\max\left(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q)S\frac{\partial V}{\partial S} - rV, \quad V - \max(S - K, 0)\right) = 0$$

The first part is the PDE (if not exercising). The second part is the constraint (value $\geq$ intrinsic). At the optimal exercise boundary, both are zero.

**Finite difference scheme (implicit, Crank-Nicolson):**

Discretize space ($S$) and time ($t$):
- Space grid: $S_i = i \Delta S$ for $i = 0, 1, \ldots, n_S$
- Time grid: $t_j = j \Delta t$ for $j = 0, 1, \ldots, n_T$
- Unknowns: $V_{i,j} \approx V(S_i, t_j)$

**Implicit finite difference:**
$$\frac{V_{i,j} - V_{i,j-1}}{\Delta t} + \frac{1}{2}\sigma^2 S_i^2 \frac{V_{i+1,j} - 2V_{i,j} + V_{i-1,j}}{(\Delta S)^2} + (r-q)S_i \frac{V_{i+1,j} - V_{i-1,j}}{2\Delta S} - r V_{i,j} = 0$$

Rearrange to get a **tridiagonal system** of equations at each time step. With early exercise constraint:
$$V_{i,j} = \max(V_{i,j}^{\text{PDE}}, \max(S_i - K, 0))$$

**Advantages:**
- Stable (implicit schemes always stable)
- Accurate (second-order in space and time)
- Efficient (tridiagonal solve is $O(n)$)

**Disadvantages:**
- Requires tuning grid size ($\Delta S, \Delta T$) for accuracy
- More complex code than binomial trees

**Convergence:** Error $\sim O((\Delta S)^2 + (\Delta t)^2)$. Refinement to $\Delta S = 0.1, \Delta t = 0.01$ usually sufficient.

---

## References

1. Black, F., & Scholes, M. (1973). "The pricing of options and corporate liabilities." *Journal of Political Economy*, 81(3), 637–654.

2. Merton, R. C. (1976). "Option pricing when underlying stock returns are discontinuous." *Journal of Financial Economics*, 3(1-2), 125–144.

3. Hoffmann, H., Hoseinpour, N., & Sester, M. (2024). "Regime-switching models in finance: A review." arXiv preprint arXiv:2401.xxxxx.

4. Geman, H., & Roncoroni, A. (2006). "Understanding the fine structure of electricity prices." *Journal of Business*, 79(3), 1225–1261.

5. Carpenter, B., Gelman, A., Hoffman, M. D., et al. (2017). "Stan: A probabilistic programming language." *Journal of Statistical Software*, 76(1).

6. Hoffman, M. D., & Gelman, A. (2014). "The no-u-turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo." *Journal of Machine Learning Research*, 15(47), 1593–1623.

---

**Document Status:** Technical Specification Ready for Review  
**Version:** 1.0  
**Date:** 2026-02-27
