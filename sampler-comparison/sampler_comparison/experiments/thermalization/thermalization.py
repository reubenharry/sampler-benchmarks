"""
Thermalization-style comparison from Gaussian draws.

Plots use **∇ log π evaluation index** on the horizontal axis: each MCMC step's standardized
error (from ``get_standardized_squared_error``) is repeated ``num_grads_per_sample`` times via
``numpy.repeat`` (UMCLMC: 2; ULMC: 1; AHMC: ``num_integration_steps``), prefixed by the initial
error repeated once for the chain ``init`` gradient.

Each sampler runs **from the same initial positions** (per chain) with independent RNG afterward:

**UMCLMC / ULMC**: Robnik step-size tuning with CLI ``--target-eevpd`` (maps to ``desired_energy_var``).
Writes ``*_eevpd.png``: attained ``(ΔE)²/d`` vs that target.

**Adjusted HMC**: one vanilla HMC (MH) transition per step; step size follows **dual averaging**
targeting a fixed acceptance rate (``dual_averaging_adaptation`` in ``blackjax/adaptation/step_size.py``
— Hoffman & Gelman Nesterov-style scheme). Identity diagonal metric like the other chains.
Outer HMC steps are ``max(1, ⌊T / L⌋)`` where ``T`` is ``--num-samples`` and ``L`` is the leapfrog
count per proposal (``num_integration_steps``, ≈ ``√d`` here), so total inner ∇ evals from
transitions is about ``T``—comparable to ULMC's ``T`` single-gradient steps.

No separate warmup phase: tuning is online from iteration one alongside Robnik on the unadjusted
kernels.

Figures are written under ``experiments/thermalization/<model>/`` where ``<model>`` is a short
folder name (e.g. ``rosenbrock``, ``german_credit``, ``item_response``, ``brownian`` for the Brownian target).
Each PNG stem includes ambient dimension ``d``, Robnik ``eevpd``, ``T`` (``--num-samples``), and ``N``
(``--num-chains``): ``thermalization_<model>_d{D}_eevpd{...}_T{T}_N{N}_{error|scatter|eevpd}.png``.

CLI: ``python thermalization.py --help``.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


_PKG_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT.parent / "sampler-evaluation"))
sys.path.append("../../../../../src/inference-gym/spinoffs/inference_gym")
import os
print(os.listdir("../../../../../src/inference-gym/"))
# raise Exception("stop")
for _bj in (
    _PKG_ROOT.parent / "blackjax",
    Path("/global/homes/r/reubenh/blackjax"),
):
    if (_bj / "blackjax").is_dir():
        sys.path.insert(0, str(_bj))
        break

import blackjax
from blackjax.adaptation.step_size import dual_averaging_adaptation
from blackjax.adaptation.unadjusted_step_size import robnik_step_size_tuning
from blackjax.mcmc.integrators import isokinetic_mclachlan, velocity_verlet
from sampler_evaluation.evaluation.ess import get_standardized_squared_error
from sampler_comparison.samplers.general import initialize_model, make_log_density_fn

SamplerName = Literal["umclmc", "ulmc", "ahmc"]

DEFAULT_DIM = 100
DEFAULT_NUM_CHAINS = 500
DEFAULT_NUM_SAMPLES = 5000
# Robnik ``desired_energy_var``: target scale for squared energy change per dimension (EEVPD).
DEFAULT_TARGET_EEVPD = 5e-4

NUM_SCATTER_TIMEPOINTS = 10

# Dual averaging target for HMC (cf. Hoffman & Gelman; Stan-style defaults often ~0.65).
DA_TARGET_ACCEPTANCE = 0.65

RIBBON_PERCENTILES = (10.0, 90.0)

# Counts of ‖∇ log π‖ evaluations (via jax.value_and_grad on the target) per MCMC transition,
# matching Blackjax's ``generalized_two_stage_integrator`` / HMC trajectory usage here:
# UMCLMC — isokinetic McLachlan: two position updates per step → 2;
# ULMC — velocity Verlet: one position update per step → 1;
# AHMC — ``num_integration_steps`` velocity-Verlet substeps per MH proposal → ``num_integration_steps``.
INIT_LOGDENSITY_GRADS = 1  # each sampler's ``init`` runs value_and_grad once

ACTIVE_SAMPLERS: tuple[SamplerName, ...] = ("umclmc", "ulmc", "ahmc")

SAMPLER_DISPLAY = {
    "umclmc": "UMCLMC + Robnik",
    "ulmc": "ULMC + Robnik",
    "ahmc": "Adjusted HMC + DA",
}


def canonical_model_key(model_name: str) -> str:
    """Normalize ``--model`` strings for lookup (hyphens, legacy spellings)."""
    k = model_name.strip().lower().replace("-", "_")
    if k == "germancredit":
        return "german_credit"
    if k == "itemresponse":
        return "item_response"
    if k == "stochasticvolatility":
        return "stochastic_volatility"
    if k in ("brownian", "brownian_motion"):
        return "brownian_motion"
    return k


def figure_output_subdir(model_name: str) -> str:
    """Directory name under ``experiments/thermalization/`` for PNG outputs."""
    key = canonical_model_key(model_name)
    if key == "brownian_motion":
        return "brownian"
    return key


def model_display_label(model_name: str) -> str:
    """Human-readable target name for plot titles (matches CLI ``--model``)."""
    key = canonical_model_key(model_name)
    known = {
        "rosenbrock": "Rosenbrock",
        "banana": "Banana",
        "german_credit": "German credit",
        "item_response": "Item response",
        "brownian_motion": "Brownian motion",
        "stochastic_volatility": "Stochastic volatility",
    }
    return known.get(key, key.replace("_", " ").title())


def target_title_fragment(model_name: str, ndims: int) -> str:
    """Short ``Target Nd`` fragment for figure titles."""
    return f"{model_display_label(model_name)} {ndims}D"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare UMCLMC, ULMC (Robnik), and adjusted HMC (per-step dual averaging on ε) "
            "from Gaussian draws."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rosenbrock",
        help=(
            "Target model name. Supported: rosenbrock, banana, german_credit, "
            "item_response, stochastic_volatility, brownian (alias: brownian_motion). "
            "Rosenbrock uses --dim (even) as ambient dimension; other models ignore --dim."
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        metavar="D",
        help=(
            "Rosenbrock ambient dimension (even positive integer; "
            "target has dim/2 (x,y) blocks). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--num-chains",
        type=int,
        default=DEFAULT_NUM_CHAINS,
        metavar="N",
        help="Parallel chains per sampler. Default: %(default)s.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        metavar="T",
        help="MCMC transitions per chain (kernel + step-size adaptation each step for all samplers). "
        "Default: %(default)s.",
    )
    parser.add_argument(
        "--target-eevpd",
        type=float,
        default=DEFAULT_TARGET_EEVPD,
        metavar="V",
        help=(
            "Robnik target EEVPD (expected squared energy error per dimension): ``desired_energy_var`` "
            "for UMCLMC/ULMC tuning. Default: %(default)s."
        ),
    )
    args = parser.parse_args(argv)
    if canonical_model_key(args.model) == "rosenbrock" and (args.dim <= 0 or args.dim % 2 != 0):
        parser.error("--dim must be a positive even integer (for rosenbrock)")
    if args.num_chains < 1:
        parser.error("--num-chains must be >= 1")
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.target_eevpd <= 0:
        parser.error("--target-eevpd must be positive")
    return args


def load_model(model_name: str, *, dim: int):
    """
    Return an Inference Gym-style model object (JAX substrate) with:
      - model.ndims
      - model.sample_transformations (incl. 'square' with ground truths)
      - model.default_event_space_bijector
      - either model.log_density_fn or (model._unnormalized_log_prob + bijector jacobian)
    """
    key = canonical_model_key(model_name)
    if key == "rosenbrock":
        from sampler_evaluation.models.rosenbrock import Rosenbrock

        return Rosenbrock(D=dim // 2)
    if key == "banana":
        from sampler_evaluation.models.banana import banana

        return banana()
    if key == "german_credit":
        from sampler_evaluation.models.german_credit import german_credit

        return german_credit()
    if key == "item_response":
        from sampler_evaluation.models.item_response import item_response

        return item_response()
    if key == "stochastic_volatility":
        from sampler_evaluation.models.stochastic_volatility import stochastic_volatility

        return stochastic_volatility()
    if key == "brownian_motion":
        from sampler_evaluation.models.brownian import brownian_motion

        return brownian_motion()
    raise ValueError(f"Unknown model {model_name!r}")


def make_kernel_and_init(
    name: SamplerName,
    *,
    desired_energy_var: float,
):
    if name == "umclmc":
        kernel = blackjax.mclmc.build_kernel(isokinetic_mclachlan)
        init_fn = blackjax.mclmc.init
    elif name == "ulmc":
        kernel = blackjax.langevin.build_kernel(
            velocity_verlet,
            desired_energy_var_max_ratio=1e3,
            desired_energy_var=desired_energy_var,
        )
        init_fn = blackjax.langevin.init
    else:
        raise ValueError(f"unknown unadjusted sampler {name!r}")
    return kernel, init_fn


def build_run_robnik_chain_jitted(
    logdensity_fn,
    ndims: int,
    *,
    L: jax.Array,
    kernel,
    init_fn: Callable,
    desired_energy_var: float,
    initial_step_size: float = 0.05,
):
    inverse_mass_matrix = jnp.ones(ndims)

    robnik_init, robnik_update, _ = robnik_step_size_tuning(
        desired_energy_var=desired_energy_var
    )

    def step(carry, step_key):
        k_state, r_state = carry
        new_state, info = kernel(
            step_key,
            k_state,
            logdensity_fn,
            L,
            r_state.step_size,
            inverse_mass_matrix,
        )
        new_r = robnik_update(r_state, info)
        return (new_state, new_r), (new_state.position, info.energy_change)

    @partial(jax.jit, static_argnums=(1,))
    def run_chain(key: jax.Array, num_steps: int, position_init: jax.Array):
        mom_key, scan_parent = jax.random.split(key)
        kernel_state = init_fn(position_init, logdensity_fn, mom_key)
        robnik_state = robnik_init(initial_step_size, ndims)
        scan_keys = jax.random.split(scan_parent, num_steps)
        _, (positions, energy_changes) = jax.lax.scan(
            step,
            (kernel_state, robnik_state),
            scan_keys,
        )
        return positions, energy_changes

    return run_chain


def build_run_hmc_da_chain_jitted(
    logdensity_fn,
    ndims: int,
    *,
    num_integration_steps: int,
    target_acceptance_rate: float = DA_TARGET_ACCEPTANCE,
    initial_step_size: float = 0.05,
):
    """One HMC transition per step; ε updated via ``dual_averaging_adaptation`` from MH acceptance."""
    kernel = blackjax.hmc.build_kernel(velocity_verlet)
    inverse_mass_matrix = jnp.ones(ndims)
    da_init, da_update, _ = dual_averaging_adaptation(target_acceptance_rate)

    def step_fn(carry, step_key):
        hmc_state, da_state = carry
        step_size = jnp.exp(da_state.log_step_size)
        new_hmc_state, info = kernel(
            step_key,
            hmc_state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            num_integration_steps,
        )
        new_da_state = da_update(da_state, info.acceptance_rate)
        return (new_hmc_state, new_da_state), new_hmc_state.position

    @partial(jax.jit, static_argnums=(1,))
    def run_chain(key: jax.Array, num_steps: int, position_init: jax.Array):
        init_state = blackjax.hmc.init(position_init, logdensity_fn)
        da_state = da_init(initial_step_size)
        scan_keys = jax.random.split(key, num_steps)
        _carry, positions = jax.lax.scan(
            step_fn,
            (init_state, da_state),
            scan_keys,
        )
        return positions

    return run_chain


def standardized_error_initial_transformed(
    position: jax.Array,
    *,
    f: Callable[[jax.Array], jax.Array],
    E_f: jax.Array,
    Var_f: jax.Array,
) -> jax.Array:
    samples = position.reshape(1, 1, -1)
    return get_standardized_squared_error(
        samples,
        f=f,
        E_f=E_f,
        Var_f=Var_f,
        contract_fn=jnp.mean,
    )[0, 0]


def scatter_snapshot_steps(num_samples: int, num_points: int = NUM_SCATTER_TIMEPOINTS) -> np.ndarray:
    raw = np.linspace(0, num_samples, num=max(2, num_points))
    snaps = np.unique(np.clip(np.round(raw).astype(np.int64), 0, num_samples))
    snaps = np.unique(np.concatenate([snaps, np.array([0, num_samples])]))
    snaps.sort()
    return snaps


def num_grads_per_mcmc_step(name: SamplerName, *, hmc_num_integration_steps: int) -> int:
    if name == "umclmc":
        return 2
    if name == "ulmc":
        return 1
    return hmc_num_integration_steps


def cumulative_logdensity_grad_evals(
    name: SamplerName,
    num_transitions: int,
    *,
    hmc_num_integration_steps: int,
) -> int:
    """Total ∇ log π evals after ``num_transitions`` kernel updates (0 = only ``init``)."""
    if num_transitions < 0:
        raise ValueError("num_transitions must be non-negative")
    g = num_grads_per_mcmc_step(name, hmc_num_integration_steps=hmc_num_integration_steps)
    return INIT_LOGDENSITY_GRADS + g * num_transitions


def num_outer_mcmc_steps(
    name: SamplerName,
    num_samples: int,
    *,
    hmc_num_integration_steps: int,
) -> int:
    """Outer MCMC transitions for ``name`` given CLI ``--num-samples`` = ``T``.

    AHMC runs ``⌊T/L⌋`` proposals (at least 1), with ``L`` leapfrog steps each, so inner
    gradient work per chain scales like ``T`` rather than ``T·L``.
    """
    if name == "ahmc":
        return max(1, num_samples // hmc_num_integration_steps)
    return num_samples


def repeat_standardized_errors_along_grad_axis(
    errors_after_each_step: np.ndarray,
    init_errors: np.ndarray,
    *,
    num_grads_per_sample: int,
    init_grad_slots: int = INIT_LOGDENSITY_GRADS,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand per-MCMC-step errors to one value per gradient slot using ``np.repeat``.

    ``errors_after_each_step`` has shape ``(num_chains, num_samples)`` (post-transition errors).
    ``init_errors`` has shape ``(num_chains,)`` at position after ``init`` (∇ eval before step 1).

    Returns ``(expanded, x_1based)`` with ``expanded`` shape
    ``(num_chains, init_grad_slots + num_samples * num_grads_per_sample)`` and
    ``x_1based = arange(1, expanded.shape[1] + 1)``.
    """
    if init_errors.ndim != 1:
        raise ValueError("init_errors must be 1D (num_chains,)")
    init_block = np.repeat(init_errors[:, np.newaxis], init_grad_slots, axis=1)
    repeated = np.repeat(errors_after_each_step, num_grads_per_sample, axis=1)
    expanded = np.concatenate([init_block, repeated], axis=1)
    x = np.arange(1, expanded.shape[1] + 1, dtype=float)
    return expanded, x


def standardized_errors_along_chain(
    positions: jax.Array,
    *,
    f: Callable[[jax.Array], jax.Array],
    E_f: jax.Array,
    Var_f: jax.Array,
) -> jax.Array:
    samples = positions[None, :, :]
    return get_standardized_squared_error(
        samples,
        f=f,
        E_f=E_f,
        Var_f=Var_f,
        contract_fn=jnp.mean,
    )[0]


def get_windowed_standardized_squared_error(
    samples: jax.Array,
    *,
    f: Callable[[jax.Array], jax.Array],
    E_f: jax.Array,
    Var_f: jax.Array,
    window: int,
    contract_fn: Callable = jnp.mean,
) -> jax.Array:
    """
    Like ``get_standardized_squared_error`` but using a trailing window of fixed size.

    For t < window, uses all samples so far (1..t+1). For t >= window, uses the last
    ``window`` samples ending at t (inclusive).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    fx = f(samples)
    csum = jnp.cumsum(fx, axis=1)
    T = fx.shape[1]

    # sum over last `window` samples: csum[t] - csum[t-window]
    pad = jnp.zeros((fx.shape[0], 1, fx.shape[2]), dtype=csum.dtype)
    csum0 = jnp.concatenate([pad, csum], axis=1)  # (B, T+1, D)
    idx = jnp.arange(T)
    start = jnp.maximum(0, idx + 1 - window)
    end = idx + 1

    win_sum = csum0[:, end, :] - csum0[:, start, :]
    denom = jnp.minimum(window, idx + 1).astype(csum.dtype)  # (T,)
    exps = win_sum / denom[None, :, None]

    error_function = lambda x: contract_fn(jnp.square(x - E_f) / Var_f)
    return jax.vmap(jax.vmap(error_function))(exps)


def thermalization_figure_stem(
    *,
    model_name: str,
    ndims: int,
    target_eevpd: float,
    num_samples: int,
    num_chains: int,
) -> str:
    """Base filename stem including ambient dimension, Robnik target EEVPD, T and N."""
    ee_slug = np.format_float_scientific(target_eevpd, precision=4, exp_digits=2).replace(" ", "")
    ee_slug = ee_slug.replace(".e", "e").replace(".E", "E")
    return (
        f"thermalization_{model_name}_d{ndims}_eevpd{ee_slug}_T{num_samples}_N{num_chains}"
    )


def main(model_name: str, dim: int, num_chains: int, num_samples: int, *, target_eevpd: float):
    jax.config.update("jax_enable_x64", True)

    model = load_model(model_name, dim=dim)
    ndims = int(model.ndims)
    fig_stem = thermalization_figure_stem(
        model_name=model_name,
        ndims=ndims,
        target_eevpd=target_eevpd,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    exp_dir = Path(__file__).resolve().parent / figure_output_subdir(model_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    logdensity_fn = make_log_density_fn(model)
    trans = model.sample_transformations["square"]
    E_f = trans.ground_truth_mean
    Var_f = trans.ground_truth_standard_deviation**2

    def square_of_params(z):
        params = model.default_event_space_bijector(z)
        return trans.fn(params)

    L = jnp.sqrt(ndims)
    num_integration_steps = max(10, int(round(float(jnp.sqrt(ndims)))))
    num_hmc_outer_steps = num_outer_mcmc_steps(
        "ahmc",
        num_samples,
        hmc_num_integration_steps=num_integration_steps,
    )

    master = jax.random.key(42)
    n_samp = len(ACTIVE_SAMPLERS)
    split_keys = jax.random.split(master, n_samp + 1)
    pos_key = split_keys[0]
    sampler_keys = {name: split_keys[1 + i] for i, name in enumerate(ACTIVE_SAMPLERS)}

    chain_pos_keys = jax.random.split(pos_key, num_chains)
    if hasattr(model, "sample_init") and model.sample_init is not None:
        positions0 = jax.vmap(model.sample_init)(chain_pos_keys)
    else:
        positions0 = jax.vmap(lambda k: initialize_model(model, k))(chain_pos_keys)

    cmap = plt.cm.tab10.colors

    errors_by_sampler: dict[SamplerName, jax.Array] = {}
    errors_last100_by_sampler: dict[SamplerName, jax.Array] = {}
    energy_changes_robnik: dict[SamplerName, jax.Array] = {}
    sampler_colors: dict[SamplerName, tuple] = {}

    run_hmc_da = build_run_hmc_da_chain_jitted(
        logdensity_fn,
        ndims,
        num_integration_steps=num_integration_steps,
        target_acceptance_rate=DA_TARGET_ACCEPTANCE,
    )

    for i, name in enumerate(ACTIVE_SAMPLERS):
        sampler_colors[name] = cmap[i % len(cmap)]
        chain_keys = jax.random.split(sampler_keys[name], num_chains)

        if name == "ahmc":
            batched = jax.jit(
                jax.vmap(
                    lambda k, p: run_hmc_da(k, num_hmc_outer_steps, p),
                    in_axes=(0, 0),
                )
            )
            positions_batch = batched(chain_keys, positions0)
        else:
            kernel, init_fn = make_kernel_and_init(
                name, desired_energy_var=target_eevpd
            )
            run_chain = build_run_robnik_chain_jitted(
                logdensity_fn,
                ndims,
                L=L,
                kernel=kernel,
                init_fn=init_fn,
                desired_energy_var=target_eevpd,
            )
            batched = jax.jit(
                jax.vmap(lambda k, p: run_chain(k, num_samples, p), in_axes=(0, 0))
            )
            positions_batch, energy_changes_batch = batched(chain_keys, positions0)
            energy_changes_robnik[name] = energy_changes_batch

        errors_batch = jax.vmap(
            lambda traj: standardized_errors_along_chain(
                traj, f=square_of_params, E_f=E_f, Var_f=Var_f
            )
        )(positions_batch)
        errors_by_sampler[name] = errors_batch

        errors_last100_batch = get_windowed_standardized_squared_error(
            positions_batch,
            f=square_of_params,
            E_f=E_f,
            Var_f=Var_f,
            window=1000,
            contract_fn=jnp.mean,
        )
        errors_last100_by_sampler[name] = errors_last100_batch

    init_errors_np = np.asarray(
        jax.jit(
            jax.vmap(
                lambda pos: standardized_error_initial_transformed(
                    pos, f=square_of_params, E_f=E_f, Var_f=Var_f
                )
            )
        )(positions0)
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in ACTIVE_SAMPLERS:
        color = sampler_colors[name]
        errors_batch = errors_by_sampler[name]
        err_np = np.asarray(errors_batch)

        label = SAMPLER_DISPLAY[name]
        g_step = num_grads_per_mcmc_step(name, hmc_num_integration_steps=num_integration_steps)
        expanded, x_grad = repeat_standardized_errors_along_grad_axis(
            err_np,
            init_errors_np,
            num_grads_per_sample=g_step,
        )
        mean_err = np.mean(expanded, axis=0)
        low = np.percentile(expanded, RIBBON_PERCENTILES[0], axis=0)
        high = np.percentile(expanded, RIBBON_PERCENTILES[1], axis=0)
        m = np.maximum(mean_err, 1e-300)
        ax.semilogy(x_grad, m, lw=1.5, label=f"{label} (mean)", color=color)
        ax.fill_between(
            x_grad,
            np.maximum(low, 1e-300),
            np.maximum(high, 1e-300),
            alpha=0.22,
            color=color,
            linewidth=0,
        )

    max_grad_x = max(
        cumulative_logdensity_grad_evals(
            name,
            num_outer_mcmc_steps(
                name,
                num_samples,
                hmc_num_integration_steps=num_integration_steps,
            ),
            hmc_num_integration_steps=num_integration_steps,
        )
        for name in ACTIVE_SAMPLERS
    )
    ax.set_xlim(1.0, float(max_grad_x))
    ax.set_xscale("log")

    ax.set_xlabel(r"$\nabla \log \pi$ evaluation index")
    ax.set_ylabel(r"standardized squared error ($f=x^2$, mean over dims)")
    pct_lo, pct_hi = RIBBON_PERCENTILES
    ax.set_title(
        f"Thermalization — {target_title_fragment(model_name, ndims)} — "
        f"mean & {pct_lo:.0f}–{pct_hi:.0f}% band ({num_chains} chains)"
    )
    ax.legend(title=f"Shaded: {pct_lo:.0f}–{pct_hi:.0f}% across chains", fontsize=8)
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.15, which="minor")
    fig.tight_layout()
    out = exp_dir / f"{fig_stem}_error.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    plt.close(fig)

    fig_w, ax_w = plt.subplots(figsize=(10, 4.5))
    for name in ACTIVE_SAMPLERS:
        color = sampler_colors[name]
        errors_batch = errors_last100_by_sampler[name]
        err_np = np.asarray(errors_batch)

        label = SAMPLER_DISPLAY[name]
        g_step = num_grads_per_mcmc_step(name, hmc_num_integration_steps=num_integration_steps)
        expanded, x_grad = repeat_standardized_errors_along_grad_axis(
            err_np,
            init_errors_np,
            num_grads_per_sample=g_step,
        )
        mean_err = np.mean(expanded, axis=0)
        low = np.percentile(expanded, RIBBON_PERCENTILES[0], axis=0)
        high = np.percentile(expanded, RIBBON_PERCENTILES[1], axis=0)
        m = np.maximum(mean_err, 1e-300)
        ax_w.semilogy(x_grad, m, lw=1.5, label=f"{label} (mean)", color=color)
        ax_w.fill_between(
            x_grad,
            np.maximum(low, 1e-300),
            np.maximum(high, 1e-300),
            alpha=0.22,
            color=color,
            linewidth=0,
        )

    ax_w.set_xlim(1.0, float(max_grad_x))
    ax_w.set_xscale("log")
    ax_w.set_xlabel(r"$\nabla \log \pi$ evaluation index")
    ax_w.set_ylabel(r"standardized squared error ($f=x^2$, mean over dims)")
    pct_lo, pct_hi = RIBBON_PERCENTILES
    ax_w.set_title(
        f"Thermalization — {target_title_fragment(model_name, ndims)} — "
        f"rolling window (last 1000 samples) — "
        f"mean & {pct_lo:.0f}–{pct_hi:.0f}% band ({num_chains} chains)"
    )
    ax_w.legend(title=f"Shaded: {pct_lo:.0f}–{pct_hi:.0f}% across chains", fontsize=8)
    ax_w.grid(True, alpha=0.3, which="major")
    ax_w.grid(True, alpha=0.15, which="minor")
    fig_w.tight_layout()
    out_w = exp_dir / f"{fig_stem}_error_last1000.png"
    fig_w.savefig(out_w, dpi=150)
    print(f"wrote {out_w}")
    plt.close(fig_w)

    scatter_s = 5 if num_chains >= 400 else 14
    scatter_alpha = 0.22 if num_chains >= 400 else 0.45

    fig_scatter, axes_scatter = plt.subplots(
        1,
        len(ACTIVE_SAMPLERS),
        figsize=(4.8 * len(ACTIVE_SAMPLERS), 5),
        sharey=True,
        squeeze=False,
    )
    for ax_s, name in zip(axes_scatter[0], ACTIVE_SAMPLERS):
        err = np.asarray(errors_by_sampler[name])
        color = sampler_colors[name]
        outer_T = num_outer_mcmc_steps(
            name,
            num_samples,
            hmc_num_integration_steps=num_integration_steps,
        )
        snapshot_steps = scatter_snapshot_steps(outer_T)
        grad_snap = np.array(
            [
                cumulative_logdensity_grad_evals(
                    name,
                    int(s),
                    hmc_num_integration_steps=num_integration_steps,
                )
                for s in snapshot_steps
            ],
            dtype=float,
        )
        for si, step_num in enumerate(snapshot_steps):
            s = int(step_num)
            if s == 0:
                y = init_errors_np
            else:
                y = err[:, s - 1]
            x = np.full(num_chains, float(grad_snap[si]))
            ax_s.scatter(
                x,
                y,
                alpha=scatter_alpha,
                s=scatter_s,
                color=color,
                edgecolors="none",
                label=("per-chain" if si == 0 else None),
            )
        mean_snap = np.array(
            [
                float(init_errors_np.mean())
                if int(s) == 0
                else float(err[:, int(s) - 1].mean())
                for s in snapshot_steps
            ]
        )
        ax_s.plot(
            grad_snap,
            np.maximum(mean_snap, 1e-300),
            color=color,
            lw=2.4,
            zorder=6,
            label="mean",
        )
        ax_s.set_xlabel(r"$\nabla \log \pi$ evals")
        ax_s.set_title(SAMPLER_DISPLAY[name], fontsize=10)
        ax_s.set_yscale("log")
        ax_s.grid(True, alpha=0.3)
        ax_s.set_xticks(grad_snap, minor=False)
        ax_s.legend(loc="best", fontsize=7)

    axes_scatter[0, 0].set_ylabel(
        r"standardized squared error ($f=x^2$, mean over dims)"
    )
    fig_scatter.suptitle(
        f"Snapshots (~{NUM_SCATTER_TIMEPOINTS} MCMC indices per column); "
        f"UMCLMC/ULMC steps 0–{num_samples}, AHMC outer steps 0–{num_hmc_outer_steps} "
        f"({num_integration_steps} leapfrog / proposal); "
        f"x = cumulative ∇ log π evals (2/step UMCLMC, 1/step ULMC, {num_integration_steps}/step AHMC); "
        f"{num_chains} scatter markers per column (one chain each). "
        f"Robnik vs DA ε (δ={DA_TARGET_ACCEPTANCE:.2f}), {target_title_fragment(model_name, ndims)}",
        fontsize=10,
    )
    fig_scatter.tight_layout()
    out_scatter = exp_dir / f"{fig_stem}_scatter.png"
    fig_scatter.savefig(out_scatter, dpi=150)
    print(f"wrote {out_scatter}")
    plt.close(fig_scatter)

    robnik_names: tuple[SamplerName, ...] = ("umclmc", "ulmc")
    fig_ee, axes_ee = plt.subplots(
        1,
        len(robnik_names),
        figsize=(5.0 * len(robnik_names), 4.2),
        squeeze=False,
    )
    mcmc_steps_np = np.arange(1, num_samples + 1, dtype=float)
    pct_lo, pct_hi = RIBBON_PERCENTILES
    for ax_ee, name in zip(axes_ee[0], robnik_names):
        color = sampler_colors[name]
        ec = np.asarray(energy_changes_robnik[name])
        attained = (ec**2) / float(ndims)
        mean_a = np.mean(attained, axis=0)
        low_a = np.percentile(attained, pct_lo, axis=0)
        high_a = np.percentile(attained, pct_hi, axis=0)
        floor = 1e-300
        ax_ee.semilogy(mcmc_steps_np, np.maximum(mean_a, floor), color=color, lw=1.6, label="mean attained")
        ax_ee.fill_between(
            mcmc_steps_np,
            np.maximum(low_a, floor),
            np.maximum(high_a, floor),
            alpha=0.22,
            color=color,
            linewidth=0,
        )
        ax_ee.axhline(target_eevpd, color="0.15", ls="--", lw=1.4, label=f"target ({target_eevpd:g})")
        ax_ee.set_xlabel("MCMC step")
        ax_ee.set_ylabel(r"attained $(\Delta E)^2\,/\,d$")
        ax_ee.set_title(SAMPLER_DISPLAY[name], fontsize=10)
        ax_ee.grid(True, alpha=0.3)
        ax_ee.legend(loc="best", fontsize=8)
    fig_ee.suptitle(
        f"Robnik EEVPD — target = {target_eevpd:g} vs attained "
        f"(mean & {pct_lo:.0f}–{pct_hi:.0f}% across {num_chains} chains); "
        f"{target_title_fragment(model_name, ndims)}",
        fontsize=10,
    )
    fig_ee.tight_layout()
    out_ee = exp_dir / f"{fig_stem}_eevpd.png"
    fig_ee.savefig(out_ee, dpi=150)
    print(f"wrote {out_ee}")
    plt.close(fig_ee)


if __name__ == "__main__":
    _args = parse_args()
    main(
        model_name=_args.model,
        dim=_args.dim,
        num_chains=_args.num_chains,
        num_samples=_args.num_samples,
        target_eevpd=_args.target_eevpd,
    )
