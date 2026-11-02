"""Phased adapter of BlackJAX ``mclmc_find_L_and_step_size`` (v1.7.1).

The three tuning stages run as three phases on the oracle-by-oracle stepper:

0. step size and preconditioner statistics (BlackJAX stages 1 and 2),
1. step-size readjustment after the preconditioner is set,
2. ``L`` from the autocorrelation of the collected samples (finalization).

Use ``make_baseline(num_steps)``; ``num_steps`` is the BlackJAX argument of the
same name (the planned number of production transitions).
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from blackjax.diagnostics import effective_sample_size
from blackjax.util import generate_unit_vector, incremental_value_update

from evaluator.interface import Result
from evaluator.mclmc_stepper import (
    Beta, advance, begin, init_state, STAGE_READY,
)


def tuning_steps(num_steps, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.1,
                 diagonal_preconditioning=True):
    """Number of transitions of each phase, following BlackJAX's rounding.

    Returns
    -------
    dict with ``n12`` (stages 1 and 2 together), ``n1``, ``n2``, ``n_readjust``, ``n3``.
    """
    n1, n2 = round(num_steps * frac_tune1), round(num_steps * frac_tune2)
    n3 = round(num_steps * frac_tune3)
    n_readjust = round(n2 / 3) if (n2 > 1 and diagonal_preconditioning) else 0
    return {"n1": n1, "n2": n2, "n12": n1 + n2, "n_readjust": n_readjust,
            "n3": n3 if n3 >= 2 else 0}


def make_baseline(num_steps, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.1,
                  desired_energy_var=5e-4, trust_in_estimate=1.5,
                  num_effective_samples=150, diagonal_preconditioning=True, l_factor=0.4):
    """Build the baseline submission for the given BlackJAX tuning settings.

    Returns
    -------
    SimpleNamespace with the submission entry points and ``num_evaluations``,
    the burn-in cost in gradient evaluations.
    """
    steps = tuning_steps(num_steps, frac_tune1, frac_tune2, frac_tune3, diagonal_preconditioning)
    lengths = (1 + 2 * steps["n12"], 2 * steps["n_readjust"], 2 * steps["n3"])
    decay = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def phase_fractions(config):
        if config.budget < sum(lengths):
            raise ValueError(f"baseline needs {sum(lengths)} evaluations, got {config.budget}")
        return tuple(n / config.budget for n in lengths)

    def init(x0, key, config):
        dim = config.dim
        init_key, tune_key = jax.random.split(key)
        part1_key, part2_key = jax.random.split(tune_key)
        keys12 = jax.random.split(part1_key, steps["n12"] + 1)
        beta = Beta(jnp.sqrt(dim) * jnp.ones(()), 0.25 * jnp.sqrt(dim) * jnp.ones(()),
                    jnp.ones(dim))
        beta = jax.tree.map(lambda a: jnp.asarray(a, x0.dtype), beta)
        stepper = init_state(x0, beta, init_key)
        return {
            "query_position": x0, "production_position": x0, "beta": beta,
            "stepper": stepper, "step_index": jnp.asarray(0),
            "keys": keys12[:-1] if steps["n12"] else keys12[:0],
            "readjust_key": keys12[-1], "keys3": jax.random.split(part2_key, max(steps["n3"], 1)),
            "adaptive": _zero_adaptive(x0), "streaming": _zero_streaming(dim, x0.dtype),
            "samples": jnp.zeros((max(steps["n3"], 1), dim), x0.dtype),
        }

    def _begin_next(carry, stepper, beta, index, keys, split_key=True):
        key = keys[jnp.minimum(index, keys.shape[0] - 1)]
        if split_key:  # the BlackJAX predictor splits off a NaN key first
            key = jax.random.split(key)[0]
        stepper = begin(stepper, beta, key)
        return dict(carry, stepper=stepper, query_position=stepper.query,
                    production_position=stepper.position, beta=beta, step_index=index)

    def adaptation_step(phase, carry, obs):
        prev = carry["stepper"]
        stepper, info = advance(prev, obs.logp, obs.grad, randomize_on_failure=False)
        done = info.completed
        beta = carry["beta"]
        dim = obs.position.shape[0]
        mask_start = steps["n1"] if phase == 0 else 0

        # predictor, using the step size that produced this transition
        success = info.nonans
        adaptive = carry["adaptive"]
        time, x_average, step_size_max = adaptive
        step_size_max = jnp.where(success, step_size_max, beta.step_size * 0.8)
        energy_change = jnp.where(success, info.energy_change, 0.0)
        xi = jnp.square(energy_change) / (dim * desired_energy_var) + 1e-8
        weight = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate)))
        x_average = decay * x_average + weight * (xi / jnp.power(beta.step_size, 6.0))
        time = decay * time + weight
        step_size = jnp.power(x_average / time, -1.0 / 6.0)
        step_size = (step_size < step_size_max) * step_size + (step_size > step_size_max) * step_size_max
        new_beta = beta._replace(step_size=step_size)

        position = jnp.nan_to_num(stepper.position)
        stepper = stepper._replace(
            position=jnp.where(success, position, stepper.position),
            momentum=jnp.where(success, jnp.nan_to_num(stepper.momentum), stepper.momentum),
            logdensity=jnp.where(success, jnp.nan_to_num(stepper.logdensity), stepper.logdensity),
            logdensity_grad=jnp.where(success, jnp.nan_to_num(stepper.logdensity_grad),
                                      stepper.logdensity_grad),
        )
        in_mask = (carry["step_index"] >= mask_start).astype(obs.position.dtype)
        streaming = incremental_value_update(
            jnp.array([stepper.position, jnp.square(stepper.position)]),
            carry["streaming"], weight=in_mask * success * step_size,
        )

        index = carry["step_index"] + done
        stepped = _begin_next(carry, stepper, new_beta, index, carry["keys"])
        stepped = dict(stepped, adaptive=(time, x_average, step_size_max), streaming=streaming)
        waiting = dict(carry, stepper=stepper, query_position=stepper.query)
        # init observation (stage READY without completion): start the first transition
        started = _begin_next(carry, stepper, beta, carry["step_index"], carry["keys"])
        return _select3(prev.stage == 0, started, done, stepped, waiting), {}

    def sampling_step(phase, carry, obs):
        stepper, info = advance(carry["stepper"], obs.logp, obs.grad, randomize_on_failure=False)
        done = info.completed
        sample_row = jnp.minimum(carry["step_index"], carry["samples"].shape[0] - 1)
        samples = carry["samples"].at[sample_row].set(
            jnp.where(done, stepper.position, carry["samples"][sample_row]))
        index = carry["step_index"] + done
        stepped = _begin_next(carry, stepper, carry["beta"], index, carry["keys"], split_key=False)
        waiting = dict(carry, stepper=stepper, query_position=stepper.query)
        return dict(_select(done, stepped, waiting), samples=samples), {}

    def step_phase(phase, carry, obs):
        if phase == 2:
            return sampling_step(phase, carry, obs)
        return adaptation_step(phase, carry, obs)

    def transition(carry, history, info):
        beta = carry["beta"]
        dim = beta.inverse_mass_matrix.shape[0]
        if info.phase == 0:
            if steps["n2"] > 1:
                _, average = carry["streaming"]
                variances = average[1] - jnp.square(average[0])
                L = jnp.sqrt(jnp.sum(variances))
                if diagonal_preconditioning:
                    beta = Beta(jnp.sqrt(dim) * jnp.ones(()).astype(L.dtype), beta.step_size, variances)
                else:
                    beta = beta._replace(L=L)
            keys = jax.random.split(carry["readjust_key"], max(steps["n_readjust"], 1))
            carry = dict(carry, keys=keys, adaptive=_zero_adaptive(beta.step_size),
                         streaming=_zero_streaming(dim, beta.step_size.dtype))
        else:
            carry = dict(carry, keys=carry["keys3"])
        if lengths[info.phase + 1] == 0:
            return dict(carry, beta=beta)
        return _begin_next(carry, carry["stepper"], beta, jnp.asarray(0), carry["keys"],
                           split_key=info.phase != 1)

    def finalize(carry, history, config):
        beta = carry["beta"]
        if steps["n3"] >= 2:
            ess = effective_sample_size(carry["samples"][None, ...])
            tau = steps["n3"] / ess
            finite = jnp.isfinite(tau)
            new_L = jnp.where(jnp.any(finite),
                              l_factor * beta.step_size * jnp.mean(tau, where=finite), beta.L)
            beta = beta._replace(L=new_L)
        return Result(carry["stepper"].position, beta)

    return SimpleNamespace(
        phase_fractions=phase_fractions, init=init, step_phase=step_phase,
        transition=transition, finalize=finalize, num_evaluations=sum(lengths),
        phase_lengths=lengths,
    )


def _zero_adaptive(like):
    dtype = jnp.asarray(like).dtype
    return (jnp.zeros((), dtype), jnp.zeros((), dtype), jnp.asarray(jnp.inf, dtype))


def _zero_streaming(dim, dtype):
    return (jnp.zeros((), dtype), jnp.zeros((2, dim), dtype))


def _select(pred, a, b):
    return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)


def _select3(is_init, started, done, stepped, waiting):
    chosen = _select(done, stepped, waiting)
    return _select(is_init, started, chosen)
