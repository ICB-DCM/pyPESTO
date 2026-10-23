"""Analytical relative inner solve for log/log10-normal PEtab-v1 data."""

from __future__ import annotations

import copy

import numpy as np

from ...C import LOG, LOG10, InnerParameterType
from ...objective.amici.amici_util import par_index_slices
from ..base_problem import scale_back_value_dict, scale_value_dict
from .solver import RelativeInnerSolver
from .util import apply_scaling, apply_sigma


class LogNormalInnerSolver(RelativeInnerSolver):
    """Profile positive multiplicative scales and log-scale normal noise.

    The density and simulations remain in the original observation units.
    Offset parameters, mixed linear/log relative data and partial pairings
    with estimated sigmas are explicitly rejected. Quantitative observables
    outside the relative inner problem are unaffected.
    """

    @staticmethod
    def validate_problem(problem):
        """Check the structural assumptions of this restricted inner solve."""
        for index, transform in enumerate(problem.observable_transformations):
            if any(mask[:, index].any() for mask in problem.data_mask):
                if transform not in (LOG, LOG10):
                    raise NotImplementedError(
                        "Mixed linear/log relative inner data are not supported."
                    )
        if problem.get_xs_for_type(InnerParameterType.OFFSET):
            raise NotImplementedError(
                "Log-normal inner offsets are not supported."
            )
        sigmas = problem.get_xs_for_type(InnerParameterType.SIGMA)
        for scale in problem.get_xs_for_type(InnerParameterType.SCALING):
            if scale.lb < 0 or scale.ub <= 0:
                raise ValueError(
                    "Log-normal scaling requires non-negative bounds and a positive upper bound."
                )
            overlapping = []
            for sigma in sigmas:
                if any(
                    np.any(s & q)
                    for s, q in zip(scale.ixs, sigma.ixs, strict=True)
                ):
                    overlapping.append(sigma)
                    if any(
                        np.any(s & ~q)
                        for s, q in zip(scale.ixs, sigma.ixs, strict=True)
                    ):
                        raise NotImplementedError(
                            "Every measurement of an inner scaling must share "
                            "the same estimated sigma, or all sigmas must be fixed."
                        )
            if len(overlapping) > 1:
                raise NotImplementedError(
                    "A log-normal scaling cannot share multiple inner sigmas."
                )
        for data, mask in zip(problem.data, problem.data_mask, strict=True):
            y = data[mask & ~np.isnan(data)]
            if np.any(~np.isfinite(y)) or np.any(y <= 0):
                raise ValueError(
                    "Log-normal observations must be finite and strictly positive."
                )

    @staticmethod
    def _terms(problem, data, sim, sigma, mask):
        observed = mask & ~np.isnan(data)
        y, h, sd = data[observed], sim[observed], sigma[observed]
        bases = np.broadcast_to(
            [
                np.log(10.0) if t == LOG10 else 1.0
                for t in problem.observable_transformations
            ],
            data.shape,
        )[observed]
        if (
            np.any(~np.isfinite(h))
            or np.any(h <= 0)
            or np.any(~np.isfinite(sd))
            or np.any(sd <= 0)
        ):
            raise ValueError(
                "Log-normal predictions and noise must be finite and strictly positive."
            )
        return observed, np.log(y), np.log(h), sd, bases

    def solve(self, problem, sim, sigma, scaled):
        """Compute bounded log scales, followed by their ML noise sigmas."""
        values = {}
        prediction = copy.deepcopy(sim)
        for parameter in problem.get_xs_for_type(InnerParameterType.SCALING):
            differences, noise = [], []
            for data_i, sim_i, sd_i, mask in zip(
                problem.data, sim, sigma, parameter.ixs, strict=True
            ):
                _, ly, lh, sd, base = self._terms(
                    problem, data_i, sim_i, sd_i, mask
                )
                differences.extend(ly - lh)
                noise.extend(base * sd)
            if not differences:
                raise ValueError(
                    "No observations for a log-normal inner scaling."
                )
            noise = np.asarray(noise)
            weight = (noise.min() / noise) ** 2
            log_scale = float(np.dot(weight, differences) / weight.sum())
            lower = np.log(parameter.lb) if parameter.lb > 0 else -np.inf
            log_scale = np.clip(log_scale, lower, np.log(parameter.ub))
            value = float(np.exp(log_scale))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "Log-normal inner scaling exceeds floating-point range."
                )
            values[parameter.inner_parameter_id] = value
            apply_scaling(value, prediction, parameter.ixs)
        for parameter in problem.get_xs_for_type(InnerParameterType.SIGMA):
            residuals = []
            for data_i, sim_i, sd_i, mask in zip(
                problem.data, prediction, sigma, parameter.ixs, strict=True
            ):
                _, ly, lh, _, base = self._terms(
                    problem, data_i, sim_i, sd_i, mask
                )
                residuals.extend((ly - lh) / base)
            if not residuals:
                raise ValueError(
                    "No observations for a log-normal inner sigma."
                )
            residuals = np.asarray(residuals)
            peak = np.max(np.abs(residuals))
            value = (
                float(peak * np.sqrt(np.mean((residuals / peak) ** 2)))
                if peak > 0
                else 0.0
            )
            if value <= 0 or not np.isfinite(value):
                raise ValueError(
                    "No finite positive log-normal sigma MLE (zero or non-finite residual variance)."
                )
            values[parameter.inner_parameter_id] = value
        return scale_value_dict(values, problem) if scaled else values

    @staticmethod
    def _apply(problem, sim, sigma, inner_parameters):
        prediction, noise = copy.deepcopy(sim), copy.deepcopy(sigma)
        values = scale_back_value_dict(inner_parameters, problem)
        for x in problem.get_xs_for_type(InnerParameterType.SCALING):
            apply_scaling(values[x.inner_parameter_id], prediction, x.ixs)
        for x in problem.get_xs_for_type(InnerParameterType.SIGMA):
            apply_sigma(values[x.inner_parameter_id], noise, x.ixs)
        return prediction, noise, values

    def calculate_obj_function(self, problem, sim, sigma, inner_parameters):
        """Compute the likelihood, including the observation Jacobian."""
        prediction, noise, _ = self._apply(
            problem, sim, sigma, inner_parameters
        )
        total = 0.0
        for data_i, sim_i, sd_i, mask in zip(
            problem.data, prediction, noise, problem.data_mask, strict=True
        ):
            _, ly, lh, sd, base = self._terms(
                problem, data_i, sim_i, sd_i, mask
            )
            residual = (ly - lh) / base
            total += np.sum(
                ly
                + np.log(base)
                + np.log(sd)
                + 0.5 * np.log(2 * np.pi)
                + 0.5 * (residual / sd) ** 2
            )
        return float(total)

    def calculate_gradients(
        self,
        problem,
        sim,
        ssim,
        sigma,
        ssigma,
        inner_parameters,
        parameter_mapping,
        par_opt_ids,
        par_sim_ids,
        snllh,
        index_slices=None,
    ):
        """Apply the envelope theorem at the exact constrained inner solution."""
        prediction, noise, values = self._apply(
            problem, sim, sigma, inner_parameters
        )
        sy = [np.moveaxis(np.array(s, copy=True), 1, 0) for s in ssim]
        ssd = [np.moveaxis(np.array(s, copy=True), 1, 0) for s in ssigma]
        for x in problem.get_xs_for_type(InnerParameterType.SCALING):
            for derivative, mask in zip(sy, x.ixs, strict=True):
                derivative[:, mask] *= values[x.inner_parameter_id]
        for x in problem.get_xs_for_type(InnerParameterType.SIGMA):
            for derivative, mask in zip(ssd, x.ixs, strict=True):
                derivative[:, mask] = 0.0
        if index_slices is None:
            index_slices = [
                par_index_slices(par_opt_ids, par_sim_ids, m.map_sim_var)
                for m in parameter_mapping
            ]
        for data_i, sim_i, sd_i, sy_i, ssd_i, mask, (is_sim, is_opt) in zip(
            problem.data,
            prediction,
            noise,
            sy,
            ssd,
            problem.data_mask,
            index_slices,
            strict=True,
        ):
            observed, ly, lh, sd, base = self._terms(
                problem, data_i, sim_i, sd_i, mask
            )
            residual = (ly - lh) / base
            dsim = -residual / (base * sd**2 * sim_i[observed])
            dnoise = (1 - (residual / sd) ** 2) / sd
            gradient = np.sum(
                sy_i[:, observed] * dsim + ssd_i[:, observed] * dnoise, axis=1
            )
            np.add.at(snllh, is_opt, gradient[is_sim])
        return snllh
