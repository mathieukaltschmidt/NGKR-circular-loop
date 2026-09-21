#!/usr/bin/env python3
"""Standalone matched NG+KR circular-loop solver."""
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
from scipy.integrate import simpson
HERE = Path(__file__).resolve().parent

#DON'T TOUCH IF YOU WANT TO REPROCE THE CORRESPONDING PAPER
DELTA_RHO_MS_FT = 1.0 / 0.472541132
MU0_OVER_V2_FT = 4.92789666

@dataclass(frozen=True)
class Matching:
    log_msR0: float
    delta_rho_ms: float = DELTA_RHO_MS_FT
    mu0_over_v2: float = MU0_OVER_V2_FT
    delta_sigma_over_delta_rho: float = 2.0
    R0: float = 1.0

    @property
    def msR0(self):
        return np.exp(self.log_msR0)

    @property
    def delta_rho(self):
        return self.delta_rho_ms * self.R0 / self.msR0

    @property
    def delta_sigma(self):
        return self.delta_sigma_over_delta_rho * self.delta_rho

    @property
    def delta_varphi(self):
        return self.delta_sigma / self.R0

    @property
    def kr_coupling(self):
        return 2.0 * np.pi / self.mu0_over_v2

    @property
    def log_inverse_delta_varphi(self):
        return -np.log(self.delta_varphi)

    @property
    def cutoff_invariant(self):
        return self.mu0_over_v2 - 2.0 * np.pi * np.log(self.delta_rho_ms)

@dataclass
class History:
    time: np.ndarray
    radius: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    tension: np.ndarray
    retarded_time_min: np.ndarray
    retarded_time_max: np.ndarray
    force_dynamic: np.ndarray
    force_static: np.ndarray
    size: int = 1

    @classmethod
    def allocate(cls, steps, time0, radius0, velocity0):
        arrays = [np.zeros(steps + 1) for _ in range(9)]
        history = cls(*arrays)
        history.time[0] = time0
        history.radius[0] = radius0
        history.velocity[0] = velocity0
        return history

    def as_legacy_array(self):
        s = slice(0, self.size)
        return np.array([self.time[s] - self.time[0], self.radius[s], self.velocity[s], self.acceleration[s], self.tension[s], self.force_dynamic[s], self.force_static[s], self.retarded_time_max[s], self.retarded_time_min[s]])

def retarded_force_kernel(expansion_power, beta, delta_varphi, time_step, radius_obs, velocity_obs, acceleration_obs, radius_ret, velocity_ret, acceleration_ret, time_obs, time_ret, enforce_angular_cutoff=True, evaluation_mask=None, include_local_counterterm=True):
    n = expansion_power
    to, t = (time_obs, np.asarray(time_ret))
    Ro, Rot, Rott = (radius_obs, velocity_obs, acceleration_obs)
    R, Rt, Rtt = (np.asarray(radius_ret), np.asarray(velocity_ret), np.asarray(acceleration_ret))
    light_cone_at_cutoff = (to - t) ** 2 - Ro ** 2 - R ** 2 + 2.0 * Ro * R * np.cos(delta_varphi)
    retarded_branch = to - t - Ro - R + time_step < 0
    mask = retarded_branch
    if enforce_angular_cutoff:
        mask &= light_cone_at_cutoff > 0
    if evaluation_mask is not None:
        mask &= np.asarray(evaluation_mask, dtype=bool)
    result = np.zeros_like(t, dtype=float)
    if not np.any(mask):
        return result
    den = -2 * (t - to) * R + R ** 2 * Rt + ((t - to) ** 2 - Ro ** 2) * Rt
    ExprSq = np.abs(1 - (-(t - to) ** 2 + R ** 2 + Ro ** 2) ** 2 / (4 * R ** 2 * Ro ** 2))
    Numer = -(2 * t ** n * to ** (-n) * (1 - Rot ** 2) ** (3 / 2) * (n * R * Rt * den ** 2 / to + beta * ((t - to) ** 2 + R ** 2 - Ro ** 2 + 2 * (to - t) * R * Rt) * den ** 2 / (2 * t * to) - Rt * (-(t - to) ** 2 - R ** 2 + Ro ** 2 + 2 * (t - to) * R * Rt) * den + n * R * ((t - to) ** 2 + R ** 2 - Ro ** 2 + 2 * (-t + to) * R * Rt) * den / t - 2 * (t - to) * R ** 2 * Rtt * den - R * ((t - to) ** 2 + R ** 2 - Ro ** 2 + 2 * (-t + to) * R * Rt) * (2 * R * (-1 + Rt ** 2) + R ** 2 * Rtt + ((t - to) ** 2 - Ro ** 2) * Rtt))) * (-t + to)
    if include_local_counterterm:
        Numer = Numer + ((1 / Ro + 2 * n * Rot / to) * (1 - Rot ** 2) + Rott) * (R * Ro * np.sqrt(ExprSq) * den ** 2)
    FullDen = R * Ro * np.sqrt(ExprSq) * den ** 2 * (-t + to)
    result[mask] = Numer[mask] / FullDen[mask]
    return result

def integrate_retarded_kernel_interpolated(expansion_power, beta, delta_varphi, time_step, radius_obs, velocity_obs, acceleration_obs, radius_ret, velocity_ret, acceleration_ret, time_obs, time_ret, include_local_counterterm=True):
    t = np.asarray(time_ret)
    if t.size < 2:
        return 0.0
    R = np.asarray(radius_ret)
    Rt = np.asarray(velocity_ret)
    Rtt = np.asarray(acceleration_ret)
    cutoff_function = (time_obs - t) ** 2 - radius_obs ** 2 - R ** 2 + 2 * radius_obs * R * np.cos(delta_varphi)
    branch_function = time_obs - t - radius_obs - R + time_step
    valid = (cutoff_function > 0) & (branch_function < 0)
    adjacent_to_valid = valid.copy()
    adjacent_to_valid[:-1] |= valid[1:]
    adjacent_to_valid[1:] |= valid[:-1]
    raw = retarded_force_kernel(expansion_power, beta, delta_varphi, time_step, radius_obs, velocity_obs, acceleration_obs, R, Rt, Rtt, time_obs, t, enforce_angular_cutoff=False, evaluation_mask=adjacent_to_valid, include_local_counterterm=include_local_counterterm)
    dt_cell = np.diff(t)
    both = valid[:-1] & valid[1:]
    integral = np.sum(0.5 * dt_cell[both] * (raw[:-1][both] + raw[1:][both]))
    transition = valid[:-1] != valid[1:]
    if np.any(transition):
        branch_transition = (branch_function[:-1] < 0) != (branch_function[1:] < 0)
        crossing_left = np.where(branch_transition, branch_function[:-1], cutoff_function[:-1])
        crossing_right = np.where(branch_transition, branch_function[1:], cutoff_function[1:])
        denominator = crossing_left - crossing_right
        usable = transition & (denominator != 0)
        fraction = np.zeros_like(denominator)
        fraction[usable] = np.clip(crossing_left[usable] / denominator[usable], 0.0, 1.0)
        crossing_value = raw[:-1] + fraction * (raw[1:] - raw[:-1])
        leaving = usable & valid[:-1]
        entering = usable & ~valid[:-1]
        integral += np.sum(0.5 * fraction[leaving] * dt_cell[leaving] * (raw[:-1][leaving] + crossing_value[leaving]))
        integral += np.sum(0.5 * (1 - fraction[entering]) * dt_cell[entering] * (crossing_value[entering] + raw[1:][entering]))
    return integral

def integrate_retarded_kernel_caustic_regularized(expansion_power, beta, delta_varphi, radius_obs, velocity_obs, acceleration_obs, radius_ret, velocity_ret, acceleration_ret, time_obs, time_ret, include_local_counterterm=True):
    t = np.asarray(time_ret)
    if t.size < 3:
        return 0.0
    R = np.asarray(radius_ret)
    Rt = np.asarray(velocity_ret)
    Rtt = np.asarray(acceleration_ret)
    far_branch = time_obs - t - radius_obs - R
    core_cutoff = (time_obs - t) ** 2 - radius_obs ** 2 - R ** 2 + 2 * radius_obs * R * np.cos(delta_varphi)
    valid = (far_branch < 0) & (core_cutoff > 0)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        return 0.0
    first, last = (valid_indices[0], valid_indices[-1])
    if np.any(~valid[first:last + 1]):
        raise RuntimeError('retarded integration domain is not contiguous')
    if first == 0:
        time_min = t[0]
    else:
        denominator = far_branch[first - 1] - far_branch[first]
        fraction = far_branch[first - 1] / denominator
        time_min = t[first - 1] + fraction * (t[first] - t[first - 1])
    evaluation_mask = valid.copy()
    if last + 1 < t.size:
        evaluation_mask[last + 1] = True
    raw = retarded_force_kernel(expansion_power, beta, delta_varphi, 0.0, radius_obs, velocity_obs, acceleration_obs, R, Rt, Rtt, time_obs, t, enforce_angular_cutoff=False, evaluation_mask=evaluation_mask, include_local_counterterm=include_local_counterterm)
    times = t[first:last + 1].copy()
    values = raw[first:last + 1].copy()
    if last + 1 < t.size and core_cutoff[last] * core_cutoff[last + 1] <= 0:
        denominator = core_cutoff[last] - core_cutoff[last + 1]
        fraction = core_cutoff[last] / denominator
        cutoff_time = t[last] + fraction * (t[last + 1] - t[last])
        cutoff_value = raw[last] + fraction * (raw[last + 1] - raw[last])
        times = np.append(times, cutoff_time)
        values = np.append(values, cutoff_value)
    separation = np.maximum(times - time_min, 0.0)
    positive = np.flatnonzero(separation > 0)
    if positive.size < 3:
        return 0.0
    first_positive = positive[0]
    fit_count = min(6, positive.size)
    fit_indices = positive[:fit_count]
    s_fit = separation[fit_indices]
    transformed = np.sqrt(s_fit) * values[fit_indices]
    degree = min(2, fit_count - 1)
    coefficients = np.polynomial.polynomial.polyfit(s_fit, transformed, degree)
    coefficients = np.pad(coefficients, (0, 3 - len(coefficients)))
    A, B, C = coefficients
    patch_end_index = fit_indices[-1]
    patch_end = separation[patch_end_index]
    endpoint_integral = 2 * A * np.sqrt(patch_end) + 2 / 3 * B * patch_end ** 1.5 + 2 / 5 * C * patch_end ** 2.5
    tail_times = times[patch_end_index:]
    tail_values = values[patch_end_index:]
    tail_integral = simpson(tail_values, x=tail_times) if len(tail_times) >= 3 else np.trapezoid(tail_values, tail_times)
    return endpoint_integral + tail_integral

def initial_matched_tension(matching):
    ds = matching.delta_varphi
    return 1.0 + matching.kr_coupling * np.log(np.sqrt(2.0 / (1.0 - np.cos(ds))))

def running_matched_tension(matching, infrared_time, radius, velocity):
    return 1.0 + matching.kr_coupling * (np.log(infrared_time) - np.log(radius / np.sqrt(1.0 - velocity ** 2)) + matching.log_inverse_delta_varphi)

def solve_acceleration(radius, velocity, time, force_a, force_b, tension, kr_coupling, expansion_power=0, release_factor=1.0, include_dq_boundary=False):
    if radius <= 0:
        raise RuntimeError('loop radius became non-positive')
    if abs(velocity) >= 1:
        raise RuntimeError('loop speed reached the speed of light')
    ng_term = (1.0 - velocity ** 2) * (2 * velocity * expansion_power / time + 1.0 / radius)
    dq_subtraction = float(include_dq_boundary)
    if not 0.0 <= dq_subtraction <= 1.0:
        raise ValueError('DQ boundary weight must lie between zero and one')
    dq_coefficient = kr_coupling * dq_subtraction
    free_acceleration = (kr_coupling * force_a / tension - ng_term + dq_coefficient * (1.0 - velocity ** 2) / (radius * tension)) / (1.0 - kr_coupling * force_b / tension + dq_coefficient / tension)
    return release_factor * free_acceleration

def smooth_release_factor(elapsed_time, release_duration):
    if release_duration <= 0 or elapsed_time >= release_duration:
        return 1.0
    if elapsed_time <= 0:
        return 0.0
    u = elapsed_time / release_duration
    return u * u * (3.0 - 2.0 * u)

def rk4_step(state, time, dt, acceleration_function):

    def derivative(y, t):
        radius, velocity = y
        return np.array([velocity, acceleration_function(radius, velocity, t)])
    k1 = derivative(state, time)
    k2 = derivative(state + dt * k1 / 2, time + dt / 2)
    k3 = derivative(state + dt * k2 / 2, time + dt / 2)
    k4 = derivative(state + dt * k3, time + dt)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def run_curve_readable(log_msR0, delta_rho_ms=DELTA_RHO_MS_FT, mu0_over_v2=MU0_OVER_V2_FT, delta_sigma_over_delta_rho=2.0, expansion_power=0, dt=0.0001, R0=1.0, velocity0=0.0, steps=17000, time0=3.0, integrator='frozen-rk4', corrector_iterations=2, quadrature='interpolated-endpoint', rhs_acceleration='implicit', include_local_counterterm=True, constant_tension=None, added_local_curvature=0.0, release_duration=0.0, include_dq_boundary=False, output_directory=HERE / 'runKR-readable', save=True):
    matching = Matching(log_msR0, delta_rho_ms, mu0_over_v2, delta_sigma_over_delta_rho, R0)
    beta = expansion_power * (expansion_power - 1) / 2
    history = History.allocate(steps, time0, R0, velocity0)
    history.retarded_time_min[0] = time0 - 2 * R0
    history.retarded_time_max[0] = time0 - R0 * np.sqrt(2 * (1 - np.cos(matching.delta_varphi)))
    history.tension[0] = initial_matched_tension(matching) if constant_tension is None else float(constant_tension)
    current_time = time0
    current_state = np.array([R0, velocity0], dtype=float)
    causal_time_min = history.retarded_time_min[0]
    if integrator not in {'frozen-rk4', 'predictor-corrector'}:
        raise ValueError(f'unknown integrator {integrator!r}')
    if quadrature not in {'masked-grid', 'interpolated-endpoint', 'caustic-regularized'}:
        raise ValueError(f'unknown quadrature {quadrature!r}')
    if rhs_acceleration not in {'implicit', 'previous-step'}:
        raise ValueError(f'unknown RHS acceleration prescription {rhs_acceleration!r}')
    added_local_curvature = float(added_local_curvature)

    def earliest_retarded_time(time_obs, radius_obs, number_of_history_points):
        static_cosine = (radius_obs ** 2 + R0 ** 2 - (time_obs - time0) ** 2) / (2 * radius_obs * R0)
        if abs(static_cosine) <= 1:
            return time_obs - radius_obs - R0
        s = slice(0, number_of_history_points)
        light_cone_test = time_obs - history.time[s] - radius_obs - history.radius[s]
        negative = np.flatnonzero(light_cone_test < 0)
        if not negative.size:
            return history.time[number_of_history_points - 1]
        upper = negative[0]
        if upper == 0:
            return history.time[0]
        lower = upper - 1
        f_lower = light_cone_test[lower]
        f_upper = light_cone_test[upper]
        fraction = f_lower / (f_lower - f_upper)
        return history.time[lower] + fraction * (history.time[upper] - history.time[lower])

    def integrate_force(observation_state, observation_time, earliest_time, trial_acceleration, number_of_history_points):
        radius_obs, velocity_obs = observation_state
        static_times = np.empty(0)
        static_radii = np.empty(0)
        static_velocities = np.empty(0)
        static_accelerations = np.empty(0)
        if earliest_time < time0:
            static_times = np.arange(earliest_time, time0, dt)
            static_radii = np.full_like(static_times, R0)
            static_velocities = np.full_like(static_times, velocity0)
            static_accelerations = np.zeros_like(static_times)
            if quadrature == 'masked-grid':
                static_integrand = retarded_force_kernel(expansion_power, beta, matching.delta_varphi, dt, radius_obs, velocity_obs, trial_acceleration, static_radii, static_velocities, static_accelerations, observation_time, static_times, include_local_counterterm=include_local_counterterm)
                static_force = np.trapezoid(static_integrand, static_times)
            else:
                static_force = 0.0
        else:
            static_force = 0.0
        s = slice(0, number_of_history_points)
        if quadrature == 'masked-grid':
            dynamic_integrand = retarded_force_kernel(expansion_power, beta, matching.delta_varphi, dt, radius_obs, velocity_obs, trial_acceleration, history.radius[s], history.velocity[s], history.acceleration[s], observation_time, history.time[s], include_local_counterterm=include_local_counterterm)
            dynamic_force = np.trapezoid(dynamic_integrand, history.time[s])
        else:
            joined_time = np.concatenate((static_times, history.time[s]))
            joined_radius = np.concatenate((static_radii, history.radius[s]))
            joined_velocity = np.concatenate((static_velocities, history.velocity[s]))
            joined_acceleration = np.concatenate((static_accelerations, history.acceleration[s]))
            if quadrature == 'interpolated-endpoint':
                dynamic_force = integrate_retarded_kernel_interpolated(expansion_power, beta, matching.delta_varphi, dt, radius_obs, velocity_obs, trial_acceleration, joined_radius, joined_velocity, joined_acceleration, observation_time, joined_time, include_local_counterterm=include_local_counterterm)
            else:
                dynamic_force = integrate_retarded_kernel_caustic_regularized(expansion_power, beta, matching.delta_varphi, radius_obs, velocity_obs, trial_acceleration, joined_radius, joined_velocity, joined_acceleration, observation_time, joined_time, include_local_counterterm=include_local_counterterm)
        return (static_force, dynamic_force)

    def affine_force(observation_state, observation_time, earliest_time, number_of_history_points):
        static0, dynamic0 = integrate_force(observation_state, observation_time, earliest_time, 0.0, number_of_history_points)
        static1, dynamic1 = integrate_force(observation_state, observation_time, earliest_time, 1.0, number_of_history_points)
        force_a = static0 + dynamic0
        force_b = static1 + dynamic1 - force_a
        return (force_a, force_b)
    static, dynamic = integrate_force(current_state, current_time, causal_time_min, 0.0, 1)
    history.force_static[0] = static
    history.force_dynamic[0] = dynamic
    for step in range(steps):
        previous_acceleration = history.acceleration[history.size - 1]

        def rhs_force_coefficients(observation_state, observation_time, earliest_time):
            if rhs_acceleration == 'implicit':
                force_a, force_b = affine_force(observation_state, observation_time, earliest_time, history.size)
            else:
                static_lagged, dynamic_lagged = integrate_force(observation_state, observation_time, earliest_time, previous_acceleration, history.size)
                force_a, force_b = (static_lagged + dynamic_lagged, 0.0)
            if added_local_curvature:
                radius, velocity = observation_state
                ng_term = (1.0 - velocity ** 2) * (2.0 * velocity * expansion_power / observation_time + 1.0 / radius)
                force_a += added_local_curvature * ng_term
                force_b += added_local_curvature
            return (force_a, force_b)
        force_a, force_b = rhs_force_coefficients(current_state, current_time, causal_time_min)
        tension = history.tension[history.size - 1]

        def acceleration(radius, velocity, time):
            release_factor = smooth_release_factor(time - time0, release_duration)
            return solve_acceleration(radius, velocity, time, force_a, force_b, tension, matching.kr_coupling, expansion_power, release_factor, include_dq_boundary)
        try:
            acceleration_start = acceleration(current_state[0], current_state[1], current_time)
            next_time = current_time + dt
            if integrator == 'frozen-rk4':
                next_state = rk4_step(current_state, current_time, dt, acceleration)
                acceleration_now = acceleration(next_state[0], next_state[1], current_time)
                next_causal_time_min = earliest_retarded_time(next_time, next_state[0], history.size)
                next_tension = running_matched_tension(matching, next_time - next_causal_time_min, next_state[0], next_state[1]) if constant_tension is None else float(constant_tension)
            else:
                next_state = np.array([current_state[0] + dt * current_state[1] + 0.5 * dt ** 2 * acceleration_start, current_state[1] + dt * acceleration_start])
                for _ in range(corrector_iterations):
                    next_causal_time_min = earliest_retarded_time(next_time, next_state[0], history.size)
                    next_tension = running_matched_tension(matching, next_time - next_causal_time_min, next_state[0], next_state[1]) if constant_tension is None else float(constant_tension)
                    next_a, next_b = rhs_force_coefficients(next_state, next_time, next_causal_time_min)
                    acceleration_end = solve_acceleration(next_state[0], next_state[1], next_time, next_a, next_b, next_tension, matching.kr_coupling, expansion_power, smooth_release_factor(next_time - time0, release_duration), include_dq_boundary)
                    corrected_velocity = current_state[1] + 0.5 * dt * (acceleration_start + acceleration_end)
                    corrected_radius = current_state[0] + 0.5 * dt * (current_state[1] + corrected_velocity)
                    next_state = np.array([corrected_radius, corrected_velocity])
                next_causal_time_min = earliest_retarded_time(next_time, next_state[0], history.size)
                next_tension = running_matched_tension(matching, next_time - next_causal_time_min, next_state[0], next_state[1]) if constant_tension is None else float(constant_tension)
                next_a, next_b = rhs_force_coefficients(next_state, next_time, next_causal_time_min)
                acceleration_now = solve_acceleration(next_state[0], next_state[1], next_time, next_a, next_b, next_tension, matching.kr_coupling, expansion_power, smooth_release_factor(next_time - time0, release_duration), include_dq_boundary)
        except RuntimeError as error:
            print(f'[log(m_s R0)={log_msR0:g}] {error}')
            break
        index = history.size
        current_state = next_state
        radius, velocity = current_state
        history.radius[index] = radius
        history.velocity[index] = velocity
        history.acceleration[index] = acceleration_now
        current_time = next_time
        history.time[index] = current_time
        history.size += 1
        causal_time_min = next_causal_time_min
        history.retarded_time_min[index] = causal_time_min
        delta_time = matching.delta_varphi * radius / np.sqrt(1.0 - velocity ** 2)
        history.retarded_time_max[index] = current_time - delta_time
        infrared_time = current_time - causal_time_min
        history.tension[index] = next_tension
        static, dynamic = integrate_force(current_state, current_time, causal_time_min, acceleration_now, history.size)
        history.force_static[index] = static
        history.force_dynamic[index] = dynamic
    output_directory = Path(output_directory)
    path = output_directory / f'KR_readable_{integrator}_{quadrature}_c{delta_sigma_over_delta_rho:g}_rhsacc-{rhs_acceleration}_DQboundary{float(include_dq_boundary):g}_trel{release_duration:g}_log{log_msR0:.1f}_dt{dt:g}.txt'
    if save:
        output_directory.mkdir(parents=True, exist_ok=True)
        header = f'log_msR0={log_msR0:g} delta_rho_ms={delta_rho_ms:g} mu0_over_v2={mu0_over_v2:g} delta_sigma_over_delta_rho={delta_sigma_over_delta_rho:g} delta_sigma={matching.delta_sigma:g} delta_varphi={matching.delta_varphi:g} integrator={integrator} quadrature={quadrature} rhs_acceleration={rhs_acceleration} local_counterterm={int(include_local_counterterm)} constant_tension={constant_tension} added_local_curvature={added_local_curvature:g} release_duration={release_duration:g}\ndq_boundary_weight={float(include_dq_boundary):g}\nrows: T R V A Mu h hS Tmax Tmin'
        np.savetxt(path, history.as_legacy_array(), header=header)
    return (matching, history, path)

def first_radius_crossing(history, target=0.5):
    radius = history.radius[:history.size]
    time = history.time[:history.size] - history.time[0]
    crossings = np.flatnonzero((radius[:-1] >= target) & (radius[1:] < target))
    if not len(crossings):
        return np.nan
    i = crossings[0]
    return time[i] + (target - radius[i]) * (time[i + 1] - time[i]) / (radius[i + 1] - radius[i])

# Final matched interface.  Everything above is private numerical machinery;
# no external jaxi module is imported.
C_FT = 0.034668629153333574

def kr_energy_integral(x):
    s = np.sqrt(1 + np.asarray(x, float) ** 2)
    return 2 * np.log((1 + s) / 2) - np.log(s)

@dataclass(frozen=True)
class NGKRMatching:
    log_msR0: float
    ms_delta_sigma: float = 1.0
    matching_scale: float = 0.25
    c_ft: float = C_FT
    R0: float = 1.0
    @property
    def delta_sigma(self): return self.ms_delta_sigma / np.exp(self.log_msR0)
    @property
    def Delta(self): return self.matching_scale
    @property
    def mu0_over_pi_fa2(self): return self.c_ft + np.log(4 * self.ms_delta_sigma)
    @property
    def mu_bar_exact_over_pi_fa2(self):
        return self.mu0_over_pi_fa2 + float(kr_energy_integral(self.Delta / self.delta_sigma))
    @property
    def mu_bar_asymptotic_over_pi_fa2(self):
        return self.c_ft + np.log(np.exp(self.log_msR0) * self.Delta)
    @property
    def matching_power_correction(self):
        return self.mu_bar_exact_over_pi_fa2 - self.mu_bar_asymptotic_over_pi_fa2
    def calF_over_pi_fa2(self, Freg, curvature):
        return Freg + float(kr_energy_integral(self.Delta / self.delta_sigma)) * curvature

def run_ngkr_final(log_msR0, ms_delta_sigma=1.0, matching_scale=.25,
        c_ft=C_FT, dt=1e-4, steps=17000, R0=1.0, velocity0=0.0,
        time0=3.0, integrator='predictor-corrector', corrector_iterations=2,
        quadrature='caustic-regularized', formulation='bare',
        output_directory=HERE / 'runNGKR-final', save=True,include_dq_boundary=False):
    m = NGKRMatching(log_msR0, ms_delta_sigma, matching_scale, c_ft, R0)
    if formulation not in {'bare', 'finite-delta'}: raise ValueError(formulation)
    exterior = float(kr_energy_integral(m.Delta / m.delta_sigma))
    tension = m.mu0_over_pi_fa2 if formulation == 'bare' else m.mu_bar_exact_over_pi_fa2
    _, h, _ = run_curve_readable(log_msR0, delta_rho_ms=1.0,
        delta_sigma_over_delta_rho=ms_delta_sigma, mu0_over_v2=2*np.pi,
        dt=dt, R0=R0, velocity0=velocity0, steps=steps, time0=time0,
        integrator=integrator, corrector_iterations=corrector_iterations,
        quadrature=quadrature, rhs_acceleration='previous-step',
        include_local_counterterm=False, constant_tension=tension,
        added_local_curvature=exterior if formulation == 'finite-delta' else 0,
        save=False,include_dq_boundary=include_dq_boundary)
    path = Path(output_directory) / (f'NGKR_log{log_msR0:g}_msds{ms_delta_sigma:g}_'
        f'Delta{matching_scale:g}_{formulation}_dt{dt:g}.txt')
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, h.as_legacy_array(), header='rows: T R V A Mu h hS Tmax Tmin')
    return m, h, path
