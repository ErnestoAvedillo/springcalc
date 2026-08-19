"""Inverse design of a linearly-tapered (diameter and pitch) conical
compression spring that fits a full target force-vs-displacement curve,
instead of just two (length, force) points.

The point-based designer (conical_comp_inv_claude.py) assumes both given
points sit in the pre-contact linear regime, which lets the rate, free
length and safety factor be solved for algebraically. A full curve doesn't
get that assumption for free: its shape -- in particular the progressive
stiffening once coils start touching -- is exactly what reveals the taper,
so there's no closed form. Instead, wire diameter, free length, and the
start/end diameter and pitch are fit by regression: simulate each candidate
geometry's force-deflection curve and minimize its RMSE against the target
curve. The safety factor target enters the same objective as a soft
(weighted) term rather than a hard constraint, since matching the curve
shape may leave no exact freedom to also hit it.

Regression uses scipy.optimize.differential_evolution (global, derivative-
free, handles the bounded 6-parameter search without needing gradients --
comparable in spirit to the CMA-ES search in src/tmp/spring_inverse.py's
general N-control-point version, but scipy-only, no extra dependency).

Why the search doesn't call CompressionSpringGeneral.simulate_progressive_
compression directly: that method (via get_h_theta_development) solves the
winding-angle <-> axial-position mapping with one fsolve call per
discretization point, because it has to support an arbitrary func_D/func_p.
Measured at only 30 points/20 steps -- already a coarse search resolution --
a single call took ~9 seconds, which would make a several-thousand-
evaluation regression take hours. But D(h) and p(h) are linear here, so that
mapping has a closed form (see _theta_h_closed_form): p(h) linear in h makes
theta(h) = integral 2*pi/p(h) dh a log, invertible with exp -- no root
solve needed. _fast_progressive_compression reuses that closed form together
with the same per-step contact-detection logic as the general method
(oblique coil-to-coil collision, floor contact, flexibility-weighted
cumulative deformation), so it is a faithful simulation of this specific
(linear-taper) geometry, just without paying for generality it doesn't need.
It's used only inside the search; the returned design is still built and
verified with the real, general CompressionSpringGeneral machinery, so
everything reported to the caller comes from the unmodified library code.
The optimizer's continuous wire diameter is snapped to the nearest standard
size and the remaining parameters are locally re-fit around it before that
final build.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pint import Quantity
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import differential_evolution, minimize

from .conical_comp_inv_claude import _pitch_integral, linear_profile
from .lineal_comp_inv_claude import _shear_stress
from ..lineal.constants import COMPRESSION_SPRING_END_TYPES, FORMING_TYPES
from ..lineal.generic_compression import CompressionSpringGeneral
from ..lineal.goodman import GoodmanAnalyzer, GoodmanData
from ..pymodels.material import Material
from ..pymodels.units import ureg
from ..pymodels.wire_characteristics import get_standard_wire_diameters


@dataclass
class CompressionCurveRequirements:
    material: Material
    security_factor: float
    csv_path: str


def load_target_curve(csv_path: str) -> tuple:
    """Read a `displacement,load` CSV (mm, N) and return sorted arrays.
    `displacement` is travel from the free (unloaded) length, i.e.
    free_length - length -- so it doesn't depend on the (not yet known)
    free length the way an absolute position would."""
    df = pd.read_csv(csv_path)
    missing = {"displacement", "load"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV at {csv_path} is missing required column(s): {sorted(missing)}")
    df = df.sort_values("displacement")
    return df["displacement"].to_numpy(dtype=float), df["load"].to_numpy(dtype=float)


def _theta_h_closed_form(pitch_start_mm: float, pitch_end_mm: float, free_length_mm: float,
                         num_points: int) -> tuple:
    """thetas and h(theta) for a linear pitch profile p(h) = pitch_start +
    slope*h, from theta(h) = integral_0^h 2*pi/p(s) ds = (2*pi/slope) *
    ln(p(h)/pitch_start), inverted to h(theta) = (pitch_start/slope) *
    (exp(slope*theta/(2*pi)) - 1). The slope~0 case (constant pitch) is the
    limit of that formula, theta = 2*pi*h/pitch_start."""
    slope = (pitch_end_mm - pitch_start_mm) / free_length_mm
    if abs(slope) < 1e-9:
        theta_max = 2 * np.pi * free_length_mm / pitch_start_mm
        thetas = np.linspace(0.0, theta_max, num_points)
        zs = pitch_start_mm * thetas / (2 * np.pi)
    else:
        theta_max = (2 * np.pi / slope) * np.log(pitch_end_mm / pitch_start_mm)
        thetas = np.linspace(0.0, theta_max, num_points)
        zs = (pitch_start_mm / slope) * (np.exp(slope * thetas / (2 * np.pi)) - 1.0)
    return thetas, np.clip(zs, 0.0, free_length_mm)


def _fast_progressive_compression(wire_diameter_mm: float, diameter_start_mm: float, diameter_end_mm: float,
                                  pitch_start_mm: float, pitch_end_mm: float, free_length_mm: float,
                                  shear_modulus_mpa: float, max_deflection_mm: float,
                                  steps: int, num_points: int) -> tuple:
    """Same contact-aware stepping algorithm as VariableLinealSpring.
    simulate_progressive_compression, but fed a closed-form (thetas, zs_free)
    instead of one numerically solved per point -- see module docstring."""
    thetas, zs_free = _theta_h_closed_form(pitch_start_mm, pitch_end_mm, free_length_mm, num_points)
    diameters = diameter_start_mm + (diameter_end_mm - diameter_start_mm) * zs_free / free_length_mm
    radii = diameters / 2.0

    dtheta = thetas[1] - thetas[0] if len(thetas) > 1 else 1.0
    points_per_turn = max(int(round((2 * np.pi) / dtheta)), 1) if dtheta > 0 else num_points

    deflection_step = max_deflection_mm / steps
    delta_y = np.zeros_like(zs_free)
    current_force = 0.0
    deflection_history = [0.0]
    force_history = [0.0]

    for step in range(1, steps + 1):
        is_active = np.ones_like(thetas)
        for i in range(len(thetas) - points_per_turn):
            i_sup = i + points_per_turn
            z_inf_actual = zs_free[i] - delta_y[i]
            z_sup_actual = zs_free[i_sup] - delta_y[i_sup]
            pz_actual = abs(z_sup_actual - z_inf_actual)
            delta_r = abs(radii[i_sup] - radii[i])
            if delta_r < wire_diameter_mm:
                pz_limit = np.sqrt(max(wire_diameter_mm**2 - delta_r**2, 0.0))
                if pz_actual <= pz_limit:
                    is_active[i:i_sup + 1] = 0.0

        z_actual = zs_free - delta_y
        touches_floor = z_actual <= 0.0
        touches_floor[0] = False
        is_active[touches_floor] = 0.0

        local_flexibility = (8 * diameters**3) / (shear_modulus_mpa * (wire_diameter_mm**4) * 2 * np.pi)
        active_flexibility = local_flexibility * is_active
        total_flex = np.trapezoid(active_flexibility, thetas)

        k_inst = float('inf') if total_flex <= 1e-9 else 1.0 / total_flex
        if k_inst != float('inf'):
            current_force += k_inst * deflection_step
            deformation_factor = active_flexibility / total_flex
            cumulative_deformation = cumulative_trapezoid(deformation_factor, thetas, initial=0.0)
            delta_y = delta_y + cumulative_deformation * deflection_step

        deflection_history.append(step * deflection_step)
        force_history.append(current_force)
        if k_inst == float('inf'):
            break

    return np.array(deflection_history), np.array(force_history)


@dataclass
class ConicalCurveInverseDesign:
    spring: CompressionSpringGeneral
    wire_diameter: Quantity
    diameter_start: Quantity
    diameter_end: Quantity
    pitch_start: Quantity
    pitch_end: Quantity
    free_length: Quantity
    nr_coils: float
    solid_length: Quantity
    safety_factor: float
    safety_factor_target: float
    curve_rmse: Quantity
    curve_rmse_relative: float
    target_displacement: np.ndarray
    target_load: np.ndarray
    simulated_displacement: np.ndarray
    simulated_load: np.ndarray

    @property
    def safety_factor_error(self) -> float:
        return self.safety_factor - self.safety_factor_target


class ConicalCurveCompressionSpringInverseDesigner:
    """Fit a linearly-tapered (diameter and pitch) compression spring's wire
    diameter, free length, and diameter/pitch profile to a full target
    force-vs-displacement curve, with the safety factor target as a soft
    term in the same regression rather than a hard constraint."""

    def __init__(self, requirements: CompressionCurveRequirements,
                 type_of_end: str = COMPRESSION_SPRING_END_TYPES[1],
                 type_conforming: str = FORMING_TYPES[1],
                 spring_index_bounds: tuple = (4.5, 12.0),
                 wire_diameter_bounds: tuple = (0.3, 10.0),
                 diameter_bounds: tuple = (3.0, 150.0),
                 pitch_bounds: tuple = (0.3, 40.0),
                 free_length_margin: tuple = (1.05, 3.0),
                 min_coils: float = 2.0,
                 safety_factor_weight: float = 1.0,
                 penalty_weight: float = 0.05,
                 search_num_points: int = 60,
                 search_steps: int = 60,
                 final_num_points: int = 500,
                 final_steps: int = 500,
                 maxiter: int = 60,
                 popsize: int = 15,
                 seed: Optional[int] = None,
                 number_cycles: int = 1_000_000,
                 shot_peening: bool = False):
        self.material = requirements.material
        self.safety_factor_target = requirements.security_factor
        self.type_of_end = type_of_end
        self.type_conforming = type_conforming
        self.spring_index_bounds = spring_index_bounds
        self.wire_diameter_bounds = wire_diameter_bounds
        self.diameter_bounds = diameter_bounds
        self.pitch_bounds = pitch_bounds
        self.free_length_margin = free_length_margin
        self.min_coils = min_coils
        self.safety_factor_weight = safety_factor_weight
        self.penalty_weight = penalty_weight
        self.search_num_points = search_num_points
        self.search_steps = search_steps
        self.final_num_points = final_num_points
        self.final_steps = final_steps
        self.maxiter = maxiter
        self.popsize = popsize
        self.seed = seed
        self.number_cycles = number_cycles
        self.shot_peening = shot_peening
        self.shear_modulus_mpa = self.material.shear_modulus.to('MPa').magnitude

        self.target_displacement, self.target_load = load_target_curve(requirements.csv_path)
        if len(self.target_displacement) < 2:
            raise ValueError("The target curve must have at least two points")
        self._target_load_scale = max(float(np.mean(np.abs(self.target_load))), 1e-6)
        self._analyzer_cache: dict = {}

    def _analyzer(self, wire_diameter_mm: float) -> GoodmanAnalyzer:
        # RMa lookup only depends on wire diameter; cache analyzers across
        # the (many) evaluations at the same diameter within one search.
        cached = self._analyzer_cache.get(wire_diameter_mm)
        if cached is None:
            goodman_data = GoodmanData(material=self.material, diameter=wire_diameter_mm,
                                       load_type='torsion', cycles=int(self.number_cycles))
            cached = GoodmanAnalyzer(goodman_data, shot_peening=self.shot_peening)
            self._analyzer_cache[wire_diameter_mm] = cached
        return cached

    def _build_spring(self, wire_diameter_mm: float, diameter_start_mm: float, diameter_end_mm: float,
                      pitch_start_mm: float, pitch_end_mm: float, free_length_mm: float) -> CompressionSpringGeneral:
        spring = CompressionSpringGeneral(material=self.material, wire_diameter=wire_diameter_mm * ureg.mm)
        spring.number_cycles = self.number_cycles
        spring.shot_peening = self.shot_peening
        spring.set_geometry(
            func_D=linear_profile(diameter_start_mm, diameter_end_mm, free_length_mm),
            func_p=linear_profile(pitch_start_mm, pitch_end_mm, free_length_mm),
            free_length=free_length_mm * ureg.mm,
            type_of_end=self.type_of_end,
            type_conforming=self.type_conforming,
        )
        return spring

    def _geometry_penalty(self, wire_diameter_mm: float, diameter_start_mm: float, diameter_end_mm: float,
                          pitch_start_mm: float, pitch_end_mm: float, nr_coils: float,
                          solid_length_mm: float, shortest_length_mm: float) -> float:
        c_lo, c_hi = self.spring_index_bounds
        penalty = 0.0
        for diameter_mm in (diameter_start_mm, diameter_end_mm):
            spring_index = diameter_mm / wire_diameter_mm
            penalty += max(0.0, c_lo - spring_index) ** 2
            penalty += max(0.0, spring_index - c_hi) ** 2
        for pitch_mm in (pitch_start_mm, pitch_end_mm):
            penalty += max(0.0, wire_diameter_mm - pitch_mm) ** 2
        penalty += max(0.0, self.min_coils - nr_coils) ** 2
        penalty += max(0.0, solid_length_mm - shortest_length_mm) ** 2
        return penalty * self.penalty_weight

    def _evaluate(self, x, num_points: int, steps: int) -> float:
        wire_diameter_mm, free_length_mm, diameter_start_mm, diameter_end_mm, pitch_start_mm, pitch_end_mm = x
        try:
            deflection, force = _fast_progressive_compression(
                wire_diameter_mm, diameter_start_mm, diameter_end_mm, pitch_start_mm, pitch_end_mm,
                free_length_mm, self.shear_modulus_mpa, free_length_mm, steps, num_points)
            predicted_load = np.interp(self.target_displacement, deflection, force)
            rmse = float(np.sqrt(np.mean((predicted_load - self.target_load) ** 2)))
            norm_rmse = rmse / self._target_load_scale

            # Critical-section stress at the shortest (max target load) and
            # longest (min target load) working lengths, using the local
            # diameter and the load read off the simulated curve there --
            # matching how CompressionSpringGeneral.calculate_stress_at_
            # position evaluates it (see module docstring).
            length_hi = free_length_mm - float(self.target_displacement.max())
            length_lo = free_length_mm - float(self.target_displacement.min())
            load_hi = float(np.interp(self.target_displacement.max(), deflection, force))
            load_lo = float(np.interp(self.target_displacement.min(), deflection, force))
            diameter_hi = diameter_start_mm + (diameter_end_mm - diameter_start_mm) * length_hi / free_length_mm
            diameter_lo = diameter_start_mm + (diameter_end_mm - diameter_start_mm) * length_lo / free_length_mm
            stress_hi = _shear_stress(diameter_hi, wire_diameter_mm, load_hi)
            stress_lo = _shear_stress(diameter_lo, wire_diameter_mm, load_lo)
            safety_factor = self._analyzer(wire_diameter_mm).calculate_safety_factor(stress_hi, stress_lo)
            safety_factor_error = abs(safety_factor - self.safety_factor_target)

            nr_coils = (free_length_mm / pitch_start_mm) * _pitch_integral(pitch_end_mm / pitch_start_mm)
            diameter_spread = abs(diameter_start_mm - diameter_end_mm)
            fully_telescoped = diameter_spread >= nr_coils * wire_diameter_mm
            solid_length_mm = wire_diameter_mm if fully_telescoped else nr_coils * wire_diameter_mm
            penalty = self._geometry_penalty(wire_diameter_mm, diameter_start_mm, diameter_end_mm,
                                             pitch_start_mm, pitch_end_mm, nr_coils,
                                             solid_length_mm, length_hi)

            return norm_rmse + self.safety_factor_weight * safety_factor_error + penalty
        except Exception:
            return 1e6

    def design(self) -> ConicalCurveInverseDesign:
        max_displacement = float(self.target_displacement.max())
        margin_lo, margin_hi = self.free_length_margin
        bounds = [
            self.wire_diameter_bounds,
            (max_displacement * margin_lo, max_displacement * margin_hi),
            self.diameter_bounds,
            self.diameter_bounds,
            self.pitch_bounds,
            self.pitch_bounds,
        ]

        result = differential_evolution(
            lambda x: self._evaluate(x, self.search_num_points, self.search_steps),
            bounds, maxiter=self.maxiter, popsize=self.popsize, seed=self.seed,
        )
        (wire_diameter_mm, free_length_mm, diameter_start_mm,
         diameter_end_mm, pitch_start_mm, pitch_end_mm) = result.x

        # Snap to a manufacturable wire diameter, then locally re-fit the
        # remaining continuous parameters around it.
        standard_diameters = get_standard_wire_diameters()
        wire_diameter_mm = min(standard_diameters, key=lambda d: abs(d - wire_diameter_mm))

        def objective_fixed_diameter(y):
            return self._evaluate([wire_diameter_mm, *y], self.search_num_points, self.search_steps)

        refined = minimize(objective_fixed_diameter,
                           x0=[free_length_mm, diameter_start_mm, diameter_end_mm, pitch_start_mm, pitch_end_mm],
                           method='Nelder-Mead', bounds=bounds[1:])
        if refined.success:
            free_length_mm, diameter_start_mm, diameter_end_mm, pitch_start_mm, pitch_end_mm = refined.x

        spring = self._build_spring(wire_diameter_mm, diameter_start_mm, diameter_end_mm,
                                    pitch_start_mm, pitch_end_mm, free_length_mm)
        spring.calculate_spring_properties(num_points=self.final_num_points)
        deflection_history, force_history, _ = spring.simulate_progressive_compression(
            max_deflection=free_length_mm * ureg.mm, steps=self.final_steps, num_points=self.final_num_points)
        predicted_load = np.interp(self.target_displacement,
                                   deflection_history.to('mm').magnitude,
                                   force_history.to('N').magnitude)
        rmse = float(np.sqrt(np.mean((predicted_load - self.target_load) ** 2)))
        norm_rmse = rmse / self._target_load_scale

        length_hi = free_length_mm - max_displacement
        length_lo = free_length_mm - float(self.target_displacement.min())
        spring.add_load_position(length_hi * ureg.mm)
        spring.add_load_position(length_lo * ureg.mm)
        safety_factor = self._analyzer(wire_diameter_mm).calculate_safety_factor(
            spring.get_stress_max(), spring.get_stress_min())

        return ConicalCurveInverseDesign(
            spring=spring,
            wire_diameter=spring.wire_diameter,
            diameter_start=diameter_start_mm * ureg.mm,
            diameter_end=diameter_end_mm * ureg.mm,
            pitch_start=pitch_start_mm * ureg.mm,
            pitch_end=pitch_end_mm * ureg.mm,
            free_length=spring.free_length,
            nr_coils=spring.nr_coils,
            solid_length=spring.calculate_solid_length(),
            safety_factor=safety_factor,
            safety_factor_target=self.safety_factor_target,
            curve_rmse=rmse * ureg.N,
            curve_rmse_relative=norm_rmse,
            target_displacement=self.target_displacement,
            target_load=self.target_load,
            simulated_displacement=deflection_history.to('mm').magnitude,
            simulated_load=force_history.to('N').magnitude,
        )
