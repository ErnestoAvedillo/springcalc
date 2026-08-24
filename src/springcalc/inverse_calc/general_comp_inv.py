"""Inverse design of a fully general compression spring -- an arbitrary
(not necessarily monotonic, not necessarily tapered) mean-diameter profile
D(h) and pitch profile p(h) -- that fits a full target force-vs-displacement
curve.

This is the general counterpart of ConicalCurveCompressionSpringInverseDesigner
(conical_curve_comp_inv.py). That designer restricts D(h) and p(h) to a
straight line between a start and end value, which gives the winding-angle
<-> axial-position mapping a closed form and makes its search two to three
orders of magnitude faster per evaluation (see its module docstring). Here
D(h) and p(h) are each a monotone cubic (PCHIP) spline through
`num_control_points` free values -- a strict superset of a linear taper (two
control points reduce to one), able to represent profiles that bulge, neck
down, or vary non-monotonically along the coil, e.g. a barrel or hourglass
shape, or a spring whose pitch tightens in the middle. A spline of that shape
has no closed-form h(theta), so every candidate the search evaluates is
simulated with the library's real, general machinery
(CompressionSpringGeneral.simulate_progressive_compression, which solves
h(theta) with one root-find per discretization point via
get_h_theta_development). That root-find is the dominant cost -- roughly
linear in `search_num_points` and essentially independent of `search_steps`,
since get_h_theta_development is called once per evaluation regardless of
step count -- so expect this search to take minutes to hours rather than the
seconds ConicalCurveCompressionSpringInverseDesigner needs. Prefer that
designer whenever a simple taper already fits the target curve; reach for
this one only when the curve's shape genuinely needs a non-taper profile.

As in ConicalCurveCompressionSpringInverseDesigner, wire diameter, free
length, and the diameter/pitch control values are fit by regression
(scipy.optimize.differential_evolution, global and derivative-free, followed
by a local Nelder-Mead polish once the wire diameter is snapped to the
nearest standard size), minimizing the RMSE between the candidate's
simulated curve and the target curve. The safety factor target enters the
same objective as a soft (weighted) term, and implausible geometry (spring
index out of bounds, pitch tighter than the wire, too few coils, a solid
length that already reaches the shortest working length) is penalized the
same way, evaluated at every control point rather than just two ends. A
cheap pre-check (a single quad integral for nr_coils, no per-point root
solve) lets grossly invalid candidates skip the expensive simulation
entirely, which matters a lot more here than in the linear-taper designer
since each simulation is so much more expensive.

The returned design is built and verified with the real, unmodified
CompressionSpringGeneral machinery, same as every other designer in this
package.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pint import Quantity
from scipy.interpolate import PchipInterpolator
from scipy.optimize import differential_evolution, minimize

from .conical_curve_comp_inv import CompressionCurveRequirements, load_target_curve
from .lineal_comp_inv import _shear_stress
from ..lineal.constants import COMPRESSION_SPRING_END_TYPES
from ..lineal.constants import OPEN_GROUND
from ..lineal.generic_compression import CompressionSpringGeneral
from ..lineal.goodman import GoodmanAnalyzer, GoodmanData
from ..pymodels.units import ureg
from ..pymodels.wire_characteristics import get_standard_wire_diameters


def spline_profile(control_h_mm: np.ndarray, control_values_mm: np.ndarray, free_length_mm: float):
    """A func_D/func_p-compatible closure following a monotone cubic (PCHIP)
    spline through (control_h_mm, control_values_mm). `h` is clamped to
    [0, free_length_mm] first, the same convention conical_comp_inv.
    linear_profile uses, so callers evaluating slightly outside the spring's
    axial extent get the boundary value instead of an extrapolated one.
    PCHIP (rather than a natural cubic spline) never overshoots between
    control points, which matters here since the control values are
    themselves the physical bounds (spring_index_bounds, pitch_bounds) the
    search is meant to respect everywhere, not just exactly at the points it
    directly controls."""
    spline = PchipInterpolator(control_h_mm, control_values_mm)

    def func(h):
        h_mm = min(max(h.to('mm').magnitude, 0.0), free_length_mm)
        return float(spline(h_mm)) * ureg.mm
    return func


@dataclass
class GeneralCurveInverseDesign:
    """The winning design: a fully built, verified `CompressionSpringGeneral`
    plus the fit quality (curve_rmse / curve_rmse_relative) and both curves
    for plotting/inspection. Unlike the tapered designers, the profile isn't
    summarized by a start/end pair -- diameter_control_points/
    pitch_control_points (at control_positions) are the values the search
    actually fit, and diameter_at/pitch_at evaluate the fitted spline at any
    axial position via the spring's own f_mean_diameter/f_pitch."""
    spring: CompressionSpringGeneral
    wire_diameter: Quantity
    free_length: Quantity
    diameter_control_points: Quantity
    pitch_control_points: Quantity
    control_positions: Quantity
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
        """Positive means the design is more conservative (safer) than
        requested; negative means it falls short of the target."""
        return self.safety_factor - self.safety_factor_target

    def diameter_at(self, h: Quantity) -> Quantity:
        """Fitted mean diameter at axial position h (0 at the base, up to
        free_length at the free end)."""
        return self.spring.f_mean_diameter(h)

    def pitch_at(self, h: Quantity) -> Quantity:
        """Fitted pitch at axial position h (0 at the base, up to
        free_length at the free end)."""
        return self.spring.f_pitch(h)


class GeneralCompressionSpringInverseDesigner:
    """Fit a compression spring's wire diameter, free length, and mean-
    diameter/pitch profiles -- each an arbitrary monotone-cubic spline
    through `num_control_points` control values, not restricted to a taper
    -- to a full target force-vs-displacement curve, with the safety factor
    target as a soft term in the same regression rather than a hard
    constraint."""

    def __init__(self, requirements: CompressionCurveRequirements,
                 type_of_end: str = COMPRESSION_SPRING_END_TYPES[OPEN_GROUND],
                 type_conforming: str = 'cold_formed',
                 num_control_points: int = 4,
                 spring_index_bounds: tuple = (4.5, 12.0),
                 wire_diameter_bounds: tuple = (0.3, 10.0),
                 diameter_bounds: tuple = (3.0, 150.0),
                 pitch_bounds: tuple = (0.3, 40.0),
                 free_length_margin: tuple = (1.05, 3.0),
                 min_coils: float = 2.0,
                 safety_factor_weight: float = 1.0,
                 penalty_weight: float = 0.05,
                 search_num_points: int = 20,
                 search_steps: int = 15,
                 final_num_points: int = 500,
                 final_steps: int = 500,
                 maxiter: int = 15,
                 popsize: int = 6,
                 polish_maxiter: int = 60,
                 seed: Optional[int] = None,
                 workers: int = 1,
                 number_cycles: int = 1_000_000,
                 shot_peening: bool = False):
        if num_control_points < 2:
            raise ValueError("num_control_points must be at least 2")
        self.material = requirements.material
        self.safety_factor_target = requirements.security_factor
        self.type_of_end = type_of_end
        self.type_conforming = type_conforming
        self.num_control_points = num_control_points
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
        self.polish_maxiter = polish_maxiter
        self.seed = seed
        self.workers = workers
        self.number_cycles = number_cycles
        self.shot_peening = shot_peening
        self.shear_modulus_mpa = self.material.shear_modulus.to('MPa').magnitude

        self.target_displacement, self.target_load = load_target_curve(requirements.csv_path)
        if len(self.target_displacement) < 2:
            raise ValueError("The target curve must have at least two points")
        # Scale RMSE by the typical target load magnitude so norm_rmse is a
        # unitless relative error, comparable across curves of very
        # different force ranges (and combinable with the safety-factor
        # error term in the same objective without one dominating on units
        # alone).
        self._target_load_scale = max(float(np.mean(np.abs(self.target_load))), 1e-6)
        self._analyzer_cache: dict = {}

    def _analyzer(self, wire_diameter_mm: float) -> GoodmanAnalyzer:
        """Build (or reuse) the Goodman fatigue analyzer for a wire
        diameter, cached across the many evaluations the search makes at
        the same candidate diameter."""
        cached = self._analyzer_cache.get(wire_diameter_mm)
        if cached is None:
            goodman_data = GoodmanData(material=self.material, diameter=wire_diameter_mm,
                                       load_type='torsion', cycles=int(self.number_cycles))
            cached = GoodmanAnalyzer(goodman_data, shot_peening=self.shot_peening)
            self._analyzer_cache[wire_diameter_mm] = cached
        return cached

    def _build_spring(self, wire_diameter_mm: float, free_length_mm: float,
                      diameter_control_mm: np.ndarray, pitch_control_mm: np.ndarray) -> CompressionSpringGeneral:
        """Construct the real, general-purpose CompressionSpringGeneral for
        a candidate geometry, with D(h)/p(h) as PCHIP splines through
        control points spread evenly over [0, free_length_mm]."""
        control_h_mm = np.linspace(0.0, free_length_mm, self.num_control_points)
        spring = CompressionSpringGeneral(material=self.material, wire_diameter=wire_diameter_mm * ureg.mm)
        spring.number_cycles = self.number_cycles
        spring.shot_peening = self.shot_peening
        spring.set_geometry(
            func_D=spline_profile(control_h_mm, diameter_control_mm, free_length_mm),
            func_p=spline_profile(control_h_mm, pitch_control_mm, free_length_mm),
            free_length=free_length_mm * ureg.mm,
            type_of_end=self.type_of_end,
            type_conforming=self.type_conforming,
        )
        return spring

    def _geometry_penalty(self, wire_diameter_mm: float, diameter_control_mm: np.ndarray,
                          pitch_control_mm: np.ndarray, nr_coils: float,
                          solid_length_mm: float, shortest_length_mm: float) -> float:
        """Soft constraints steering the search away from geometries that
        fit the curve well numerically but aren't sound springs, evaluated
        at every control point (not just two ends, since the profile isn't
        assumed monotonic): spring index in bounds, pitch never smaller than
        the wire, enough coils, and a solid length that doesn't already
        reach the shortest length the target curve exercises."""
        c_lo, c_hi = self.spring_index_bounds
        spring_index = diameter_control_mm / wire_diameter_mm
        penalty = float(np.sum(np.clip(c_lo - spring_index, 0.0, None) ** 2))
        penalty += float(np.sum(np.clip(spring_index - c_hi, 0.0, None) ** 2))
        penalty += float(np.sum(np.clip(wire_diameter_mm - pitch_control_mm, 0.0, None) ** 2))
        penalty += max(0.0, self.min_coils - nr_coils) ** 2
        penalty += max(0.0, solid_length_mm - shortest_length_mm) ** 2
        return penalty * self.penalty_weight

    def _evaluate(self, x, num_points: int, steps: int) -> float:
        """Objective function minimized by the search: how well one
        candidate (wire diameter, free length, diameter/pitch control
        values) reproduces the target curve, adjusted by how close it lands
        to the safety-factor target and penalized for implausible geometry.
        Lower is better; any failure is scored as a large constant so the
        optimizer treats it as clearly worse than any successful
        evaluation, without crashing the search."""
        n = self.num_control_points
        wire_diameter_mm, free_length_mm = float(x[0]), float(x[1])
        diameter_control_mm = np.asarray(x[2:2 + n], dtype=float)
        pitch_control_mm = np.asarray(x[2 + n:2 + 2 * n], dtype=float)
        try:
            spring = self._build_spring(wire_diameter_mm, free_length_mm, diameter_control_mm, pitch_control_mm)
            # Cheap pre-check: calculate_theta_max is a single quad integral
            # (no per-point root solve), unlike the simulation below, so a
            # grossly invalid candidate can be rejected here without paying
            # for it.
            spring.calculate_theta_max()
            nr_coils = spring.nr_coils
            diameter_spread_mm = float(diameter_control_mm.max() - diameter_control_mm.min())
            fully_telescoped = diameter_spread_mm >= nr_coils * wire_diameter_mm
            solid_length_mm = wire_diameter_mm if fully_telescoped else nr_coils * wire_diameter_mm
            length_hi = free_length_mm - float(self.target_displacement.max())
            penalty = self._geometry_penalty(wire_diameter_mm, diameter_control_mm, pitch_control_mm,
                                             nr_coils, solid_length_mm, length_hi)
            if penalty > 5.0:
                return 1.0 + penalty

            deflection_history, force_history, _ = spring.simulate_progressive_compression(
                max_deflection=free_length_mm * ureg.mm, steps=steps, num_points=num_points)
            deflection = deflection_history.to('mm').magnitude
            force = force_history.to('N').magnitude
            predicted_load = np.interp(self.target_displacement, deflection, force)
            rmse = float(np.sqrt(np.mean((predicted_load - self.target_load) ** 2)))
            norm_rmse = rmse / self._target_load_scale

            # Critical-section stress at the shortest and longest working
            # lengths, using the local diameter there and the load read off
            # the simulated curve -- the same convention
            # CompressionSpringGeneral.calculate_stress_at_position uses
            # (evaluating f_mean_diameter directly at the working length).
            length_lo = free_length_mm - float(self.target_displacement.min())
            load_hi = float(np.interp(self.target_displacement.max(), deflection, force))
            load_lo = float(np.interp(self.target_displacement.min(), deflection, force))
            diameter_hi = float(spring.f_mean_diameter(length_hi * ureg.mm).to('mm').magnitude)
            diameter_lo = float(spring.f_mean_diameter(length_lo * ureg.mm).to('mm').magnitude)
            stress_hi = _shear_stress(diameter_hi, wire_diameter_mm, load_hi)
            stress_lo = _shear_stress(diameter_lo, wire_diameter_mm, load_lo)
            # GoodmanAnalyzer.calculate_safety_factor returns a dimensionless
            # pint Quantity despite its float type hint; float() it before it
            # reaches differential_evolution's objective. Left as a Quantity,
            # it forces scipy's population energy array to dtype=object,
            # which silently turns every per-generation numpy op (sorting,
            # best-selection) into slow Python-level pint arithmetic --
            # observed to inflate this search from seconds to minutes.
            safety_factor = float(self._analyzer(wire_diameter_mm).calculate_safety_factor(stress_hi, stress_lo))
            safety_factor_error = abs(safety_factor - self.safety_factor_target)

            return norm_rmse + self.safety_factor_weight * safety_factor_error + penalty
        except Exception:
            return 1e6

    def design(self) -> GeneralCurveInverseDesign:
        """Run the two-stage regression (global search, then a local polish
        around the nearest standard wire diameter) and return the best
        design, rebuilt and re-verified with the real spring machinery."""
        max_displacement = float(self.target_displacement.max())
        margin_lo, margin_hi = self.free_length_margin
        n = self.num_control_points

        bounds = [self.wire_diameter_bounds, (max_displacement * margin_lo, max_displacement * margin_hi)]
        bounds += [self.diameter_bounds] * n
        bounds += [self.pitch_bounds] * n

        # Global, derivative-free search over the full (2 + 2*n)-parameter
        # space. workers != 1 parallelizes the (expensive, per module
        # docstring) evaluations across processes; deferred updating is
        # required by scipy whenever workers isn't 1. polish=False disables
        # differential_evolution's own built-in local polish (gradient-based
        # L-BFGS-B, via finite-difference gradients -- ~dims+1 evaluations
        # per step, effectively unbounded): harmless for the closed-form
        # search in ConicalCurveCompressionSpringInverseDesigner where each
        # evaluation costs milliseconds, but here every evaluation pays for a
        # real, root-solve-based simulation, so left enabled it can dwarf the
        # entire rest of the search. The Nelder-Mead refine below is this
        # designer's (bounded, via polish_maxiter) local polish instead.
        result = differential_evolution(
            lambda x: self._evaluate(x, self.search_num_points, self.search_steps),
            bounds, maxiter=self.maxiter, popsize=self.popsize, seed=self.seed,
            workers=self.workers, updating='immediate' if self.workers == 1 else 'deferred',
            polish=False,
        )
        wire_diameter_mm, free_length_mm = float(result.x[0]), float(result.x[1])
        diameter_control_mm = np.asarray(result.x[2:2 + n], dtype=float)
        pitch_control_mm = np.asarray(result.x[2 + n:2 + 2 * n], dtype=float)

        # Snap to a manufacturable wire diameter, then locally re-fit the
        # remaining continuous parameters around it.
        standard_diameters = get_standard_wire_diameters()
        wire_diameter_mm = min(standard_diameters, key=lambda d: abs(d - wire_diameter_mm))

        def objective_fixed_diameter(y):
            return self._evaluate([wire_diameter_mm, *y], self.search_num_points, self.search_steps)

        # Nelder-Mead's scipy default (~200 evaluations per dimension) is
        # free for ConicalCurveCompressionSpringInverseDesigner's closed-form
        # search, but each evaluation here pays for a real, root-solve-based
        # simulation (per module docstring) -- left uncapped, this "polish"
        # step could dominate total runtime and dwarf the global search it's
        # meant to merely refine. polish_maxiter keeps it a bounded local
        # polish rather than a from-scratch optimization.
        refined = minimize(objective_fixed_diameter,
                           x0=[free_length_mm, *diameter_control_mm, *pitch_control_mm],
                           method='Nelder-Mead', bounds=bounds[1:],
                           options={'maxiter': self.polish_maxiter, 'maxfev': self.polish_maxiter})
        if refined.success:
            free_length_mm = float(refined.x[0])
            diameter_control_mm = np.asarray(refined.x[1:1 + n], dtype=float)
            pitch_control_mm = np.asarray(refined.x[1 + n:1 + 2 * n], dtype=float)

        # From here on, everything is computed with the real, general,
        # contact-aware CompressionSpringGeneral (final_num_points/
        # final_steps -- fine resolution, since this only runs once), so the
        # reported curve, stresses and safety factor come from the
        # unmodified library machinery rather than the search's coarse
        # evaluations.
        spring = self._build_spring(wire_diameter_mm, free_length_mm, diameter_control_mm, pitch_control_mm)
        spring.calculate_spring_properties(num_points=self.final_num_points)
        deflection_history, force_history, _ = spring.simulate_progressive_compression(
            max_deflection=free_length_mm * ureg.mm, steps=self.final_steps, num_points=self.final_num_points)
        predicted_load = np.interp(self.target_displacement,
                                   deflection_history.to('mm').magnitude,
                                   force_history.to('N').magnitude)
        rmse = float(np.sqrt(np.mean((predicted_load - self.target_load) ** 2)))
        norm_rmse = rmse / self._target_load_scale

        # Register every point of the target curve with the real spring, so
        # its own stress-at-position machinery (not the search's
        # approximation) supplies the reported safety factor.
        for displacement_mm in self.target_displacement:
            spring.add_load_position((free_length_mm - float(displacement_mm)) * ureg.mm)
        # float() for the same reason as in _evaluate: calculate_safety_factor
        # returns a dimensionless pint Quantity despite its float type hint.
        safety_factor = float(self._analyzer(wire_diameter_mm).calculate_safety_factor(
            spring.get_stress_max(), spring.get_stress_min()))

        control_positions_mm = np.linspace(0.0, free_length_mm, n)
        return GeneralCurveInverseDesign(
            spring=spring,
            wire_diameter=spring.wire_diameter,
            free_length=spring.free_length,
            diameter_control_points=diameter_control_mm * ureg.mm,
            pitch_control_points=pitch_control_mm * ureg.mm,
            control_positions=control_positions_mm * ureg.mm,
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
