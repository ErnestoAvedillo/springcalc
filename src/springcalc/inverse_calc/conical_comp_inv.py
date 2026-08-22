"""Inverse design of a conical compression spring with linear diameter and
linear pitch profiles: D(h) = D_start + (D_end - D_start) * h / free_length,
p(h) = p_start + (p_end - p_start) * h / free_length.

Like the cylindrical designer (lineal_comp_inv.py), the target rate and free
length are solved from the given length/force points (solve_rate_target) --
either exactly two, solved algebraically, or several via a CSV of
(length, force) samples, fit by least squares -- and the wire diameter is
swept over the standard series.

What's new here is that a linear taper adds two free shape parameters -- the
diameter taper ratio tau_D = D_end/D_start and the pitch taper ratio
tau_p = p_end/p_start -- that the rate and safety factor alone don't pin
down: for *any* (tau_D, tau_p) shape there is generally a pitch scale that
reproduces the target rate exactly and a diameter scale that lands on the
target safety factor exactly (see the derivation below), so infinitely many
tapers satisfy both. The extra degree of freedom is resolved by minimizing
solid length (i.e. preferring designs that telescope/nest as much as
possible), which is the usual reason to choose a conical spring over a
cylindrical one.

Derivation (change of variable from the coil-winding angle theta to the
axial position h, using dtheta = 2*pi/p(h) dh):
    1/K = integral_0^theta_max [8 D(theta)^3 / (G d^4 2 pi)] dtheta
        = integral_0^free_length [8 D(h)^3 / (G d^4 p(h))] dh
This is mathematically equivalent to VariableLinealSpring.calculate_spring_
constant() (which integrates over theta after solving h(theta) by root
finding), but integrating directly over h avoids that root solve, which
matters here since the shape search evaluates this many times. The final
chosen design is still built and verified with the real, contact-aware
CompressionSpringGeneral machinery.

With D(h) = D_start * shape_D(h/free_length), shape_D(u) = 1 + (tau_D-1)*u
(and similarly for p with tau_p and p_start), the rate becomes:
    K = G d^4 p_start / (8 D_start^3 free_length * I(tau_D, tau_p))
    I(tau_D, tau_p) = integral_0^1 [shape_D(u)^3 / shape_p(u)] du
So for a fixed (d, tau_D, tau_p, p_start), D_start is solved in closed form
from K = spring_constant_target. Because the wire stress at any axial
position depends only on the local diameter (torque = force * local
radius), and the two given points are assumed to sit in the pre-contact
linear regime, the critical section is always the larger-diameter end, at
D_max = D_start * max(1, tau_D) -- independent of pitch. So for fixed
(d, tau_D, tau_p), the safety factor is monotonic in p_start alone (bigger
p_start requires a bigger D_start to hold the same rate, which raises stress
and lowers the safety factor), and p_start is solved by 1-D root finding to
hit the target safety factor. That leaves (tau_D, tau_p) as the only truly
free parameters per wire diameter, searched by grid + local polish to
minimize solid length.
"""
from dataclasses import dataclass, field
from math import log
from typing import List, Optional

import numpy as np
from pint import Quantity
from scipy.integrate import quad
from scipy.optimize import brentq, minimize

from .lineal_comp_inv import Requirements, _shear_stress, solve_rate_target
from ..lineal.constants import COMPRESSION_SPRING_END_TYPES, FORMING_TYPES
from ..lineal.generic_compression import CompressionSpringGeneral
from ..lineal.goodman import GoodmanAnalyzer, GoodmanData
from ..pymodels.units import ureg
from ..pymodels.wire_characteristics import get_standard_wire_diameters


def linear_profile(start_mm: float, end_mm: float, free_length_mm: float):
    """A func_D/func_p-compatible closure for a quantity that varies linearly
    from `start_mm` (at h=0) to `end_mm` (at h=free_length_mm). `h` is clamped
    to [0, free_length_mm] first so callers evaluating slightly outside the
    spring's axial extent (e.g. numerical overshoot) get the boundary value
    instead of an extrapolated one."""
    def func(h):
        u = min(max(h.to('mm').magnitude / free_length_mm, 0.0), 1.0)
        return (start_mm + (end_mm - start_mm) * u) * ureg.mm
    return func


def _shape(u: np.ndarray, ratio: float) -> np.ndarray:
    """Normalized linear taper shape on u in [0, 1]: 1 at u=0, `ratio` at
    u=1. Used for both the diameter taper (ratio=tau_D) and the pitch taper
    (ratio=tau_p) so D(h) = D_start * shape(h/L, tau_D) and likewise for p."""
    return 1.0 + (ratio - 1.0) * u


def _taper_integral(tau_D: float, tau_p: float) -> float:
    """I(tau_D, tau_p) = integral_0^1 shape_D(u)^3 / shape_p(u) du -- the
    dimensionless shape factor that the module docstring's rate derivation
    factors out of D(h)^3/p(h). Computed numerically (no closed form once
    both diameter and pitch taper) since it only needs to be evaluated once
    per (tau_D, tau_p) grid point / optimizer iteration, not per simulation
    step."""
    value, _ = quad(lambda u: _shape(u, tau_D)**3 / _shape(u, tau_p), 0.0, 1.0)
    return value


def _pitch_integral(tau_p: float) -> float:
    """integral_0^1 1/shape_p(u) du, in closed form (log mean). Used to turn
    a pitch profile into a coil count: nr_coils = (free_length/p_start) *
    this integral, since dtheta = 2*pi/p(h) dh integrates to 2*pi*nr_coils.
    The tau_p -> 1 (constant pitch) case is a removable singularity of
    log(tau_p)/(tau_p-1) that is handled by returning its limit, 1."""
    if abs(tau_p - 1.0) < 1e-9:
        return 1.0
    return log(tau_p) / (tau_p - 1.0)


@dataclass
class ShapeCandidate:
    """One evaluated (wire diameter, tau_D, tau_p) combination: the
    diameter/pitch geometry it implies (solved to hit the rate and, via
    pitch_start, the safety factor target), plus enough bookkeeping
    (nr_coils, solid_length_mm, valid/rejection_reason) to rank it against
    other candidates or explain why it was thrown out."""
    wire_diameter_mm: float
    tau_D: float
    tau_p: float
    diameter_start_mm: float
    diameter_end_mm: float
    pitch_start_mm: float
    pitch_end_mm: float
    nr_coils: float
    solid_length_mm: float
    safety_factor: float
    valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class ConicalInverseCompressionDesign:
    """The winning tapered design: a fully built, verified
    `CompressionSpringGeneral` plus the geometry/performance numbers that
    justify the choice, and the full `candidates` list (best shape per
    standard wire diameter tried) for traceability."""
    spring: CompressionSpringGeneral
    wire_diameter: Quantity
    diameter_start: Quantity
    diameter_end: Quantity
    pitch_start: Quantity
    pitch_end: Quantity
    free_length: Quantity
    nr_coils: float
    solid_length: Quantity
    spring_constant: Quantity
    safety_factor: float
    safety_factor_target: float
    candidates: List[ShapeCandidate] = field(default_factory=list)

    @property
    def safety_factor_error(self) -> float:
        """Positive means the design is more conservative (safer) than
        requested; negative means it falls short of the target."""
        return self.safety_factor - self.safety_factor_target


class ConicalCompressionSpringInverseDesigner:
    """Find the geometry of a linearly-tapered (diameter and pitch)
    compression spring that matches a target rate (given as two length/force
    points) with a safety factor as close as possible to a target value,
    preferring the most compact (shortest solid length) taper available."""

    def __init__(self, requirements: Requirements,
                 type_of_end: str = COMPRESSION_SPRING_END_TYPES[1],
                 type_conforming: str = FORMING_TYPES[1],
                 spring_index_bounds: tuple = (4.5, 12.0),
                 taper_ratio_bounds: tuple = (0.3, 1.0),
                 pitch_ratio_bounds: tuple = (0.3, 3.0),
                 wire_diameter_bounds: tuple = (0.0, float('inf')),
                 min_coils: float = 2.0,
                 shape_grid_resolution: int = 9,
                 number_cycles: int = 1_000_000,
                 shot_peening: bool = False):
        self.material = requirements.material
        self.type_of_end = type_of_end
        self.type_conforming = type_conforming
        self.spring_index_bounds = spring_index_bounds
        self.taper_ratio_bounds = taper_ratio_bounds
        self.pitch_ratio_bounds = pitch_ratio_bounds
        self.wire_diameter_bounds = wire_diameter_bounds
        self.min_coils = min_coils
        self.shape_grid_resolution = shape_grid_resolution
        self.number_cycles = number_cycles
        self.shot_peening = shot_peening
        self.safety_factor_target = requirements.security_factor
        self.shear_modulus_mpa = self.material.shear_modulus.to('MPa').magnitude

        rate = solve_rate_target(requirements)
        self.length_lo, self.force_hi = rate.length_lo, rate.force_hi
        self.length_hi, self.force_lo = rate.length_hi, rate.force_lo
        self.spring_constant_target = rate.spring_constant
        self.free_length_target = rate.free_length

    def _solve_diameter_start(self, pitch_start_mm: float, wire_diameter_mm: float, taper_integral: float) -> float:
        """Closed-form D_start that makes this (wire diameter, taper shape,
        pitch scale) hit the target spring rate exactly -- the module
        docstring's K = G*d^4*p_start / (8*D_start^3*L*I(tau_D,tau_p))
        solved for D_start."""
        return (self.shear_modulus_mpa * wire_diameter_mm**4 * pitch_start_mm /
                (8 * self.free_length_target * taper_integral * self.spring_constant_target)) ** (1.0 / 3.0)

    def _safety_factor_for_pitch_start(self, pitch_start_mm: float, tau_D: float, wire_diameter_mm: float,
                                       taper_integral: float, analyzer: GoodmanAnalyzer) -> tuple:
        """Safety factor implied by a given pitch_start, for a fixed shape
        (tau_D, tau_p folded into taper_integral) and wire diameter. First
        solves D_start to satisfy the rate, then evaluates stress at the
        larger-diameter end (D_start * max(1, tau_D)), which the module
        docstring shows is always the critical section regardless of
        pitch."""
        diameter_start_mm = self._solve_diameter_start(pitch_start_mm, wire_diameter_mm, taper_integral)
        diameter_max_mm = diameter_start_mm * max(1.0, tau_D)
        stress_hi = _shear_stress(diameter_max_mm, wire_diameter_mm, self.force_hi)
        stress_lo = _shear_stress(diameter_max_mm, wire_diameter_mm, self.force_lo)
        return analyzer.calculate_safety_factor(stress_hi, stress_lo), diameter_start_mm

    def _evaluate_shape(self, tau_D: float, tau_p: float, wire_diameter_mm: float,
                        analyzer: GoodmanAnalyzer) -> ShapeCandidate:
        """For one fixed taper shape (tau_D, tau_p) and wire diameter, solve
        for the pitch_start that hits the safety-factor target (root finding,
        since safety factor is monotonic in pitch_start per the module
        docstring), then derive the rest of the geometry and check it's
        physically valid."""
        taper_integral = _taper_integral(tau_D, tau_p)
        pitch_integral = _pitch_integral(tau_p)

        # Lower bound: pitch just barely bigger than the wire diameter at the
        # tightest point of the pitch taper (min(1, tau_p) is the smaller end
        # of the shape function), with a 0.1% margin so coils don't touch at
        # free length. Upper bound: generous enough to contain the root for
        # any sane free length while never requiring an unbounded search.
        margin = 1.001
        pitch_start_lo = margin * wire_diameter_mm / min(1.0, tau_p)
        pitch_start_hi = max(self.free_length_target, 50 * pitch_start_lo)

        def f(pitch_start_mm: float) -> float:
            sf, _ = self._safety_factor_for_pitch_start(pitch_start_mm, tau_D, wire_diameter_mm,
                                                        taper_integral, analyzer)
            return sf - self.safety_factor_target

        f_lo, f_hi = f(pitch_start_lo), f(pitch_start_hi)
        if f_lo * f_hi <= 0:
            pitch_start_mm = brentq(f, pitch_start_lo, pitch_start_hi)
        else:
            # Target unreachable within bounds for this shape/wire diameter:
            # report whichever bound gets closest to it.
            pitch_start_mm = pitch_start_lo if abs(f_lo) < abs(f_hi) else pitch_start_hi

        safety_factor, diameter_start_mm = self._safety_factor_for_pitch_start(
            pitch_start_mm, tau_D, wire_diameter_mm, taper_integral, analyzer)
        diameter_end_mm = diameter_start_mm * tau_D
        pitch_end_mm = pitch_start_mm * tau_p
        nr_coils = (self.free_length_target / pitch_start_mm) * pitch_integral

        # If the diameter change over the coil stack (diameter_spread_mm) is
        # at least as large as the stacked wire thickness (nr_coils *
        # wire_diameter_mm), the cone is steep enough that every coil can
        # nest fully inside the next, so the spring compresses down to
        # essentially one wire diameter. Otherwise it solid-packs coil by
        # coil like a cylindrical spring.
        diameter_spread_mm = abs(diameter_start_mm - diameter_end_mm)
        fully_telescoped = diameter_spread_mm >= nr_coils * wire_diameter_mm
        solid_length_mm = wire_diameter_mm if fully_telescoped else nr_coils * wire_diameter_mm

        rejection_reason = None
        c_lo, c_hi = self.spring_index_bounds
        c_start, c_end = diameter_start_mm / wire_diameter_mm, diameter_end_mm / wire_diameter_mm
        if not (c_lo <= c_start <= c_hi) or not (c_lo <= c_end <= c_hi):
            rejection_reason = f"spring index out of bounds (C_start={c_start:.2f}, C_end={c_end:.2f})"
        elif nr_coils < self.min_coils:
            rejection_reason = f"too few coils ({nr_coils:.2f})"
        elif solid_length_mm >= self.length_lo:
            rejection_reason = f"solid length ({solid_length_mm:.2f} mm) reaches the shortest working length"

        return ShapeCandidate(
            wire_diameter_mm=wire_diameter_mm, tau_D=tau_D, tau_p=tau_p,
            diameter_start_mm=diameter_start_mm, diameter_end_mm=diameter_end_mm,
            pitch_start_mm=pitch_start_mm, pitch_end_mm=pitch_end_mm,
            nr_coils=nr_coils, solid_length_mm=solid_length_mm,
            safety_factor=safety_factor, valid=rejection_reason is None,
            rejection_reason=rejection_reason,
        )

    @staticmethod
    def _rank(candidate: ShapeCandidate, target: float) -> tuple:
        """Sort key used everywhere a "best" candidate is picked: hitting
        the safety-factor target dominates, and among equally-good safety
        factors the more compact (shorter solid length) shape wins -- the
        stated reason to reach for a conical spring at all."""
        return (abs(candidate.safety_factor - target), candidate.solid_length_mm)

    def _search_wire_diameter(self, wire_diameter_mm: float) -> ShapeCandidate:
        """For one wire diameter, find the best (tau_D, tau_p) taper shape:
        first coarsely by evaluating every point of a shape_grid_resolution^2
        grid (cheap since each point is a closed-form/1-D-root-find
        evaluation, and a grid avoids a local optimizer getting stuck on a
        bad starting guess in a 2-D, possibly multi-modal objective), then
        locally polished with Nelder-Mead around the grid winner."""
        goodman_data = GoodmanData(material=self.material, diameter=wire_diameter_mm,
                                   load_type='torsion', cycles=int(self.number_cycles))
        analyzer = GoodmanAnalyzer(goodman_data, shot_peening=self.shot_peening)

        taper_grid = np.linspace(*self.taper_ratio_bounds, self.shape_grid_resolution)
        pitch_grid = np.linspace(*self.pitch_ratio_bounds, self.shape_grid_resolution)
        grid_candidates = [self._evaluate_shape(tau_D, tau_p, wire_diameter_mm, analyzer)
                           for tau_D in taper_grid for tau_p in pitch_grid]

        best = min(grid_candidates, key=lambda c: self._rank(c, self.safety_factor_target))

        # Only polish if the grid already found a physically valid shape --
        # polishing an invalid one has no valid neighborhood to refine into,
        # and the grid search across the full shape space is more reliable
        # than a local optimizer for finding validity in the first place.
        if best.valid:
            def objective(x):
                candidate = self._evaluate_shape(x[0], x[1], wire_diameter_mm, analyzer)
                penalty = 0.0 if candidate.valid else 1e6
                return abs(candidate.safety_factor - self.safety_factor_target) * 1e4 \
                    + candidate.solid_length_mm + penalty

            polished = minimize(objective, x0=[best.tau_D, best.tau_p], method='Nelder-Mead',
                                bounds=[self.taper_ratio_bounds, self.pitch_ratio_bounds])
            if polished.success:
                polished_candidate = self._evaluate_shape(polished.x[0], polished.x[1], wire_diameter_mm, analyzer)
                if self._rank(polished_candidate, self.safety_factor_target) < self._rank(best, self.safety_factor_target):
                    best = polished_candidate

        return best

    def design(self) -> ConicalInverseCompressionDesign:
        """Search the standard wire diameter series and return the best design."""
        d_min, d_max = self.wire_diameter_bounds
        wire_diameters = [d for d in get_standard_wire_diameters() if d_min <= d <= d_max]

        # One independent shape search per standard wire diameter.
        candidates = [self._search_wire_diameter(d) for d in wire_diameters]
        valid_candidates = [c for c in candidates if c.valid]
        if not valid_candidates:
            raise ValueError("No standard wire diameter yields a valid tapered geometry for this "
                             "rate/safety-factor target within the given bounds")

        best = min(valid_candidates, key=lambda c: self._rank(c, self.safety_factor_target))

        free_length_mm = self.free_length_target
        # Rebuild the winning geometry through the real, contact-aware
        # CompressionSpringGeneral (func_D/func_p based) rather than trusting
        # the closed-form pre-contact numbers used during the search, so the
        # returned design's reported stress/safety-factor come from the same
        # verified machinery every other spring type in this library uses.
        spring = CompressionSpringGeneral(material=self.material, wire_diameter=best.wire_diameter_mm * ureg.mm)
        spring.number_cycles = self.number_cycles
        spring.shot_peening = self.shot_peening
        spring.set_geometry(
            func_D=linear_profile(best.diameter_start_mm, best.diameter_end_mm, free_length_mm),
            func_p=linear_profile(best.pitch_start_mm, best.pitch_end_mm, free_length_mm),
            free_length=free_length_mm * ureg.mm,
            type_of_end=self.type_of_end,
            type_conforming=self.type_conforming,
        )
        spring.calculate_spring_properties(num_points=500)
        spring.add_load_position(self.length_lo * ureg.mm)
        spring.add_load_position(self.length_hi * ureg.mm)

        goodman_data = GoodmanData(material=self.material, diameter=best.wire_diameter_mm,
                                   load_type='torsion', cycles=int(self.number_cycles))
        analyzer = GoodmanAnalyzer(goodman_data, shot_peening=self.shot_peening)
        achieved_safety_factor = analyzer.calculate_safety_factor(spring.get_stress_max(), spring.get_stress_min())

        return ConicalInverseCompressionDesign(
            spring=spring,
            wire_diameter=spring.wire_diameter,
            diameter_start=best.diameter_start_mm * ureg.mm,
            diameter_end=best.diameter_end_mm * ureg.mm,
            pitch_start=best.pitch_start_mm * ureg.mm,
            pitch_end=best.pitch_end_mm * ureg.mm,
            free_length=spring.free_length,
            nr_coils=spring.nr_coils,
            solid_length=spring.calculate_solid_length(),
            spring_constant=spring.spring_constant,
            safety_factor=achieved_safety_factor,
            safety_factor_target=self.safety_factor_target,
            candidates=candidates,
        )
