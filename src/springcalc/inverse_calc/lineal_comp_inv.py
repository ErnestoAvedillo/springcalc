"""Inverse design of a standard cylindrical compression spring.

Given a target spring rate (expressed as either two length/force points or a
CSV of several length/force samples) and a target fatigue safety factor,
find the wire diameter, mean diameter and free length that reproduce that
rate while landing the safety factor as close as possible to the target.

For a fixed pair of load points, the wire stress at those points depends only
on the wire diameter and the mean diameter (the number of coils only sets the
rate, and is solved for so the rate always matches exactly). So the search is
over the standard wire diameter series (DIAMETRO_TOLERANCIAS.csv): for each
wire diameter, the mean diameter that hits the target safety factor is found
by 1-D root finding, since the safety factor is monotonic in the mean
diameter (through the spring index) for indices away from the C=4 asymptote.
Among the wire diameters that can reach the target, the design whose safety
factor is nearest to target is kept, tie-broken by the spring index closest
to the middle of spring_index_bounds.

The rate/free-length target itself can come from two exact points
(length1/length2/force1/force2), solved algebraically, or from a CSV of
several (length, force) samples (`csv_path`), fit by least squares --
useful when the two numbers come from an actual load-deflection
measurement rather than a spec sheet and are noisy. Both paths assume the
samples sit in the spring's pre-contact linear regime, same as the two-point
case; with exactly two points the least-squares fit reduces to the same
line the algebraic solve gives.
"""
from dataclasses import dataclass, field
from math import pi
from typing import List, Optional

import numpy as np
import pandas as pd
from pint import Quantity
from scipy.optimize import brentq

from ..lineal.compresion import CompressionSpring
from ..lineal.constants import COMPRESSION_SPRING_END_TYPES
from ..lineal.constants import OPEN_GROUND, CLOSED_GROUND
from ..lineal.goodman import GoodmanAnalyzer, GoodmanData
from ..pymodels.material import Material
from ..pymodels.units import ureg
from ..pymodels.wire_characteristics import get_standard_wire_diameters


@dataclass
class Requirements:
    """User-facing design brief: a material, a target fatigue safety factor,
    and a rate target given either as two explicit (length, force) points
    or as a CSV of several length/force samples. Provide exactly one of the
    two: either all of length1/length2/force1/force2, or `csv_path` (a CSV
    with `length` and `force` columns, in mm and N). Everything else (wire
    diameter, mean diameter, number of coils) is derived from the resulting
    rate/free-length by the designer classes in this module and the conical
    variants."""
    material: Material
    security_factor: float
    length1: Optional[float] = None
    length2: Optional[float] = None
    force1: Optional[float] = None
    force2: Optional[float] = None
    csv_path: Optional[str] = None


def _to_mm(value) -> float:
    """Accept either a bare number (assumed already in mm) or a pint
    Quantity, and normalize to a plain float in millimeters. Lets callers
    pass Requirements fields as raw numbers or as unit-aware Quantities."""
    return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)


def _to_n(value) -> float:
    """Same normalization as `_to_mm`, but to newtons."""
    return float(value.to('N').magnitude) if isinstance(value, Quantity) else float(value)


def _wahl_factor(spring_index: float) -> float:
    """Wahl stress-concentration factor, which corrects the naive torsion
    formula for the extra shear from direct (transverse) load and for the
    curvature of the coil, both of which get worse as the spring index C
    (mean diameter / wire diameter) shrinks toward tight coiling."""
    return (4 * spring_index - 1) / (4 * spring_index - 4) + 0.615 / spring_index


def _shear_stress(mean_diameter_mm: float, wire_diameter_mm: float, load_n: float) -> float:
    """Corrected shear stress in the wire cross-section (standard round-wire
    compression spring formula, tau = 8*D*F*Kw / (pi*d^3)), used throughout
    this module as the input to the Goodman fatigue check."""
    spring_index = mean_diameter_mm / wire_diameter_mm
    return 8 * mean_diameter_mm * load_n * _wahl_factor(spring_index) / (pi * wire_diameter_mm**3)


@dataclass
class RateTarget:
    """The linear rate/free-length consistent with the given length/force
    samples. Two points on a line fully determine it (F = k*(free_length -
    length)), so with exactly two points this is solved directly; with more,
    it is the least-squares line through them (see
    `_solve_rate_target_from_points`). length_lo/force_hi and length_hi/
    force_lo are the line's values at the shortest and longest sampled
    lengths -- the two representative points later used to check stress at
    the critical (highest-load) and least-critical ends of the working
    range."""
    length_lo: float  # mm, the shorter (more compressed, higher-force) length
    force_hi: float    # N, force at length_lo
    length_hi: float    # mm, the longer (less compressed, lower-force) length
    force_lo: float      # N, force at length_hi
    spring_constant: float  # N/mm
    free_length: float        # mm


def load_length_force_points(csv_path: str) -> tuple:
    """Read a `length,force` CSV (mm, N) of several samples along a
    compression spring's pre-contact linear force-deflection line, and
    return sorted arrays. This is the multi-point generalization of the two
    explicit (length, force) points `Requirements` also accepts -- typical
    use is a real load-deflection measurement with more than two samples
    (and some noise), fit by least squares in
    `_solve_rate_target_from_points` rather than solved exactly."""
    df = pd.read_csv(csv_path)
    missing = {"length", "force"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV at {csv_path} is missing required column(s): {sorted(missing)}")
    df = df.sort_values("length")
    return df["length"].to_numpy(dtype=float), df["force"].to_numpy(dtype=float)


def _solve_rate_target_from_points(lengths_mm: np.ndarray, forces_n: np.ndarray) -> RateTarget:
    """Fit F(length) = slope*length + intercept to an arbitrary number of
    (length, force) samples by least squares (np.polyfit, degree 1), then
    read the spring rate and free length off that line -- the same
    algebraic relationships `solve_rate_target` uses for exactly two points,
    generalized to fit through many. With exactly two points the
    least-squares line passes through both exactly, so this reduces to the
    same result as the two-point algebraic solve."""
    if len(lengths_mm) < 2:
        raise ValueError("At least two (length, force) points are required")
    if len(np.unique(lengths_mm)) < 2:
        raise ValueError("All points have the same length; at least two distinct lengths are required")

    slope, intercept = np.polyfit(lengths_mm, forces_n, 1)
    if slope >= 0:
        raise ValueError("Force must decrease with length for a compression spring "
                         "(the fitted line has a non-negative slope)")

    spring_constant = -slope
    # Zero-force x-intercept of the fitted line is the free length, same
    # extrapolation idea as the two-point case.
    free_length = -intercept / slope

    length_lo = float(lengths_mm.min())
    length_hi = float(lengths_mm.max())
    # Report the *fitted* line's force at the extreme sampled lengths (not
    # the raw, possibly noisy, measured values there), so the stress checks
    # downstream are consistent with the rate that was actually fit.
    force_hi = float(slope * length_lo + intercept)
    force_lo = float(slope * length_hi + intercept)
    return RateTarget(length_lo=length_lo, force_hi=force_hi, length_hi=length_hi,
                      force_lo=force_lo, spring_constant=spring_constant, free_length=free_length)


def solve_rate_target(requirements: Requirements) -> RateTarget:
    """Turn the rate target carried by `requirements` into the line it
    defines: F(length) = spring_constant * (free_length - length). If
    `csv_path` is given, the line is the least-squares fit through however
    many samples the CSV holds (see `_solve_rate_target_from_points`);
    otherwise it is solved directly from the two explicit points, since a
    compression spring's pre-contact force-deflection behavior is linear and
    any two distinct points fully determine both the rate (the line's slope)
    and the free length (its zero-force x-intercept) -- no fitting needed."""
    if requirements.csv_path is not None:
        lengths_mm, forces_n = load_length_force_points(requirements.csv_path)
        return _solve_rate_target_from_points(lengths_mm, forces_n)

    if None in (requirements.length1, requirements.length2, requirements.force1, requirements.force2):
        raise ValueError("Requirements must provide either csv_path, or all of "
                         "length1, length2, force1 and force2")

    length_a, force_a = _to_mm(requirements.length1), _to_n(requirements.force1)
    length_b, force_b = _to_mm(requirements.length2), _to_n(requirements.force2)
    if length_a == length_b:
        raise ValueError("length1 and length2 must be different")
    # Sort so 'lo' is the shorter (more compressed, higher-force) position.
    if length_a < length_b:
        length_lo, force_hi = length_a, force_a
        length_hi, force_lo = length_b, force_b
    else:
        length_lo, force_hi = length_b, force_b
        length_hi, force_lo = length_a, force_a
    if force_hi <= force_lo:
        raise ValueError("The point with the smaller length must have the larger "
                         "force for a compression spring")

    # Slope of the F-vs-length line (rise over run between the two points).
    spring_constant = (force_hi - force_lo) / (length_hi - length_lo)
    # Extrapolate from the more-compressed point back to zero force: that is
    # by definition the free (unloaded) length.
    free_length = length_lo + force_hi / spring_constant
    return RateTarget(length_lo=length_lo, force_hi=force_hi, length_hi=length_hi,
                      force_lo=force_lo, spring_constant=spring_constant, free_length=free_length)


def _active_coils_offset(type_of_end: str, type_conforming: str) -> float:
    """Reuse CompressionSpring.calculate_active_coils to get the constant
    offset between nr_coils and nr_active_coils for the given end/forming
    types, instead of re-deriving that formula here."""
    probe = CompressionSpring(material=Material(material_name="SL"), wire_diameter=1.0,
                              type_of_end=type_of_end, type_conforming=type_conforming)
    return -probe.calculate_active_coils(nr_coils=0)


@dataclass
class CandidateDesign:
    """One evaluated (wire diameter, mean diameter) pair from the search,
    kept around (valid or not) so callers/tests can inspect why the search
    picked what it picked, or why every attempt for a given wire diameter
    was rejected."""
    wire_diameter_mm: float
    mean_diameter_mm: float
    spring_index: float
    safety_factor: float
    valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class InverseCompressionDesign:
    """The winning design: a fully built, ready-to-use `CompressionSpring`
    plus the numbers that justify the choice, and the full `candidates` list
    (one per standard wire diameter tried) for traceability."""
    spring: CompressionSpring
    wire_diameter: Quantity
    mean_diameter: Quantity
    free_length: Quantity
    nr_coils: float
    spring_index: float
    spring_constant: Quantity
    safety_factor: float
    safety_factor_target: float
    candidates: List[CandidateDesign] = field(default_factory=list)

    @property
    def safety_factor_error(self) -> float:
        """Positive means the design is more conservative (safer) than
        requested; negative means it falls short of the target."""
        return self.safety_factor - self.safety_factor_target


class CompressionSpringInverseDesigner:
    """Find the geometry of a standard compression spring that matches a
    target rate (given as two length/force points) with a safety factor as
    close as possible to a target value."""

    def __init__(self, requirements: Requirements,
                 type_of_end: str = COMPRESSION_SPRING_END_TYPES[CLOSED_GROUND],
                 type_conforming: str = 'cold_formed',
                 spring_index_bounds: tuple = (4.5, 12.0),
                 wire_diameter_bounds: tuple = (0.0, float('inf')),
                 min_active_coils: float = 2.0,
                 number_cycles: int = 1_000_000,
                 shot_peening: bool = False):
        self.material = requirements.material
        self.type_of_end = type_of_end
        self.type_conforming = type_conforming
        self.spring_index_bounds = spring_index_bounds
        self.wire_diameter_bounds = wire_diameter_bounds
        self.min_active_coils = min_active_coils
        self.number_cycles = number_cycles
        self.shot_peening = shot_peening
        self.safety_factor_target = requirements.security_factor

        rate = solve_rate_target(requirements)
        self.length_lo, self.force_hi = rate.length_lo, rate.force_hi
        self.length_hi, self.force_lo = rate.length_hi, rate.force_lo
        self.spring_constant_target = rate.spring_constant
        self.free_length_target = rate.free_length

        self._coil_offset = _active_coils_offset(type_of_end, type_conforming)

    def _active_coils(self, mean_diameter_mm: float, wire_diameter_mm: float) -> float:
        """Number of active coils that gives this (mean diameter, wire
        diameter) pair the target spring rate, from the closed-form helical
        spring rate formula K = G*d^4 / (8*D^3*n_active), solved for
        n_active. This is what lets the rate always match exactly: once
        diameters are picked, the coil count is whatever is needed to hit
        the target K, not a free/searched parameter."""
        shear_modulus_mpa = self.material.shear_modulus.to('MPa').magnitude
        return (shear_modulus_mpa * wire_diameter_mm**4 /
                (8 * mean_diameter_mm**3 * self.spring_constant_target))

    def _evaluate(self, wire_diameter_mm: float, mean_diameter_mm: float, analyzer: GoodmanAnalyzer) -> float:
        """Fatigue safety factor for this geometry at the two given load
        points (the operating stress range the spring actually sees)."""
        stress_hi = _shear_stress(mean_diameter_mm, wire_diameter_mm, self.force_hi)
        stress_lo = _shear_stress(mean_diameter_mm, wire_diameter_mm, self.force_lo)
        return analyzer.calculate_safety_factor(stress_hi, stress_lo)

    def _validate_geometry(self, wire_diameter_mm: float, mean_diameter_mm: float) -> Optional[str]:
        """Reject geometries that satisfy the rate/safety-factor equations
        but aren't physically buildable: too few coils to call it a spring,
        a solid (fully-compressed) length that would already be reached
        before the shortest working length (i.e. the spring would bottom out
        under the specified load), or a pitch tighter than the wire itself
        (coils would already be touching at free length)."""
        nr_active_coils = self._active_coils(mean_diameter_mm, wire_diameter_mm)
        if nr_active_coils < self.min_active_coils:
            return f"too few active coils ({nr_active_coils:.2f})"
        nr_coils = nr_active_coils + self._coil_offset
        solid_length = nr_coils * wire_diameter_mm
        if solid_length >= self.length_lo:
            return f"solid length ({solid_length:.2f} mm) reaches the shortest working length"
        pitch = self.free_length_target / nr_coils
        if pitch <= wire_diameter_mm:
            return "pitch would be smaller than the wire diameter"
        return None

    def _search_wire_diameter(self, wire_diameter_mm: float) -> CandidateDesign:
        """For one candidate wire diameter, find the mean diameter that hits
        the target safety factor exactly (if reachable within
        spring_index_bounds), via 1-D root finding. Root finding works here
        because the safety factor is monotonic in mean diameter: a bigger
        mean diameter raises the stress (through the D in tau = 8*D*F*Kw/
        (pi*d^3)) and thus lowers the safety factor, away from the C=4
        asymptote where the Wahl factor blows up."""
        c_lo, c_hi = self.spring_index_bounds
        d_lo, d_hi = c_lo * wire_diameter_mm, c_hi * wire_diameter_mm

        goodman_data = GoodmanData(material=self.material, diameter=wire_diameter_mm,
                                   load_type='torsion', cycles=int(self.number_cycles))
        analyzer = GoodmanAnalyzer(goodman_data, shot_peening=self.shot_peening)

        def f(mean_diameter_mm: float) -> float:
            return self._evaluate(wire_diameter_mm, mean_diameter_mm, analyzer) - self.safety_factor_target

        f_lo, f_hi = f(d_lo), f(d_hi)
        if f_lo * f_hi <= 0:
            # Sign change across the bracket: the target safety factor is
            # achievable somewhere inside spring_index_bounds.
            mean_diameter_mm = brentq(f, d_lo, d_hi)
        else:
            # Target unreachable within spring_index_bounds for this wire
            # diameter: report whichever bound gets closest to it.
            mean_diameter_mm = d_lo if abs(f_lo) < abs(f_hi) else d_hi

        safety_factor = self._evaluate(wire_diameter_mm, mean_diameter_mm, analyzer)
        rejection_reason = self._validate_geometry(wire_diameter_mm, mean_diameter_mm)
        return CandidateDesign(
            wire_diameter_mm=wire_diameter_mm,
            mean_diameter_mm=mean_diameter_mm,
            spring_index=mean_diameter_mm / wire_diameter_mm,
            safety_factor=safety_factor,
            valid=rejection_reason is None,
            rejection_reason=rejection_reason,
        )

    def design(self) -> InverseCompressionDesign:
        """Search the standard wire diameter series and return the best design."""
        d_min, d_max = self.wire_diameter_bounds
        wire_diameters = [d for d in get_standard_wire_diameters() if d_min <= d <= d_max]

        # One independent 1-D root-find per standard wire diameter -- cheap
        # enough to just brute-force the whole series rather than search it.
        candidates = [self._search_wire_diameter(d) for d in wire_diameters]
        valid_candidates = [c for c in candidates if c.valid]
        if not valid_candidates:
            raise ValueError("No standard wire diameter yields a valid geometry for this "
                             "rate/safety-factor target within spring_index_bounds="
                             f"{self.spring_index_bounds}")

        # Primary criterion: closeness to the safety-factor target. Ties (or
        # near-ties, e.g. two diameters both saturating at a bound) are
        # broken by preferring a spring index near the middle of the allowed
        # range, away from manufacturing/behavior extremes at either edge.
        c_mid = sum(self.spring_index_bounds) / 2
        best = min(valid_candidates,
                   key=lambda c: (abs(c.safety_factor - self.safety_factor_target), abs(c.spring_index - c_mid)))

        nr_active_coils = self._active_coils(best.mean_diameter_mm, best.wire_diameter_mm)
        nr_coils = nr_active_coils + self._coil_offset

        # Rebuild the winning geometry through the real CompressionSpring
        # class so the returned object carries the library's full, verified
        # behavior (stress at arbitrary positions, etc.), not just the
        # numbers used during the search.
        spring = CompressionSpring(material=self.material, wire_diameter=best.wire_diameter_mm * ureg.mm,
                                   type_of_end=self.type_of_end, type_conforming=self.type_conforming)
        spring.number_cycles = self.number_cycles
        spring.shot_peening = self.shot_peening
        spring.set_geometry(mean_diameter=best.mean_diameter_mm * ureg.mm,
                            nr_coils=nr_coils,
                            free_length=self.free_length_target * ureg.mm,
                            type_of_end=self.type_of_end,
                            type_conforming=self.type_conforming)
        spring.add_load_position(self.length_lo * ureg.mm)
        spring.add_load_position(self.length_hi * ureg.mm)

        return InverseCompressionDesign(
            spring=spring,
            wire_diameter=spring.wire_diameter,
            mean_diameter=spring.mean_diameter,
            free_length=spring.free_length,
            nr_coils=spring.nr_coils,
            spring_index=best.spring_index,
            spring_constant=spring.spring_constant,
            safety_factor=best.safety_factor,
            safety_factor_target=self.safety_factor_target,
            candidates=candidates,
        )
