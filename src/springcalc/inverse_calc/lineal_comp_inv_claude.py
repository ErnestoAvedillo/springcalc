"""Inverse design of a standard cylindrical compression spring.

Given a target spring rate (expressed as two length/force points) and a
target fatigue safety factor, find the wire diameter, mean diameter and free
length that reproduce that rate while landing the safety factor as close as
possible to the target.

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
"""
from dataclasses import dataclass, field
from math import pi
from typing import List, Optional

from pint import Quantity
from scipy.optimize import brentq

from ..lineal.compresion import CompressionSpring
from ..lineal.constants import COMPRESSION_SPRING_END_TYPES, FORMING_TYPES
from ..lineal.goodman import GoodmanAnalyzer, GoodmanData
from ..pymodels.material import Material
from ..pymodels.units import ureg
from ..pymodels.wire_characteristics import get_standard_wire_diameters


@dataclass
class Requirements:
    material: Material
    security_factor: float
    length1: float
    length2: float
    force1: float
    force2: float


def _to_mm(value) -> float:
    return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)


def _to_n(value) -> float:
    return float(value.to('N').magnitude) if isinstance(value, Quantity) else float(value)


def _wahl_factor(spring_index: float) -> float:
    return (4 * spring_index - 1) / (4 * spring_index - 4) + 0.615 / spring_index


def _shear_stress(mean_diameter_mm: float, wire_diameter_mm: float, load_n: float) -> float:
    spring_index = mean_diameter_mm / wire_diameter_mm
    return 8 * mean_diameter_mm * load_n * _wahl_factor(spring_index) / (pi * wire_diameter_mm**3)


@dataclass
class RateTarget:
    """The unique linear rate/free-length consistent with two (length, force)
    points. Two points on a line fully determine it (F = k*(free_length -
    length)), so this is solved directly rather than searched for."""
    length_lo: float  # mm, the shorter (more compressed, higher-force) length
    force_hi: float    # N, force at length_lo
    length_hi: float    # mm, the longer (less compressed, lower-force) length
    force_lo: float      # N, force at length_hi
    spring_constant: float  # N/mm
    free_length: float        # mm


def solve_rate_target(requirements: Requirements) -> RateTarget:
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

    spring_constant = (force_hi - force_lo) / (length_hi - length_lo)
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
    wire_diameter_mm: float
    mean_diameter_mm: float
    spring_index: float
    safety_factor: float
    valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class InverseCompressionDesign:
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
        return self.safety_factor - self.safety_factor_target


class CompressionSpringInverseDesigner:
    """Find the geometry of a standard compression spring that matches a
    target rate (given as two length/force points) with a safety factor as
    close as possible to a target value."""

    def __init__(self, requirements: Requirements,
                 type_of_end: str = COMPRESSION_SPRING_END_TYPES[1],
                 type_conforming: str = FORMING_TYPES[1],
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
        shear_modulus_mpa = self.material.shear_modulus.to('MPa').magnitude
        return (shear_modulus_mpa * wire_diameter_mm**4 /
                (8 * mean_diameter_mm**3 * self.spring_constant_target))

    def _evaluate(self, wire_diameter_mm: float, mean_diameter_mm: float, analyzer: GoodmanAnalyzer) -> float:
        stress_hi = _shear_stress(mean_diameter_mm, wire_diameter_mm, self.force_hi)
        stress_lo = _shear_stress(mean_diameter_mm, wire_diameter_mm, self.force_lo)
        return analyzer.calculate_safety_factor(stress_hi, stress_lo)

    def _validate_geometry(self, wire_diameter_mm: float, mean_diameter_mm: float) -> Optional[str]:
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
        c_lo, c_hi = self.spring_index_bounds
        d_lo, d_hi = c_lo * wire_diameter_mm, c_hi * wire_diameter_mm

        goodman_data = GoodmanData(material=self.material, diameter=wire_diameter_mm,
                                   load_type='torsion', cycles=int(self.number_cycles))
        analyzer = GoodmanAnalyzer(goodman_data, shot_peening=self.shot_peening)

        def f(mean_diameter_mm: float) -> float:
            return self._evaluate(wire_diameter_mm, mean_diameter_mm, analyzer) - self.safety_factor_target

        f_lo, f_hi = f(d_lo), f(d_hi)
        if f_lo * f_hi <= 0:
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

        candidates = [self._search_wire_diameter(d) for d in wire_diameters]
        valid_candidates = [c for c in candidates if c.valid]
        if not valid_candidates:
            raise ValueError("No standard wire diameter yields a valid geometry for this "
                             "rate/safety-factor target within spring_index_bounds="
                             f"{self.spring_index_bounds}")

        c_mid = sum(self.spring_index_bounds) / 2
        best = min(valid_candidates,
                   key=lambda c: (abs(c.safety_factor - self.safety_factor_target), abs(c.spring_index - c_mid)))

        nr_active_coils = self._active_coils(best.mean_diameter_mm, best.wire_diameter_mm)
        nr_coils = nr_active_coils + self._coil_offset

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
