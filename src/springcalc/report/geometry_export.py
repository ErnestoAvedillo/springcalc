"""Structured geometry export for sharing a spring's design: a JSON-safe
dict (and JSON/CSV writers built on it) covering the same data as
`SpringPDFReport`'s data table, plus manufacturing-relevant attributes
(end type, forming, wire tolerance, RMa range) and any design-specific
extras (taper diameters/pitches, safety-factor target, curve fit quality)
carried by an inverse-design result. Meant for handing the geometry to a
colleague's script/spreadsheet, not just a printed report.
"""
import csv
import dataclasses
import json

import numpy as np
from pint import Quantity

from ..pymodels.units import ureg

# Fields on an inverse-design dataclass (ConicalCurveInverseDesign,
# ConicalInverseCompressionDesign, InverseCompressionDesign) that duplicate
# what get_spring_data()/the spring's own attributes already report, or that
# aren't meant for a geometry hand-off (candidate list from the search, raw
# curve-fit arrays).
_SKIP_FIELDS = {
    "spring", "candidates",
    "target_displacement", "target_load",
    "simulated_displacement", "simulated_load",
}


def _serialize(value):
    if isinstance(value, Quantity):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def spring_geometry_profile(spring, num_points: int = 5) -> list:
    """Sample the spring's actual mean-diameter and pitch profile at
    `num_points` evenly spaced positions from the fixed end (h=0) to the
    free end (h=free_length), evaluating `spring.f_mean_diameter`/`f_pitch`
    directly rather than assuming a linear taper. This is accurate for any
    geometry a `CompressionSpringGeneral` can hold -- constant, linear
    taper, or an arbitrary profile -- not just designs that happen to
    expose explicit diameter_start/end fields. Returns `[]` for a spring
    type (e.g. plain `CompressionSpring`) that has no `f_mean_diameter`/
    `f_pitch`, since those already report an exact, non-varying diameter
    via `get_spring_data()`."""
    if not (hasattr(spring, "f_mean_diameter") and hasattr(spring, "f_pitch")):
        return []
    free_length_mm = spring.free_length.to("mm").magnitude
    profile = []
    for h_mm in np.linspace(0.0, free_length_mm, num_points):
        h = h_mm * ureg.mm
        diameter = spring.f_mean_diameter(h)
        pitch = spring.f_pitch(h)
        profile.append({
            "position_mm": round(float(h_mm), 3),
            "mean_diameter_mm": round(float(diameter.to("mm").magnitude), 4),
            "pitch_mm": round(float(pitch.to("mm").magnitude), 4),
        })
    return profile


def spring_geometry_to_dict(design) -> dict:
    """JSON-safe geometry dict for a built spring (`CompressionSpring` /
    `CompressionSpringGeneral`) or an inverse-design result that carries one
    as `design.spring`. `pint.Quantity` values are rendered as
    "<magnitude> <units>" strings."""
    spring = getattr(design, "spring", design)
    data = dict(spring.get_spring_data())
    data["type_of_end"] = getattr(spring, "type_of_end", None)
    data["type_conforming"] = getattr(spring, "type_conforming", None)
    data["diameter_tolerance_mm"] = getattr(spring, "diameter_tolerance", None)
    data["RMa_min_MPa"] = getattr(spring, "RMa_min", None)
    data["RMa_max_MPa"] = getattr(spring, "RMa_max", None)
    profile = spring_geometry_profile(spring)
    if profile:
        data["geometry_profile"] = profile

    if dataclasses.is_dataclass(design) and not isinstance(design, type):
        for field in dataclasses.fields(design):
            if field.name not in _SKIP_FIELDS:
                data[field.name] = getattr(design, field.name)

    return {key: _serialize(value) for key, value in data.items()}


def spring_geometry_to_json(design, indent: int = 2) -> str:
    return json.dumps(spring_geometry_to_dict(design), indent=indent)


def spring_geometry_to_csv(design, output_path: str) -> str:
    """Write the geometry as a two-column (property, value) CSV, followed by
    a `position_mm, mean_diameter_mm, pitch_mm` profile table when the
    spring's diameter/pitch is sampled (see `spring_geometry_profile`).
    Returns `output_path`."""
    data = spring_geometry_to_dict(design)
    profile = data.pop("geometry_profile", None)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["property", "value"])
        for key, value in data.items():
            writer.writerow([key, value])
        if profile:
            writer.writerow([])
            writer.writerow(["position_mm", "mean_diameter_mm", "pitch_mm"])
            for row in profile:
                writer.writerow([row["position_mm"], row["mean_diameter_mm"], row["pitch_mm"]])
    return output_path
