"""Round-trip export/import for spring models: hand a live spring to a
colleague (or your own future self) as a portable JSON file, and get back a
fully functional object of the same type -- not just a flattened report.

Export captures the inputs the spring was built from (material, wire
diameter, geometry, and every load position added via
`add_load_position`/`add_position`) rather than freezing every derived
field. Import replays those same inputs through the class's own public API
(`set_geometry`, `add_load_position`, ...), so the reconstructed spring is
recomputed by -- and stays consistent with -- whatever version of the
calculation logic the recipient has, instead of trusting a snapshot of
numbers that could silently drift from what the current code would compute.

A `results` section (on by default) additionally embeds a snapshot of the
computed geometry/properties (via `report.geometry_export`) and the
positions table, so the JSON is also useful on its own -- to a spreadsheet,
a colleague's script, or a human -- without needing this library to make
sense of it. That section is descriptive only: `import_model` never reads
it back.

`CompressionSpringGeneral`'s geometry is an arbitrary `f_mean_diameter`/
`f_pitch` function pair rather than a handful of numbers, so it's captured
by sampling the profile at `profile_points` positions and reconstructing a
piecewise-linear interpolation through them on import. For the linear
tapers this library's own inverse-design tools produce, that reconstructs
exactly; for any other shape, fidelity improves with `profile_points`.

Reconstructing `Material` only works if the recipient's `materials.csv` has
a matching `material_name` entry (the table shipped with the package, or
one added locally via `Material.create_material`) -- a custom material that
was never shared this way cannot round-trip.
"""
import json

import numpy as np
from pint import Quantity

from .lineal.compresion import CompressionSpring
from .lineal.extension import ExtensionSpring
from .lineal.generic_compression import CompressionSpringGeneral
from .lineal.torsion import TorsionSpring
from .pymodels.material import Material
from .pymodels.units import ureg
from .report.geometry_export import spring_geometry_profile, spring_geometry_to_dict

SCHEMA_VERSION = 1

_SPRING_TYPES = {
    "CompressionSpring": CompressionSpring,
    "CompressionSpringGeneral": CompressionSpringGeneral,
    "ExtensionSpring": ExtensionSpring,
    "TorsionSpring": TorsionSpring,
}


def _qstr(value):
    """Render a `pint.Quantity` as its round-trippable string form; pass
    everything else (numbers, strings, bools, None) through unchanged."""
    if isinstance(value, Quantity):
        return str(value)
    return value


def _q(value):
    """Parse a quantity string back into a `pint.Quantity`; pass numbers/
    None through unchanged, since every field validator in this library
    already accepts either."""
    if isinstance(value, str):
        return ureg(value)
    return value


def _position_results(spring) -> list:
    """JSON-safe snapshot of the spring's positions table."""
    fields = ("position", "travel", "load", "stress", "outer_diameter", "inner_diameter")
    return [{key: _qstr(getattr(pc, key)) for key in fields} for pc in spring.positions.positions]


# ---------------------------------------------------------------------------
# Per-type input capture and reconstruction. Each spring class has its own
# set_geometry() signature, so these stay separate; material handling and
# positions replay are shared below.
# ---------------------------------------------------------------------------

def _export_inputs_compression(spring: CompressionSpring, profile_points: int) -> dict:
    return {
        "mean_diameter": _qstr(spring.mean_diameter),
        "nr_coils": spring.nr_coils,
        "pitch": _qstr(spring.pitch),
        "type_of_end": spring.type_of_end,
        "type_conforming": spring.type_conforming,
        "shot_peening": spring.shot_peening,
        "coating": spring.coating,
        "number_cycles": spring.number_cycles,
    }


def _build_compression(material: Material, wire_diameter, inputs: dict) -> CompressionSpring:
    spring = CompressionSpring(material, wire_diameter)
    spring.set_geometry(
        mean_diameter=_q(inputs["mean_diameter"]),
        nr_coils=inputs["nr_coils"],
        pitch=_q(inputs["pitch"]),
        type_of_end=inputs.get("type_of_end"),
    )
    if inputs.get("type_conforming") is not None:
        spring.type_conforming = inputs["type_conforming"]
    spring.shot_peening = inputs.get("shot_peening", False)
    spring.coating = inputs.get("coating")
    spring.number_cycles = inputs.get("number_cycles", spring.number_cycles)
    return spring


def _interp_profile_func(h_mm: np.ndarray, values_mm: np.ndarray):
    """A f_mean_diameter/f_pitch-compatible closure that piecewise-linearly
    interpolates a sampled profile, clamped to the sampled range -- the
    same clamping convention as `inverse_calc.conical_comp_inv.linear_profile`,
    generalized to any (not just linear) geometry."""
    def func(h):
        h_val = min(max(float(h.to('mm').magnitude), h_mm[0]), h_mm[-1])
        return float(np.interp(h_val, h_mm, values_mm)) * ureg.mm
    return func


def _export_inputs_compression_general(spring: CompressionSpringGeneral, profile_points: int) -> dict:
    return {
        "free_length": _qstr(spring.free_length),
        "geometry_profile": spring_geometry_profile(spring, num_points=profile_points),
        "type_of_end": spring.type_of_end,
        "type_conforming": spring.type_conforming,
        "shot_peening": spring.shot_peening,
        "coating": spring.coating,
        "number_cycles": spring.number_cycles,
    }


def _build_compression_general(material: Material, wire_diameter, inputs: dict) -> CompressionSpringGeneral:
    profile = inputs["geometry_profile"]
    h_mm = np.array([p["position_mm"] for p in profile], dtype=float)
    d_mm = np.array([p["mean_diameter_mm"] for p in profile], dtype=float)
    p_mm = np.array([p["pitch_mm"] for p in profile], dtype=float)

    spring = CompressionSpringGeneral(material, wire_diameter)
    spring.shot_peening = inputs.get("shot_peening", False)
    spring.coating = inputs.get("coating")
    spring.number_cycles = inputs.get("number_cycles", spring.number_cycles)
    spring.set_geometry(
        f_mean_diameter=_interp_profile_func(h_mm, d_mm),
        f_pitch=_interp_profile_func(h_mm, p_mm),
        free_length=_q(inputs["free_length"]),
        type_of_end=inputs.get("type_of_end"),
    )
    if inputs.get("type_conforming") is not None:
        spring.type_conforming = inputs["type_conforming"]
    spring.calculate_spring_properties()
    return spring


def _export_inputs_extension(spring: ExtensionSpring, profile_points: int) -> dict:
    return {
        "mean_diameter": _qstr(spring.mean_diameter),
        "nr_coils": spring.nr_coils,
        "pitch": _qstr(spring.pitch),
        "free_length": _qstr(spring.free_length),
        "type_of_end": spring.type_of_end,
        "shot_peening": spring.shot_peening,
        "coating": spring.coating,
        "number_cycles": spring.number_cycles,
        "initial_force": _qstr(spring.initial_force),
        "initial_stress": _qstr(spring.initial_stress),
    }


def _build_extension(material: Material, wire_diameter, inputs: dict) -> ExtensionSpring:
    spring = ExtensionSpring(material, wire_diameter)
    spring.set_geometry(
        mean_diameter=_q(inputs["mean_diameter"]),
        nr_coils=inputs["nr_coils"],
        pitch=_q(inputs["pitch"]),
        free_length=_q(inputs["free_length"]),
    )
    if inputs.get("type_of_end") is not None:
        spring.type_of_end = inputs["type_of_end"]
    spring.shot_peening = inputs.get("shot_peening", False)
    spring.coating = inputs.get("coating")
    spring.number_cycles = inputs.get("number_cycles", spring.number_cycles)
    spring.initial_force = _q(inputs.get("initial_force", 0.0))
    spring.initial_stress = _q(inputs.get("initial_stress", 0.0))
    return spring


def _export_inputs_torsion(spring: TorsionSpring, profile_points: int) -> dict:
    return {
        "mean_diameter": _qstr(spring.mean_diameter),
        "nr_coils": spring.nr_coils,
        "pitch": _qstr(spring.pitch),
        "free_angle": _qstr(spring.free_angle),
        "fixed_leg_radius": _qstr(spring.fixed_leg_radius),
        "mobile_leg_radius": _qstr(spring.mobile_leg_radius),
        "shot_peening": spring.shot_peening,
        "number_cycles": spring.number_cycles,
    }


def _build_torsion(material: Material, wire_diameter, inputs: dict) -> TorsionSpring:
    spring = TorsionSpring(material, wire_diameter)
    spring.set_geometry(
        mean_diameter=_q(inputs["mean_diameter"]),
        nr_coils=inputs["nr_coils"],
        pitch=_q(inputs["pitch"]),
        free_angle=_q(inputs["free_angle"]),
        fixed_leg_radius=_q(inputs["fixed_leg_radius"]),
        mobile_leg_radius=_q(inputs["mobile_leg_radius"]),
    )
    spring.shot_peening = inputs.get("shot_peening", False)
    spring.number_cycles = inputs.get("number_cycles", spring.number_cycles)
    return spring


# Positions are replayed by whatever value the spring's own add_* method
# takes: an absolute position length for the linear types, an angular
# travel for TorsionSpring (its own add_load_position wraps add_position
# with angle_travel, so exporting the travel and always replaying through
# angle_travel -- never torque -- covers a position added either way).
_EXPORT_POSITIONS = {
    "CompressionSpring": lambda s: [_qstr(pc.position) for pc in s.positions.positions],
    "CompressionSpringGeneral": lambda s: [_qstr(pc.position) for pc in s.positions.positions],
    "ExtensionSpring": lambda s: [_qstr(pc.position) for pc in s.positions.positions],
    "TorsionSpring": lambda s: [_qstr(pc.travel) for pc in s.positions.positions],
}

_REPLAY_POSITIONS = {
    "CompressionSpring": lambda s, v: s.add_load_position(_q(v)),
    "CompressionSpringGeneral": lambda s, v: s.add_load_position(_q(v)),
    # Unlike the other add_load_position()s, ExtensionSpring's expects a bare
    # mm magnitude (it does `length - self.length_btw_hooks.magnitude`
    # directly, with no Quantity normalization) -- pass a Quantity here and
    # that mixed arithmetic raises.
    "ExtensionSpring": lambda s, v: s.add_load_position(_q(v).to('mm').magnitude),
    "TorsionSpring": lambda s, v: s.add_position(angle_travel=_q(v)),
}

_EXPORT_INPUTS = {
    "CompressionSpring": _export_inputs_compression,
    "CompressionSpringGeneral": _export_inputs_compression_general,
    "ExtensionSpring": _export_inputs_extension,
    "TorsionSpring": _export_inputs_torsion,
}

_BUILD = {
    "CompressionSpring": _build_compression,
    "CompressionSpringGeneral": _build_compression_general,
    "ExtensionSpring": _build_extension,
    "TorsionSpring": _build_torsion,
}


def export_model(spring, include_results: bool = True, profile_points: int = 50) -> dict:
    """A JSON-safe dict capturing everything needed to rebuild `spring`
    through its own public API (its `inputs`), plus, by default, a
    `results` snapshot of its current computed geometry/properties and
    positions table for reading without importing this library.

    `spring` must be a `CompressionSpring`, `CompressionSpringGeneral`,
    `ExtensionSpring`, or `TorsionSpring` instance.
    """
    spring_type = type(spring).__name__
    if spring_type not in _SPRING_TYPES:
        raise TypeError(
            f"Don't know how to export a {spring_type!r}; supported types are "
            f"{', '.join(_SPRING_TYPES)}."
        )

    data = {
        "schema_version": SCHEMA_VERSION,
        "spring_type": spring_type,
        "inputs": {
            "material": spring.material.material_name,
            "wire_diameter": _qstr(spring.wire_diameter),
            **_EXPORT_INPUTS[spring_type](spring, profile_points),
        },
        "positions": _EXPORT_POSITIONS[spring_type](spring),
    }
    if include_results:
        results = spring_geometry_to_dict(spring)
        results["positions"] = _position_results(spring)
        data["results"] = results
    return data


def import_model(data: dict):
    """Rebuild a live spring object from a dict produced by `export_model`
    (or an equivalent hand-written one). Reconstructs the spring by
    replaying `inputs` through the same public API used to build it in the
    first place (`set_geometry`, then `add_load_position`/`add_position`
    for each entry in `positions`). The `results` section, if present, is
    never read back -- it's for humans/other tools, not reconstruction.
    """
    spring_type = data.get("spring_type")
    if spring_type not in _SPRING_TYPES:
        raise ValueError(
            f"Unknown or missing spring_type {spring_type!r}; expected one of "
            f"{', '.join(_SPRING_TYPES)}."
        )
    inputs = data["inputs"]
    material = Material(material_name=inputs["material"])
    spring = _BUILD[spring_type](material, _q(inputs["wire_diameter"]), inputs)

    replay = _REPLAY_POSITIONS[spring_type]
    for value in data.get("positions", []):
        replay(spring, value)

    return spring


def model_to_json(spring, include_results: bool = True, profile_points: int = 50, indent: int = 2) -> str:
    """`export_model(spring, ...)`, serialized to a JSON string."""
    return json.dumps(
        export_model(spring, include_results=include_results, profile_points=profile_points),
        indent=indent,
    )


def model_from_json(text: str):
    """`import_model` from a JSON string produced by `model_to_json`."""
    return import_model(json.loads(text))


def save_model(spring, path: str, include_results: bool = True, profile_points: int = 50, indent: int = 2) -> str:
    """Write `spring` to `path` as JSON (see `export_model`). Returns `path`."""
    with open(path, "w") as f:
        f.write(model_to_json(spring, include_results=include_results, profile_points=profile_points, indent=indent))
    return path


def load_model(path: str):
    """Read a spring previously written by `save_model` and rebuild it as a live object."""
    with open(path) as f:
        return model_from_json(f.read())
