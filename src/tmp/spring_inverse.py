"""Inverse design: find a general (conical, variable-pitch) compression spring
geometry whose simulated load-displacement curve matches a target CSV.

Geometry is parametrized by control points for the mean-diameter profile D(h)
and the pitch profile p(h), interpolated with a monotone cubic (PCHIP) spline,
plus wire diameter and free length. Material is swept over a small set of
representative materials (one per distinct shear modulus, since shear modulus
is what drives the force-deflection shape) with an inner CMA-ES search over
the continuous geometry parameters for each.
"""
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import cma
import numpy as np
import pandas as pd
from pint import Quantity
from scipy.interpolate import PchipInterpolator

# springcalc's root-solve for h(theta) can warn about slow fsolve convergence
# at the coarse resolution used during the search; it still clips to a valid
# range, so this is noise rather than a correctness problem.
warnings.filterwarnings("ignore", message="The iteration is not making good progress*")

from springcalc import CompressionSpringGeneral, Material, ureg

# One representative material per distinct shear_modulus value in materials.csv.
REPRESENTATIVE_MATERIALS = [
    "DH",              # shear_modulus 81500 MPa (most carbon/hard steels)
]

N_CONTROL_POINTS = 4
WIRE_DIAMETER_BOUNDS_MM = (0.5, 8.0)
D_CONTROL_BOUNDS_MM = (5.0, 150.0)
P_CONTROL_BOUNDS_MM = (0.2, 25.0)
SPRING_INDEX_BOUNDS = (4.0, 22.0)

SEARCH_NUM_POINTS = 30
SEARCH_STEPS = 20
FINAL_NUM_POINTS = 500
FINAL_STEPS = 500


def load_target_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a `displacement,load` CSV (mm, N) and return sorted arrays."""
    df = pd.read_csv(path).sort_values("displacement")
    return df["displacement"].to_numpy(dtype=float), df["load"].to_numpy(dtype=float)


@dataclass
class DesignParams:
    wire_diameter_mm: float
    free_length_mm: float
    d_control_mm: np.ndarray
    p_control_mm: np.ndarray


def _param_bounds(target_disp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper bounds for the flat parameter vector:
    [wire_diameter, free_length, D_ctrl x N, p_ctrl x N]."""
    max_disp = float(target_disp.max())
    lower = [WIRE_DIAMETER_BOUNDS_MM[0], max_disp * 1.05]
    upper = [WIRE_DIAMETER_BOUNDS_MM[1], max_disp * 3.0]
    lower += [D_CONTROL_BOUNDS_MM[0]] * N_CONTROL_POINTS
    upper += [D_CONTROL_BOUNDS_MM[1]] * N_CONTROL_POINTS
    lower += [P_CONTROL_BOUNDS_MM[0]] * N_CONTROL_POINTS
    upper += [P_CONTROL_BOUNDS_MM[1]] * N_CONTROL_POINTS
    return np.array(lower), np.array(upper)


def _unpack(x: np.ndarray) -> DesignParams:
    return DesignParams(
        wire_diameter_mm=x[0],
        free_length_mm=x[1],
        d_control_mm=x[2:2 + N_CONTROL_POINTS],
        p_control_mm=x[2 + N_CONTROL_POINTS:2 + 2 * N_CONTROL_POINTS],
    )


def _build_geometry_functions(params: DesignParams) -> tuple[Callable[[Quantity], Quantity], Callable[[Quantity], Quantity]]:
    ctrl_h = np.linspace(0.0, params.free_length_mm, N_CONTROL_POINTS)
    d_spline = PchipInterpolator(ctrl_h, params.d_control_mm)
    p_spline = PchipInterpolator(ctrl_h, params.p_control_mm)
    free_length_mm = params.free_length_mm

    def func_D(h: Quantity) -> Quantity:
        h_mm = min(max(h.to("mm").magnitude, 0.0), free_length_mm)
        return float(d_spline(h_mm)) * ureg.mm

    def func_p(h: Quantity) -> Quantity:
        h_mm = min(max(h.to("mm").magnitude, 0.0), free_length_mm)
        return float(p_spline(h_mm)) * ureg.mm

    return func_D, func_p


def format_profile_formula(params: DesignParams) -> str:
    """Render D(h) and p(h) as explicit piecewise-cubic formulas.

    The optimizer only ever produces N_CONTROL_POINTS control values; the
    smooth curve between them is a PCHIP (monotone Hermite) spline, which is
    itself a cubic polynomial on each interval between consecutive control
    points. This extracts those exact per-interval coefficients from
    PchipInterpolator instead of leaving the "formula" implicit in code.
    """
    ctrl_h = np.linspace(0.0, params.free_length_mm, N_CONTROL_POINTS)

    def render(name: str, values: np.ndarray) -> str:
        spline = PchipInterpolator(ctrl_h, values)
        lines = [f"{name}(h) for h in [0, {params.free_length_mm:.3f}] mm, "
                 f"control points at h={np.round(ctrl_h, 2).tolist()} -> "
                 f"{name}={np.round(values, 3).tolist()} mm:"]
        for i in range(len(ctrl_h) - 1):
            c3, c2, c1, c0 = spline.c[:, i]
            h0 = ctrl_h[i]
            lines.append(
                f"  for {h0:8.3f} <= h < {ctrl_h[i + 1]:8.3f} mm:  "
                f"{name}(h) = {c3:.6e}*(h-{h0:.3f})^3 + {c2:.6e}*(h-{h0:.3f})^2 "
                f"+ {c1:.6e}*(h-{h0:.3f}) + {c0:.6e}"
            )
        return "\n".join(lines)

    return (render("D", params.d_control_mm) + "\n\n"
            + render("p", params.p_control_mm))


def build_spring(material: Material, params: DesignParams) -> CompressionSpringGeneral:
    func_D, func_p = _build_geometry_functions(params)
    spring = CompressionSpringGeneral(material=material, wire_diameter=params.wire_diameter_mm * ureg.mm)
    spring.set_geometry(func_D=func_D, func_p=func_p, free_length=params.free_length_mm * ureg.mm)
    return spring


def _geometry_penalty(params: DesignParams, nr_coils: float, target_disp: np.ndarray) -> float:
    spring_index = params.d_control_mm / params.wire_diameter_mm
    below = np.clip(SPRING_INDEX_BOUNDS[0] - spring_index, 0, None)
    above = np.clip(spring_index - SPRING_INDEX_BOUNDS[1], 0, None)
    penalty = float(np.sum(below**2 + above**2)) * 0.01

    pitch_violation = np.clip(params.wire_diameter_mm - params.p_control_mm, 0, None)
    penalty += float(np.sum(pitch_violation**2)) * 0.1

    solid_length_estimate = nr_coils * params.wire_diameter_mm
    reach_violation = max(0.0, float(target_disp.max()) - (params.free_length_mm - solid_length_estimate))
    penalty += reach_violation**2 * 0.05

    return penalty


def evaluate_design(material: Material, params: DesignParams, target_disp: np.ndarray, target_load: np.ndarray,
                     num_points: int = SEARCH_NUM_POINTS, steps: int = SEARCH_STEPS) -> float:
    """Normalized RMSE between the simulated and target load curves, plus soft
    penalties for physically invalid geometry. Returns a large finite value on
    any simulation failure instead of raising, so CMA-ES can keep exploring."""
    try:
        spring = build_spring(material, params)
        free_length = params.free_length_mm * ureg.mm
        deflection_history, force_history, _ = spring.simulate_progressive_compression(
            max_deflection=free_length, steps=steps, num_points=num_points,
        )
        pred_load = np.interp(
            target_disp,
            deflection_history.to("mm").magnitude,
            force_history.to("N").magnitude,
        )
        rmse = float(np.sqrt(np.mean((pred_load - target_load) ** 2)))
        norm_rmse = rmse / max(float(np.mean(np.abs(target_load))), 1e-6)

        penalty = _geometry_penalty(params, spring.nr_coils, target_disp)
        return norm_rmse + penalty
    except Exception:
        return 1e6


def optimize_material(material_name: str, target_disp: np.ndarray, target_load: np.ndarray,
                       maxfevals: int = 300, seed: int | None = None) -> tuple[DesignParams, float]:
    """CMA-ES search (in a normalized [0,1]^n space) over geometry parameters
    for a fixed material. Returns the best DesignParams found and its loss."""
    material = Material(material_name=material_name)
    lower, upper = _param_bounds(target_disp)

    def objective(u: Sequence[float]) -> float:
        u = np.clip(np.asarray(u), 0.0, 1.0)
        x = lower + u * (upper - lower)
        params = _unpack(x)
        return evaluate_design(material, params, target_disp, target_load)

    x0 = [0.5] * len(lower)
    es = cma.CMAEvolutionStrategy(x0, 0.25, {
        "bounds": [0.0, 1.0],
        "maxfevals": maxfevals,
        "verbose": -9,
        "seed": seed,
    })
    es.optimize(objective)

    best_u = np.clip(np.asarray(es.result.xbest), 0.0, 1.0)
    best_x = lower + best_u * (upper - lower)
    return _unpack(best_x), float(es.result.fbest)


@dataclass
class InverseDesignResult:
    material_name: str
    params: DesignParams
    loss: float
    spring: CompressionSpringGeneral


def run_inverse_design(csv_path: str, materials: Sequence[str] = REPRESENTATIVE_MATERIALS,
                       maxfevals: int = 300, verbose: bool = True) -> InverseDesignResult:
    target_disp, target_load = load_target_curve(csv_path)

    best: InverseDesignResult | None = None
    for material_name in materials:
        params, loss = optimize_material(material_name, target_disp, target_load, maxfevals=maxfevals)
        if verbose:
            print(f"material={material_name:16s} loss={loss:.4f} "
                  f"d={params.wire_diameter_mm:.2f}mm L={params.free_length_mm:.1f}mm")
        if best is None or loss < best.loss:
            best = InverseDesignResult(material_name=material_name, params=params, loss=loss, spring=None)

    assert best is not None
    material = Material(material_name=best.material_name)
    spring = build_spring(material, best.params)
    spring.calculate_spring_properties(num_points=FINAL_NUM_POINTS)
    best.spring = spring
    return best
