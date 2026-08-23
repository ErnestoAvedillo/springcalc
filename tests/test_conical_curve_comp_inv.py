import numpy as np
import pandas as pd
import pytest

from springcalc.inverse_calc.conical_comp_inv import linear_profile
from springcalc.inverse_calc.conical_curve_comp_inv import (
    CompressionCurveRequirements,
    ConicalCurveCompressionSpringInverseDesigner,
    _fast_progressive_compression,
    load_target_curve,
)
from springcalc.lineal.generic_compression import CompressionSpringGeneral
from springcalc.pymodels.material import Material
from springcalc.pymodels.units import ureg
from springcalc.report.pdf_report import SpringPDFReport

@pytest.fixture
def material():
    return Material(material_name="SL")


@pytest.fixture
def target_curve_csv(tmp_path, material):
    # Simulate a known linearly-tapered spring with the real (slow, general)
    # machinery and save its curve, so the regression has a recoverable
    # target with genuine coil-contact nonlinearity in it.
    wire_d, d_start, d_end, p_start, p_end, free_length = 2.5, 25.0, 12.0, 6.0, 4.0, 70.0
    spring = CompressionSpringGeneral(material=material, wire_diameter=wire_d * ureg.mm)
    spring.set_geometry(func_D=linear_profile(d_start, d_end, free_length),
                        func_p=linear_profile(p_start, p_end, free_length),
                        free_length=free_length * ureg.mm)
    deflection, force, _ = spring.simulate_progressive_compression(
        max_deflection=34 * ureg.mm, steps=100, num_points=100)
    df = pd.DataFrame({"displacement": deflection.to("mm").magnitude, "load": force.to("N").magnitude})
    df = df.iloc[::10].reset_index(drop=True)
    csv_path = tmp_path / "target_curve.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_target_curve_sorts_by_displacement(tmp_path):
    csv_path = tmp_path / "curve.csv"
    pd.DataFrame({"displacement": [10, 0, 5], "load": [100, 0, 50]}).to_csv(csv_path, index=False)

    displacement, load = load_target_curve(str(csv_path))

    assert list(displacement) == [0, 5, 10]
    assert list(load) == [0, 50, 100]


def test_load_target_curve_rejects_missing_columns(tmp_path):
    csv_path = tmp_path / "bad_curve.csv"
    pd.DataFrame({"position": [0, 10], "force": [0, 100]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        load_target_curve(str(csv_path))


def test_fast_progressive_compression_matches_real_simulation(material):
    # The closed-form (theta, h) mapping used for speed during the search
    # must reproduce the real, fsolve-based simulation exactly (up to
    # floating point) for a linear D(h)/p(h) profile -- this is the
    # correctness guarantee the whole search's speed depends on.
    wire_d, d_start, d_end, p_start, p_end, free_length = 2.5, 25.0, 12.0, 6.0, 4.0, 70.0
    spring = CompressionSpringGeneral(material=material, wire_diameter=wire_d * ureg.mm)
    spring.set_geometry(func_D=linear_profile(d_start, d_end, free_length),
                        func_p=linear_profile(p_start, p_end, free_length),
                        free_length=free_length * ureg.mm)
    real_deflection, real_force, _ = spring.simulate_progressive_compression(
        max_deflection=40 * ureg.mm, steps=40, num_points=40)

    fast_deflection, fast_force = _fast_progressive_compression(
        wire_d, d_start, d_end, p_start, p_end, free_length,
        material.shear_modulus.to("MPa").magnitude, 40.0, 40, 40)

    assert fast_deflection == pytest.approx(real_deflection.to("mm").magnitude, abs=1e-6)
    assert fast_force == pytest.approx(real_force.to("N").magnitude, abs=1e-6)


def test_fast_progressive_compression_handles_constant_pitch(material):
    # tau_p = 1 (p_start == p_end) hits the closed form's slope~0 branch.
    fast_deflection, fast_force = _fast_progressive_compression(
        2.5, 20.0, 20.0, 5.0, 5.0, 60.0,
        material.shear_modulus.to("MPa").magnitude, 20.0, 20, 20)

    assert np.all(np.isfinite(fast_force))
    assert fast_force[0] == 0.0
    assert np.all(np.diff(fast_force) >= 0.0)


def test_designer_rejects_curve_with_fewer_than_two_points(tmp_path, material):
    csv_path = tmp_path / "one_point.csv"
    pd.DataFrame({"displacement": [0], "load": [0]}).to_csv(csv_path, index=False)
    requirements = CompressionCurveRequirements(material=material, security_factor=1.5, csv_path=str(csv_path))

    with pytest.raises(ValueError):
        ConicalCurveCompressionSpringInverseDesigner(requirements)


def test_evaluate_penalizes_out_of_bounds_geometry(material, target_curve_csv):
    requirements = CompressionCurveRequirements(material=material, security_factor=1.5, csv_path=target_curve_csv)
    designer = ConicalCurveCompressionSpringInverseDesigner(requirements)

    # A spring index far outside spring_index_bounds should score worse than
    # a reasonable one, all else equal.
    reasonable = designer._evaluate([2.5, 70.0, 25.0, 12.0, 6.0, 4.0],
                                    designer.search_num_points, designer.search_steps)
    extreme_index = designer._evaluate([2.5, 70.0, 200.0, 190.0, 6.0, 4.0],
                                       designer.search_num_points, designer.search_steps)

    assert extreme_index > reasonable


def test_full_curve_fit_design_reproduces_target_curve_shape(material, target_curve_csv):
    # Exercises the full regression pipeline: differential_evolution search
    # (fast closed-form simulator) + snap-to-standard-diameter + local
    # refine + final build with the real CompressionSpringGeneral. This is
    # the slow part of this test (the final build alone takes ~20-30s).
    requirements = CompressionCurveRequirements(material=material, security_factor=1.5, csv_path=target_curve_csv)
    designer = ConicalCurveCompressionSpringInverseDesigner(requirements, seed=0)
    result = designer.design()
    report = SpringPDFReport(spring=result.spring, title="Conic result")
    report.build(output_path="Report conical variable.pdf")

    assert result.curve_rmse_relative < 0.2
    # safety_factor_weight makes this a *soft* term competing with curve
    # RMSE and geometry penalties in the same objective (by design -- see
    # the module docstring), so it lands in the neighborhood of the target
    # rather than tracking it tightly; a generous tolerance here checks it's
    # influencing the search at all without over-constraining a soft term.
    assert result.safety_factor == pytest.approx(requirements.security_factor, abs=0.3)
    # diameter_start/diameter_end share the same bounds, so the fit is free
    # to taper in either direction (D_start > D_end or the reverse -- both
    # are equally valid conical geometries, just relabeling which end is
    # h=0); only check that a real taper was actually found, not which way.
    diameter_start_mm = result.diameter_start.to("mm").magnitude
    diameter_end_mm = result.diameter_end.to("mm").magnitude
    assert abs(diameter_start_mm - diameter_end_mm) > 0.05 * max(diameter_start_mm, diameter_end_mm)
    assert result.wire_diameter.to("mm").magnitude > 0
    assert result.solid_length.to("mm").magnitude < (
        result.free_length.to("mm").magnitude - float(designer.target_displacement.max())
    )
    # Every point of the target curve -- not just its two extremes -- must be
    # registered on the returned spring, so a later report can list/plot them all.
    assert len(result.spring.get_data_positions()) == len(designer.target_displacement)


if __name__ == "__main__":
    mat = Material(material_name="SL")
    test_fast_progressive_compression_matches_real_simulation(mat)
    test_fast_progressive_compression_handles_constant_pitch(mat)
    print("All conical_curve_comp_inv_claude fast-path tests passed.")
    