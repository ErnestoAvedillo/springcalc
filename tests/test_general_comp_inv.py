import numpy as np
import pandas as pd
import pytest

from springcalc.inverse_calc.conical_comp_inv import linear_profile
from springcalc.inverse_calc.conical_curve_comp_inv import CompressionCurveRequirements
from springcalc.inverse_calc.general_comp_inv import (
    GeneralCompressionSpringInverseDesigner,
    GeneralCurveInverseDesign,
    spline_profile,
)
from springcalc.lineal.generic_compression import CompressionSpringGeneral
from springcalc.pymodels.material import Material
from springcalc.pymodels.units import ureg


@pytest.fixture
def material():
    return Material(material_name="SL")


@pytest.fixture
def target_curve_csv(tmp_path, material):
    # A small, well-behaved cylindrical spring (constant D/p -- the
    # num_control_points=2 degenerate case of the general profile),
    # simulated with the real machinery so the regression has a genuinely
    # recoverable target. Kept physically small and far from coil contact
    # (max displacement well under half the free length) so even the
    # minimal search budget in designer_kwargs finishes in seconds rather
    # than the minutes-to-hours the module docstring warns a wide-open
    # general search can take (every evaluation here pays for a real,
    # root-solve-based simulation, unlike the closed-form conical search).
    wire_d, d_start, d_end, p_start, p_end, free_length = 2.0, 18.0, 18.0, 5.0, 5.0, 20.0
    spring = CompressionSpringGeneral(material=material, wire_diameter=wire_d * ureg.mm)
    spring.set_geometry(f_mean_diameter=linear_profile(d_start, d_end, free_length),
                        f_pitch=linear_profile(p_start, p_end, free_length),
                        free_length=free_length * ureg.mm)
    deflection, force, _ = spring.simulate_progressive_compression(
        max_deflection=6 * ureg.mm, steps=30, num_points=30)
    df = pd.DataFrame({"displacement": deflection.to("mm").magnitude, "load": force.to("N").magnitude})
    df = df.iloc[::5].reset_index(drop=True)
    csv_path = tmp_path / "target_curve.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def designer_kwargs():
    # Deliberately tight bounds and a minimal search budget, bracketing the
    # known target_curve_csv geometry closely -- keeps this test's design()
    # call to a matter of seconds instead of the minutes-to-hours a wide,
    # thorough general search can take (see module docstring).
    return dict(
        num_control_points=2,
        wire_diameter_bounds=(1.5, 3.0),
        diameter_bounds=(14.0, 22.0),
        pitch_bounds=(3.0, 7.0),
        free_length_margin=(3.0, 3.7),
        search_num_points=8,
        search_steps=8,
        final_num_points=60,
        final_steps=60,
        maxiter=1,
        popsize=2,
        polish_maxiter=10,
        seed=0,
    )


def test_spline_profile_passes_through_control_points_and_clamps():
    control_h = np.array([0.0, 10.0, 20.0])
    control_values = np.array([15.0, 20.0, 12.0])
    func = spline_profile(control_h, control_values, free_length_mm=20.0)

    for h_mm, expected in zip(control_h, control_values):
        assert func(h_mm * ureg.mm).to("mm").magnitude == pytest.approx(expected)

    # Outside [0, free_length_mm] the profile is clamped to the boundary
    # value rather than extrapolated.
    assert func(-5.0 * ureg.mm).to("mm").magnitude == pytest.approx(control_values[0])
    assert func(999.0 * ureg.mm).to("mm").magnitude == pytest.approx(control_values[-1])


def test_designer_rejects_curve_with_fewer_than_two_points(tmp_path, material):
    csv_path = tmp_path / "one_point.csv"
    pd.DataFrame({"displacement": [0], "load": [0]}).to_csv(csv_path, index=False)
    requirements = CompressionCurveRequirements(material=material, safety_factor=1.5, csv_path=str(csv_path))

    with pytest.raises(ValueError):
        GeneralCompressionSpringInverseDesigner(requirements)


def test_designer_rejects_fewer_than_two_control_points(material, target_curve_csv):
    requirements = CompressionCurveRequirements(material=material, safety_factor=1.5, csv_path=target_curve_csv)

    with pytest.raises(ValueError, match="num_control_points"):
        GeneralCompressionSpringInverseDesigner(requirements, num_control_points=1)


def test_geometry_penalty_penalizes_out_of_bounds_geometry(material, target_curve_csv, designer_kwargs):
    requirements = CompressionCurveRequirements(material=material, safety_factor=1.5, csv_path=target_curve_csv)
    designer = GeneralCompressionSpringInverseDesigner(requirements, **designer_kwargs)

    reasonable = designer._geometry_penalty(
        wire_diameter_mm=2.0, diameter_control_mm=np.array([18.0, 18.0]),
        pitch_control_mm=np.array([5.0, 5.0]), nr_coils=4.0, solid_length_mm=8.0, shortest_length_mm=14.0)
    # A spring index far outside spring_index_bounds at one control point
    # should score worse than a reasonable, in-bounds geometry.
    extreme_index = designer._geometry_penalty(
        wire_diameter_mm=2.0, diameter_control_mm=np.array([18.0, 200.0]),
        pitch_control_mm=np.array([5.0, 5.0]), nr_coils=4.0, solid_length_mm=8.0, shortest_length_mm=14.0)

    assert extreme_index > reasonable


def test_evaluate_skips_expensive_simulation_for_grossly_invalid_geometry(material, target_curve_csv, designer_kwargs):
    requirements = CompressionCurveRequirements(material=material, safety_factor=1.5, csv_path=target_curve_csv)
    designer = GeneralCompressionSpringInverseDesigner(requirements, **designer_kwargs)

    # Pitch smaller than the wire diameter at every control point is a
    # severe geometry-penalty violation, well past the 5.0 pre-check
    # threshold in _evaluate -- this should short-circuit before the
    # expensive contact simulation runs, unlike a real evaluation (which
    # takes on the order of 0.1-0.5s for this fixture's geometry).
    x = [2.0, 20.0, 18.0, 18.0, 0.1, 0.1]

    import time
    t0 = time.time()
    value = designer._evaluate(x, designer.search_num_points, designer.search_steps)
    elapsed = time.time() - t0

    assert isinstance(value, float)
    assert value >= 1.0
    assert elapsed < 0.1


def test_evaluate_returns_a_plain_float_for_a_valid_geometry(material, target_curve_csv, designer_kwargs):
    # Regression test: GoodmanAnalyzer.calculate_safety_factor returns a
    # dimensionless pint Quantity despite its float type hint. Left
    # unconverted, that Quantity leaking into differential_evolution's
    # objective forces its population energy array to dtype=object, which
    # silently turns per-generation numpy bookkeeping into slow
    # element-by-element pint arithmetic -- observed to inflate a search
    # from seconds to many minutes. _evaluate must always return a plain
    # float.
    requirements = CompressionCurveRequirements(material=material, safety_factor=1.5, csv_path=target_curve_csv)
    designer = GeneralCompressionSpringInverseDesigner(requirements, **designer_kwargs)

    x = [2.0, 20.0, 18.0, 18.0, 5.0, 5.0]
    value = designer._evaluate(x, designer.search_num_points, designer.search_steps)

    assert type(value) is float
    assert np.isfinite(value)


def test_full_curve_fit_design_produces_valid_general_spring(material, target_curve_csv, designer_kwargs):
    # Exercises the full regression pipeline: differential_evolution search
    # (real, root-solve-based simulation -- see module docstring) +
    # snap-to-standard-diameter + bounded Nelder-Mead polish + final build
    # with the real CompressionSpringGeneral. This is the slow part of this
    # test (tens of seconds even with designer_kwargs' minimal budget,
    # since every evaluation here -- unlike
    # ConicalCurveCompressionSpringInverseDesigner's closed-form search --
    # pays for a real simulation).
    requirements = CompressionCurveRequirements(material=material, safety_factor=1.5, csv_path=target_curve_csv)
    designer = GeneralCompressionSpringInverseDesigner(requirements, **designer_kwargs)

    result = designer.design()

    assert isinstance(result, GeneralCurveInverseDesign)
    # Same Quantity-leak regression guard as test_evaluate_returns_a_plain_
    # float_for_a_valid_geometry, but for the value the final build reports.
    assert type(result.safety_factor) is float
    assert result.curve_rmse_relative < 0.3
    assert result.wire_diameter.to("mm").magnitude > 0

    n = designer.num_control_points
    assert len(result.diameter_control_points) == n
    assert len(result.pitch_control_points) == n
    assert len(result.control_positions) == n
    assert result.control_positions[0].to("mm").magnitude == pytest.approx(0.0)
    assert result.control_positions[-1].to("mm").magnitude == pytest.approx(
        result.free_length.to("mm").magnitude)

    # The spline passes exactly through its own control points, so
    # diameter_at/pitch_at at the two ends must reproduce them.
    assert result.diameter_at(result.control_positions[0]).to("mm").magnitude == pytest.approx(
        result.diameter_control_points[0].to("mm").magnitude)
    assert result.pitch_at(result.control_positions[-1]).to("mm").magnitude == pytest.approx(
        result.pitch_control_points[-1].to("mm").magnitude)

    # The design shouldn't already be solid within the target curve's
    # working range.
    max_displacement_mm = float(designer.target_displacement.max())
    assert result.solid_length.to("mm").magnitude < (
        result.free_length.to("mm").magnitude - max_displacement_mm)

    # Every point of the target curve -- not just its extremes -- must be
    # registered on the returned spring.
    assert len(result.spring.get_data_positions()) == len(designer.target_displacement)
