import pytest

from springcalc.inverse_calc.conical_comp_inv import (
    ConicalCompressionSpringInverseDesigner,
    _pitch_integral,
    _taper_integral,
)
from springcalc.inverse_calc.lineal_comp_inv import Requirements
from springcalc.lineal.goodman import GoodmanAnalyzer, GoodmanData
from springcalc.pymodels.material import Material


@pytest.fixture
def material():
    return Material(material_name="SL")


def make_requirements(material, safety_factor=1.2, length1=60, force1=200, length2=90, force2=50):
    return Requirements(
        material=material,
        safety_factor=safety_factor,
        length1=length1, force1=force1,
        length2=length2, force2=force2,
    )


def test_taper_integral_reduces_to_cube_of_wire_diameter_ratio_when_no_taper():
    # With tau_D = tau_p = 1 (no taper), shape_D(u) = shape_p(u) = 1, so the
    # integral of shape_D(u)^3 / shape_p(u) over [0, 1] is just 1.
    assert _taper_integral(1.0, 1.0) == pytest.approx(1.0, rel=1e-6)


def test_pitch_integral_reduces_to_one_when_no_taper():
    assert _pitch_integral(1.0) == pytest.approx(1.0, rel=1e-9)


def test_pitch_integral_matches_numeric_quadrature():
    from scipy.integrate import quad
    tau_p = 2.4
    expected, _ = quad(lambda u: 1.0 / (1.0 + (tau_p - 1.0) * u), 0.0, 1.0)
    assert _pitch_integral(tau_p) == pytest.approx(expected, rel=1e-9)


def test_shape_evaluation_hits_target_safety_factor(material):
    requirements = make_requirements(material)
    designer = ConicalCompressionSpringInverseDesigner(requirements)
    goodman_data = GoodmanData(material=material, wire_diameter=3.0, load_type="torsion", number_cycles=int(1e6))
    analyzer = GoodmanAnalyzer(goodman_data, shot_peening=False)

    candidate = designer._evaluate_shape(tau_D=0.6, tau_p=1.5, wire_diameter_mm=3.0, analyzer=analyzer)

    assert candidate.safety_factor == pytest.approx(requirements.safety_factor, abs=1e-4)
    assert candidate.diameter_end_mm == pytest.approx(candidate.diameter_start_mm * 0.6, rel=1e-9)
    assert candidate.pitch_end_mm == pytest.approx(candidate.pitch_start_mm * 1.5, rel=1e-9)


def test_shape_flexibility_increases_with_taper_so_more_coils_are_needed(material):
    # The stressed (larger) end's diameter is pinned by the safety-factor
    # target regardless of taper, so tapering the small end down only makes
    # the average coil stiffer -- it must be compensated with more coils to
    # keep hitting the same target rate.
    requirements = make_requirements(material)
    designer = ConicalCompressionSpringInverseDesigner(requirements)
    goodman_data = GoodmanData(material=material, wire_diameter=3.2, load_type="torsion", number_cycles=int(1e6))
    analyzer = GoodmanAnalyzer(goodman_data, shot_peening=False)

    no_taper = designer._evaluate_shape(tau_D=1.0, tau_p=1.0, wire_diameter_mm=3.2, analyzer=analyzer)
    tapered = designer._evaluate_shape(tau_D=0.5, tau_p=1.0, wire_diameter_mm=3.2, analyzer=analyzer)

    assert tapered.nr_coils > no_taper.nr_coils
    assert tapered.diameter_start_mm == pytest.approx(no_taper.diameter_start_mm, rel=1e-2)


def test_spring_index_out_of_bounds_is_rejected(material):
    requirements = make_requirements(material)
    designer = ConicalCompressionSpringInverseDesigner(requirements, spring_index_bounds=(4.5, 12.0))
    goodman_data = GoodmanData(material=material, wire_diameter=3.2, load_type="torsion", number_cycles=int(1e6))
    analyzer = GoodmanAnalyzer(goodman_data, shot_peening=False)

    # Extreme taper pushes the small end's spring index below the bound.
    candidate = designer._evaluate_shape(tau_D=0.15, tau_p=1.0, wire_diameter_mm=3.2, analyzer=analyzer)

    assert not candidate.valid
    assert "spring index" in candidate.rejection_reason


def test_equal_lengths_raise(material):
    requirements = make_requirements(material, length1=60, length2=60)
    with pytest.raises(ValueError):
        ConicalCompressionSpringInverseDesigner(requirements)


def test_force_must_be_larger_at_the_shorter_length(material):
    requirements = make_requirements(material, length1=60, force1=50, length2=90, force2=200)
    with pytest.raises(ValueError):
        ConicalCompressionSpringInverseDesigner(requirements)


def test_no_valid_candidate_raises(material):
    requirements = make_requirements(material)
    designer = ConicalCompressionSpringInverseDesigner(requirements, wire_diameter_bounds=(0.05, 0.06))
    with pytest.raises(ValueError):
        designer.design()


def test_full_design_matches_target_rate_and_reproduces_load_points(material):
    # Exercises the full design() pipeline, including building the real
    # CompressionSpringGeneral (theta/h mapping + progressive-compression
    # simulation), which is the slow part of this test.
    requirements = make_requirements(material)
    designer = ConicalCompressionSpringInverseDesigner(requirements, shape_grid_resolution=7)
    result = designer.design()

    assert result.spring_constant.to("N/mm").magnitude == pytest.approx(
        designer.spring_constant_target, rel=1e-2
    )
    assert result.free_length.to("mm").magnitude == pytest.approx(
        designer.free_length_target, rel=1e-9
    )
    assert result.safety_factor == pytest.approx(requirements.safety_factor, abs=0.1)

    positions = {round(p.position.to("mm").magnitude, 3): p.load.to("N").magnitude
                 for p in result.spring.get_data_positions()}
    assert positions[60.0] == pytest.approx(200, rel=0.05)
    assert positions[90.0] == pytest.approx(50, rel=0.05)

    assert result.diameter_start.to("mm").magnitude > 0
    assert result.diameter_end.to("mm").magnitude > 0
    assert result.pitch_start.to("mm").magnitude > result.wire_diameter.to("mm").magnitude
    assert result.pitch_end.to("mm").magnitude > result.wire_diameter.to("mm").magnitude


if __name__ == "__main__":
    mat = Material(material_name="SL")
    test_shape_evaluation_hits_target_safety_factor(mat)
    test_shape_flexibility_increases_with_taper_so_more_coils_are_needed(mat)
    test_spring_index_out_of_bounds_is_rejected(mat)
    test_full_design_matches_target_rate_and_reproduces_load_points(mat)
    print("All conical_comp_inv_claude tests passed.")
