import pytest

from springcalc.inverse_calc.lineal_comp_inv import (
    CompressionSpringInverseDesigner,
    Requirements,
)
from springcalc.pymodels.material import Material
from springcalc.pymodels.units import ureg
from springcalc.report.pdf_report import SpringPDFReport

@pytest.fixture
def material():
    return Material(material_name="SL")


def make_requirements(material, safety_factor=1.5, length1=60, force1=200, length2=90, force2=50):
    return Requirements(
        material=material,
        safety_factor=safety_factor,
        length1=length1, force1=force1,
        length2=length2, force2=force2,
    )


def test_design_matches_target_spring_constant_exactly(material):
    requirements = make_requirements(material)
    designer = CompressionSpringInverseDesigner(requirements)
    result = designer.design()

    assert result.spring_constant.to("N/mm").magnitude == pytest.approx(
        designer.spring_constant_target, rel=1e-9
    )
    assert result.free_length.to("mm").magnitude == pytest.approx(
        designer.free_length_target, rel=1e-9
    )


def test_design_hits_safety_factor_target(material):
    requirements = make_requirements(material, safety_factor=1.5)
    designer = CompressionSpringInverseDesigner(requirements)
    result = designer.design()

    assert result.safety_factor == pytest.approx(1.5, abs=1e-4)
    assert result.safety_factor_error == pytest.approx(0.0, abs=1e-4)


def test_design_reproduces_the_two_load_points(material):
    requirements = make_requirements(material, length1=60, force1=200, length2=90, force2=50)
    designer = CompressionSpringInverseDesigner(requirements)
    result = designer.design()

    positions = {round(p.position.to("mm").magnitude, 3): p.load.to("N").magnitude
                 for p in result.spring.get_data_positions()}
    assert positions[60.0] == pytest.approx(200, rel=1e-6)
    assert positions[90.0] == pytest.approx(50, rel=1e-6)


def test_design_spring_index_within_bounds(material):
    bounds = (4.5, 12.0)
    requirements = make_requirements(material)
    designer = CompressionSpringInverseDesigner(requirements, spring_index_bounds=bounds)
    result = designer.design()

    assert bounds[0] <= result.spring_index <= bounds[1]


def test_design_returns_geometry_that_does_not_bind_before_the_shortest_length(material):
    requirements = make_requirements(material)
    designer = CompressionSpringInverseDesigner(requirements)
    result = designer.design()

    solid_length = result.spring.calculate_solid_length()
    assert solid_length.to("mm").magnitude < 60.0


def test_unreachable_safety_factor_returns_closest_achievable(material):
    requirements = make_requirements(material, safety_factor=50.0)
    designer = CompressionSpringInverseDesigner(requirements)
    result = designer.design()

    # No standard wire diameter can hit SF=50 within the default spring-index
    # bounds for this rate, so the best candidate falls well short of it.
    assert result.safety_factor < requirements.safety_factor


def test_equal_lengths_raise(material):
    requirements = make_requirements(material, length1=60, length2=60)
    with pytest.raises(ValueError):
        CompressionSpringInverseDesigner(requirements)


def test_force_must_be_larger_at_the_shorter_length(material):
    requirements = make_requirements(material, length1=60, force1=50, length2=90, force2=200)
    with pytest.raises(ValueError):
        CompressionSpringInverseDesigner(requirements)


def test_no_valid_candidate_raises(material):
    requirements = make_requirements(material)
    # Restricting to a single, very thin wire diameter makes the required
    # active-coil count (and therefore geometry) infeasible for this rate.
    designer = CompressionSpringInverseDesigner(requirements, wire_diameter_bounds=(0.05, 0.06))
    with pytest.raises(ValueError):
        designer.design()


def test_calculate_spring(material:Material):
    load1 = 400 * ureg.N
    load2 = 40 * ureg.N
    pos1 = 65 * ureg.mm
    pos2 = 120 * ureg.mm
    req = Requirements(material=material,
                       length1=pos1,
                       length2=pos2,
                       force1=load1,
                       force2=load2,
                       safety_factor=1.5)
    designer = CompressionSpringInverseDesigner(requirements=req)
    best_spring = designer.design()
    report = SpringPDFReport(spring=best_spring.spring,
                             title="Best spring found")
    report.build(output_path="test_report.pdf")


if __name__ == "__main__":
    mat = Material(material_name="SL")
    test_design_matches_target_spring_constant_exactly(mat)
    test_design_hits_safety_factor_target(mat)
    test_design_reproduces_the_two_load_points(mat)
    test_design_spring_index_within_bounds(mat)
    test_design_returns_geometry_that_does_not_bind_before_the_shortest_length(mat)
    test_unreachable_safety_factor_returns_closest_achievable(mat)
    print("All lineal_comp_inv_claude tests passed.")
    test_calculate_spring(material=mat)