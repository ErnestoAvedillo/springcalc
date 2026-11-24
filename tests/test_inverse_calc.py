
from springcalc.inverse_calc.conical_curve_comp_inv import CompressionCurveRequirements
from springcalc.inverse_calc.conical_curve_comp_inv import ConicalCurveCompressionSpringInverseDesigner
from springcalc.report.pdf_report import SpringPDFReport
from springcalc.pymodels.material import Material
from springcalc.persistence import save_model, load_model
from pathlib import Path

import pytest

DATA_CSV = Path(__file__).parent / "data.csv"


@pytest.fixture
def material():
    return Material(material_name="DH")


@pytest.fixture
def target_curve_csv():
    return str(DATA_CSV)


def test_full_curve_fit_design_reproduces_target_curve_shape(material, target_curve_csv, tmp_path):
    # Exercises the full regression pipeline: differential_evolution search
    # (fast closed-form simulator) + snap-to-standard-diameter + local
    # refine + final build with the real CompressionSpringGeneral. This is
    # the slow part of this test (the final build alone takes ~20-30s).
    requirements = CompressionCurveRequirements(material=material, safety_factor=1.5, csv_path=target_curve_csv)
    designer = ConicalCurveCompressionSpringInverseDesigner(requirements, seed=0)
    result = designer.design()
    report = SpringPDFReport(spring=result.spring, title="Conic result")
    report.build(output_path=str(tmp_path / "Report conical variable.pdf"))
    save_model(result, str(tmp_path / "conical_result.json"), include_results=True, profile_points=50)
    loaded_spring = load_model(str(tmp_path / "conical_result.json"))
    report = SpringPDFReport(spring=loaded_spring, title="Loaded Conic result")
    report.build(output_path=str(tmp_path / "Loaded Report conical variable.pdf"))
    assert result.curve_rmse_relative < 0.2
    # safety_factor_weight makes this a *soft* term competing with curve
    # RMSE and geometry penalties in the same objective (by design -- see
    # the module docstring), so it lands in the neighborhood of the target
    # rather than tracking it tightly; a generous tolerance here checks it's
    # influencing the search at all without over-constraining a soft term.
    assert result.safety_factor == pytest.approx(requirements.safety_factor, abs=0.3)
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
    import tempfile
    test_full_curve_fit_design_reproduces_target_curve_shape(
        Material(material_name="DH"), str(DATA_CSV), Path(tempfile.mkdtemp()))
    print("All conical_curve_comp_inv_claude fast-path tests passed.")
