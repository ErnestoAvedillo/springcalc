import pytest

from springcalc import (
    CompressionSpring,
    CompressionSpringGeneral,
    ExtensionSpring,
    Material,
    TorsionSpring,
    export_model,
    import_model,
    load_model,
    model_from_json,
    model_to_json,
    save_model,
)
from springcalc.pymodels.units import ureg


def _make_compression_spring():
    material = Material(material_name="SL")
    spring = CompressionSpring(material=material, wire_diameter=2.5)
    spring.set_geometry(outer_diameter=30, pitch=20, free_length=100,
                        type_of_end="closed_ground")
    spring.type_conforming = "hot_formed"
    spring.shot_peening = True
    spring.coating = "zinc"
    spring.number_cycles = 12345
    spring.add_load_position(40)
    spring.add_load_position(60)
    return spring


def _make_conical_spring():
    material = Material(material_name="DH")
    spring = CompressionSpringGeneral(material=material, wire_diameter=2 * ureg.mm)
    free_length = 70 * ureg.mm

    def f_mean_diameter(h):
        u = h / free_length
        return 20 * ureg.mm + (60 - 20) * ureg.mm * u

    def f_pitch(h):
        u = h / free_length
        return 10 * ureg.mm + (3 - 10) * ureg.mm * u

    spring.set_geometry(f_mean_diameter=f_mean_diameter, f_pitch=f_pitch, free_length=free_length)
    spring.calculate_spring_properties()
    spring.add_load_position(40 * ureg.mm)
    spring.add_load_position(60 * ureg.mm)
    return spring


def _make_extension_spring():
    material = Material(material_name="SL")
    spring = ExtensionSpring(material=material, wire_diameter=1.5)
    spring.set_geometry(outer_diameter=15, nr_coils=20, pitch=1.5, free_length=100)
    spring.add_load_position(110)
    spring.add_load_position(130)
    return spring


def _make_torsion_spring():
    material = Material(material_name="SL")
    spring = TorsionSpring(material=material, wire_diameter=2.5)
    spring.set_geometry(mean_diameter=20.0, nr_coils=10, pitch=5.0,
                        free_angle=45.0, fixed_leg_radius=10.0, mobile_leg_radius=10.0)
    spring.add_position(angle_travel=10)
    spring.add_position(angle_travel=20)
    return spring


@pytest.mark.parametrize("factory", [
    _make_compression_spring,
    _make_extension_spring,
    _make_torsion_spring,
])
def test_round_trip_preserves_type_and_key_properties(factory):
    spring = factory()
    data = export_model(spring)

    assert data["spring_type"] == type(spring).__name__
    assert "results" in data

    rebuilt = import_model(data)

    assert type(rebuilt) is type(spring)
    original_units = str(spring.spring_constant.units)
    assert rebuilt.spring_constant.to(original_units).magnitude == pytest.approx(
        spring.spring_constant.to(original_units).magnitude, rel=1e-9
    )
    assert len(rebuilt.positions.positions) == len(spring.positions.positions)
    for original, restored in zip(spring.positions.positions, rebuilt.positions.positions):
        assert restored.load.magnitude == pytest.approx(original.load.magnitude, rel=1e-6)


def test_round_trip_conical_spring_reconstructs_geometry_profile():
    spring = _make_conical_spring()
    data = export_model(spring)
    rebuilt = import_model(data)

    assert isinstance(rebuilt, CompressionSpringGeneral)
    assert rebuilt.spring_constant.to("N/mm").magnitude == pytest.approx(
        spring.spring_constant.to("N/mm").magnitude, rel=1e-3
    )
    for h_mm in (0.0, 17.5, 35.0, 52.5, 70.0):
        h = h_mm * ureg.mm
        assert rebuilt.f_mean_diameter(h).to("mm").magnitude == pytest.approx(
            spring.f_mean_diameter(h).to("mm").magnitude, rel=1e-6
        )
        assert rebuilt.f_pitch(h).to("mm").magnitude == pytest.approx(
            spring.f_pitch(h).to("mm").magnitude, rel=1e-6
        )
    assert len(rebuilt.positions.positions) == len(spring.positions.positions)


def test_include_results_false_omits_results_but_still_round_trips():
    spring = _make_compression_spring()
    data = export_model(spring, include_results=False)

    assert "results" not in data

    rebuilt = import_model(data)
    assert rebuilt.spring_constant.to("N/mm").magnitude == pytest.approx(
        spring.spring_constant.to("N/mm").magnitude, rel=1e-9
    )


def test_json_round_trip():
    spring = _make_compression_spring()
    text = model_to_json(spring)
    rebuilt = model_from_json(text)

    assert type(rebuilt) is CompressionSpring
    assert rebuilt.spring_constant.to("N/mm").magnitude == pytest.approx(
        spring.spring_constant.to("N/mm").magnitude, rel=1e-9
    )


def test_save_and_load_model_file_round_trip(tmp_path):
    spring = _make_compression_spring()
    path = save_model(spring, str(tmp_path / "spring.json"))
    rebuilt = load_model(path)

    assert type(rebuilt) is CompressionSpring
    assert rebuilt.mean_diameter.to("mm").magnitude == pytest.approx(
        spring.mean_diameter.to("mm").magnitude, rel=1e-9
    )


def test_export_model_rejects_unsupported_type():
    with pytest.raises(TypeError):
        export_model(object())


def test_import_model_rejects_unknown_spring_type():
    with pytest.raises(ValueError):
        import_model({"spring_type": "NotASpring", "inputs": {}})
