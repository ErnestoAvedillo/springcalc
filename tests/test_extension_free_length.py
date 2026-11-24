import pytest

from springcalc import ExtensionSpring, Material
from springcalc.pymodels.units import ureg


def _spring():
    return ExtensionSpring(material=Material(material_name="SH"), wire_diameter=1.5)


def test_set_geometry_uses_given_free_length():
    spring = _spring()
    spring.set_geometry(outer_diameter=15, nr_coils=10, pitch=1.5, free_length=40)
    assert spring.free_length.to("mm").magnitude == pytest.approx(40)
    assert spring.length_btw_hooks.to("mm").magnitude == pytest.approx(40)

    # Travel and load are measured from the 40 mm free length, not the 15 mm body.
    spring.add_load_position(50)
    position = spring.get_data_positions()[0]
    assert position.travel.to("mm").magnitude == pytest.approx(10)
    expected_load = (spring.spring_constant * 10 * ureg.mm).to("N").magnitude
    assert position.load.to("N").magnitude == pytest.approx(expected_load)


def test_free_length_defaults_to_body_length():
    spring = _spring()
    spring.set_diameter(outer_diameter=15)
    spring.calculate_spring_properties(nr_coils=10)  # pitch defaults to wire diameter
    assert spring.free_length.to("mm").magnitude == pytest.approx(15)


def test_free_length_shorter_than_body_raises():
    spring = _spring()
    spring.set_diameter(outer_diameter=15)
    with pytest.raises(ValueError, match="cannot be shorter"):
        spring.calculate_spring_properties(nr_coils=10, pitch=1.5, free_length=10)
