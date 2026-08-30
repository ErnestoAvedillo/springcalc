import pytest

from springcalc import Material, CompressionSpring


def test_set_geometry_matches_separate_calls():
    material = Material(material_name="SL")

    combined = CompressionSpring(material=material, wire_diameter=2.5)
    combined.set_geometry(outer_diameter=30, pitch=20, free_length=100)

    separate = CompressionSpring(material=material, wire_diameter=2.5)
    separate.set_diameter(outer_diameter=30)
    separate.calculate_spring_properties(pitch=20, free_length=100)

    assert combined.get_spring_data() == separate.get_spring_data()


def test_set_geometry_returns_spring_data():
    material = Material(material_name="SL")
    spring = CompressionSpring(material=material, wire_diameter=2.5)

    data = spring.set_geometry(outer_diameter=30, pitch=20, free_length=100)

    assert data["mean_diameter"].to("mm").magnitude == pytest.approx(27.5)
    # nr_coils = (free_length - end_length) / pitch + inactive_coils, using the
    # default (unspecified) type_of_end 'closed_ground': end_length = 1.5 *
    # wire_diameter = 3.75mm, inactive_coils = 2 -> (100 - 3.75)/20 + 2.
    assert data["nr_coils"] == pytest.approx(6.8125)


def test_set_geometry_rejects_more_than_one_diameter():
    material = Material(material_name="SL")
    spring = CompressionSpring(material=material, wire_diameter=2.5)

    with pytest.raises(ValueError):
        spring.set_geometry(outer_diameter=30, mean_diameter=27.5, pitch=20, free_length=100)


def test_set_geometry_rejects_wrong_number_of_length_params():
    material = Material(material_name="SL")
    spring = CompressionSpring(material=material, wire_diameter=2.5)

    with pytest.raises(ValueError):
        spring.set_geometry(outer_diameter=30, nr_coils=5, pitch=20, free_length=100)
