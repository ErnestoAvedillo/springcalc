import numpy as np
import pytest

from springcalc import Material, spring_centerline_points, spring_geometry_to_step
from springcalc.lineal.generic_compression import CompressionSpringGeneral
from springcalc.pymodels.units import ureg


def _conical_spring():
    spring = CompressionSpringGeneral(material=Material(material_name="SH"), wire_diameter=2.0)
    spring.set_geometry(
        f_mean_diameter=lambda h: (30 - 10 * h.to("mm").magnitude / 60) * ureg.mm,
        f_pitch=lambda h: 6 * ureg.mm,
        free_length=60 * ureg.mm,
    )
    spring.calculate_spring_properties()
    return spring


def test_free_centerline_spans_free_length():
    points = spring_centerline_points(_conical_spring(), num_points=200)
    assert points.shape == (200, 3)
    assert points[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert points[-1, 2] == pytest.approx(60.0, rel=1e-3)


def test_compressed_centerline_reaches_requested_length_and_keeps_radii():
    spring = _conical_spring()
    free = spring_centerline_points(spring, num_points=200)
    compressed = spring_centerline_points(spring, num_points=200, length=40 * ureg.mm, steps=40)

    assert compressed[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert compressed[:, 2].max() == pytest.approx(40.0, rel=1e-3)
    np.testing.assert_allclose(np.hypot(compressed[:, 0], compressed[:, 1]),
                               np.hypot(free[:, 0], free[:, 1]), rtol=1e-9)


def test_plain_number_length_is_mm():
    spring = _conical_spring()
    a = spring_centerline_points(spring, num_points=100, length=45, steps=20)
    b = spring_centerline_points(spring, num_points=100, length=45 * ureg.mm, steps=20)
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("length", [5 * ureg.mm, 70 * ureg.mm])
def test_length_out_of_range_raises(length):
    with pytest.raises(ValueError, match="length must be between"):
        spring_centerline_points(_conical_spring(), num_points=100, length=length)


def test_step_file_with_centerline_at_compressed_length(tmp_path):
    cq = pytest.importorskip("cadquery")
    spring = _conical_spring()
    path = spring_geometry_to_step(spring, str(tmp_path / "spring.step"), num_points=120,
                                   length=40 * ureg.mm, steps=20)
    shape = cq.importers.importStep(path).val()
    assert len(shape.Edges()) > 0
    z_max = shape.BoundingBox().zmax
    assert z_max == pytest.approx(40.0 + 1.0, abs=0.5)  # top of wire = centerline + wire radius
    # The spline centerline is a standalone edge, not part of any solid
    solid_edges = {e.hashCode() for s in shape.Solids() for e in s.Edges()}
    assert any(e.hashCode() not in solid_edges and e.geomType() == "BSPLINE" for e in shape.Edges())
