import math
import warnings

import numpy as np
import pytest

from springcalc.lineal.generic_compression import CompressionSpringGeneral
from springcalc.pymodels.material import Material
from springcalc.pymodels.units import ureg


def _spring(f_pitch, free_length_mm=60.0, mean_diameter_mm=20.0, wire_diameter=2.0):
    spring = CompressionSpringGeneral(material=Material(material_name="SH"), wire_diameter=wire_diameter)
    spring.set_geometry(f_mean_diameter=lambda h: mean_diameter_mm * ureg.mm,
                        f_pitch=f_pitch,
                        free_length=free_length_mm * ureg.mm)
    return spring


def test_h_theta_matches_closed_form_for_linear_pitch():
    """p(h) = p0 + s*h  ->  theta(h) = (2*pi/s) * ln(p(h)/p0), so h(theta) is explicit."""
    p0, slope, length = 4.0, 0.05, 60.0
    spring = _spring(lambda h: (p0 + slope * h.magnitude) * ureg.mm, free_length_mm=length)
    theta_max = spring.calculate_theta_max()

    assert theta_max == pytest.approx((2 * math.pi / slope) * math.log((p0 + slope * length) / p0), rel=1e-6)

    thetas, zs = spring.get_h_theta_development(200)
    expected = p0 * (np.exp(thetas * slope / (2 * math.pi)) - 1.0) / slope
    assert zs == pytest.approx(expected, abs=1e-4)
    assert zs[0] == 0.0
    assert zs[-1] == pytest.approx(length, abs=1e-4)


def test_h_theta_with_pitch_jumps_is_exact_and_warning_free():
    """Piecewise-constant pitch (closed end zones next to a wider working pitch)."""
    end_pitch, mid_pitch, end_zone, length = 2.0, 8.0, 5.0, 60.0

    def f_pitch(h):
        x = h.magnitude
        return (end_pitch if x < end_zone or x > length - end_zone else mid_pitch) * ureg.mm

    spring = _spring(f_pitch, free_length_mm=length)

    with warnings.catch_warnings():
        # quad complaining about the discontinuity (IntegrationWarning) or a
        # root-find stalling (RuntimeWarning) used to be the symptom of this case.
        warnings.simplefilter("error")
        theta_max = spring.calculate_theta_max()
        thetas, zs = spring.get_h_theta_development(300)

    coils_expected = 2 * end_zone / end_pitch + (length - 2 * end_zone) / mid_pitch
    assert spring.nr_coils == pytest.approx(coils_expected, rel=1e-6)
    assert theta_max == pytest.approx(2 * math.pi * coils_expected, rel=1e-6)

    assert np.all(np.diff(zs) >= 0.0)
    assert zs[-1] == pytest.approx(length, abs=1e-3)
    # The first closed zone is exactly end_zone / end_pitch turns long. Ask for
    # that angle directly: interpolating the 300-point curve above would smear
    # the kink between two samples. The tabulated inverse is only approximate
    # inside the single table cell that holds the jump (60 mm / 2000 = 0.03 mm).
    _, z_zone_end = spring.get_h_theta_development(2, theta_end=2 * math.pi * end_zone / end_pitch)
    assert z_zone_end[-1] == pytest.approx(end_zone, abs=0.05)


def test_h_theta_development_evaluates_pitch_sparingly():
    """The whole property calculation must not solve h(theta) point by point.

    That used to cost ~10^5 evaluations of the user's pitch function (each one
    a pint-heavy Python call) and minutes for pitch profiles with jumps.
    """
    calls = [0]

    def f_pitch(h):
        calls[0] += 1
        return (6.0 + 0.02 * h.magnitude) * ureg.mm

    spring = _spring(f_pitch)
    spring.calculate_spring_properties()  # wire length + spring constant: 2 x 500 points
    spring.get_h_theta_development(500)   # already tabulated: free

    assert calls[0] < 10_000


def test_theta_table_follows_free_length_reassignment():
    """Callers may reassign free_length after computing (theta_max stays cached)."""
    spring = _spring(lambda h: 6.0 * ureg.mm, free_length_mm=60.0)
    thetas, zs = spring.get_h_theta_development(10, theta_end=2 * math.pi * 10)
    assert zs[-1] == pytest.approx(60.0)

    spring.free_length = 40.0 * ureg.mm
    thetas, zs = spring.get_h_theta_development(10, theta_end=2 * math.pi * 10)
    assert zs[-1] == pytest.approx(40.0)


def test_non_positive_pitch_is_rejected():
    spring = _spring(lambda h: (5.0 - 0.1 * h.magnitude) * ureg.mm, free_length_mm=60.0)  # hits 0 at h=50
    with pytest.raises(ValueError, match="greater than zero"):
        spring.get_h_theta_development(50, theta_end=10.0)
