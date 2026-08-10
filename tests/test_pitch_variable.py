import math
import numpy as np
import pytest
from pint import Quantity

from springcalc.lineal.plotting import interactive_backend
from springcalc.pymodels.material import Material
from springcalc.pymodels.units import ureg
from springcalc.lineal.generic_compression import CompressionSpringGeneral
from springcalc.lineal.compresion import CompressionSpring
from matplotlib import pyplot as plt


def _build_constant_geometry_spring():
    """A constant-diameter, constant-pitch spring should behave like a regular one."""
    material = Material(material_name="SH")
    spring = CompressionSpringGeneral(material=material, wire_diameter=2.0)
    spring.set_geometry(func_D=lambda x: 20 * ureg.mm, func_p=lambda x: 6 * ureg.mm, free_length=60 * ureg.mm)
    std_spring = CompressionSpring(material=material, wire_diameter=2.0)
    std_spring.set_geometry(mean_diameter=20 * ureg.mm, pitch=6 * ureg.mm, free_length=60 * ureg.mm)
    positions = [30, 50]
    for pos in positions:
        spring.add_load_position(length=pos * ureg.mm)
        std_spring.add_load_position(length=pos * ureg.mm)
    spring.calculate_spring_properties()
    spring_data = spring.get_spring_data()
    std_spring_data = std_spring.get_spring_data()
    print("✅ Constant-geometry CompressionSpringGeneral test, data of the generic class:")
    for key, value in spring_data.items():
        print(f"{key}: {value}")
    print("✅ Constant-geometry CompressionSpring test, data of the standard class:")
    for key, value in std_spring_data.items():
        print(f"{key}: {value}")
    print("✅ Comparing CompressionSpringGeneral and CompressionSpring data:")
    # spring_constant is excluded: CompressionSpringGeneral computes it over the
    # full winding (no end-coil discount), while the legacy CompressionSpring
    # still discounts inactive end coils, so the two are expected to differ.
    for key, value in spring_data.items():
        if key == 'spring_constant':
            continue
        std_value = std_spring_data[key]
        print(f"{key}: {value}, std: {std_value}")
        if isinstance(value, Quantity):
            assert value.magnitude == pytest.approx(std_value.to(value.units).magnitude, rel=0.03)
        else:
            assert value == pytest.approx(std_value, rel=0.03)
    return spring


def test_theta_max_matches_coil_count():
    spring = _build_constant_geometry_spring()
    theta_max = spring.calculate_theta_max()

    assert theta_max == pytest.approx(2 * math.pi * 10)
    assert spring.nr_coils == pytest.approx(10)


def test_2K_constant_pitch(x: Quantity):
    if not isinstance(x, Quantity):
        x = x * ureg.mm
    if x.magnitude < 0 or x.magnitude > 100:
        raise ValueError("x must be between 0 and 100 mm")
    if x.magnitude < 30:
        pitch = 8.0 * ureg.mm
    elif x.magnitude < 60:
        pitch = 2.5 * ureg.mm
    else:
        pitch = 8.0 * ureg.mm
    return pitch


def test_simulate_progressive_compression():
    spring = _build_constant_geometry_spring()

    spring.set_geometry(func_D=lambda x: 20 * ureg.mm,
                        func_p=lambda x: test_2K_constant_pitch(x),
                        free_length=100 * ureg.mm)

    deflection, force, stiffness = spring.simulate_progressive_compression(
        max_deflection=40 * ureg.mm, steps=20
    )

    # get_progressive_compression_graph returns a base64 PNG (same convention
    # as CompressionSpring's other graph methods, for embedding in HTML/PDF).
    # For local debugging, decode it and write it to disk to look at it,
    # since plt.show() can't open a window under the "Agg" backend.
    spring.get_progressive_compression_graph(deflection, force, show=True)


def cuadratic_middle_pitch(x: Quantity,
                           min_pitch: Quantity,
                           max_pitch: Quantity,
                           free_length: Quantity) -> Quantity:
    if not isinstance(x, Quantity):
        x = x * ureg.mm
    if x.magnitude < 0 or x.magnitude > free_length.magnitude:
        raise ValueError("x must be between 0 and free_length")
    a = 2 * (max_pitch - min_pitch) / free_length
    pitch = a * (x - x ** 2 / free_length) + min_pitch
    # pitch = 2.5 * ureg.mm + (x.magnitude / 100) * (10.0 - 2.5) * ureg.mm
    return pitch


def progressive_pitch(x: Quantity,
                      min_pitch: Quantity,
                      max_pitch: Quantity,
                      free_length: Quantity) -> Quantity:
    if not isinstance(x, Quantity):
        x = x * ureg.mm
    if x.magnitude < 0 or x.magnitude > free_length.magnitude:
        raise ValueError("x must be between 0 and free_length")
    # generate a linear pitch
    pitch = min_pitch + (x.magnitude / free_length.magnitude) * (max_pitch - min_pitch)
    return pitch


def cuadratic_pitch(x: Quantity,
                    min_pitch: Quantity,
                    max_pitch: Quantity,
                    free_length: Quantity,
                    alfa: float,
                    beta: float) -> Quantity:
    if not isinstance(x, Quantity):
        x = x * ureg.mm
    if x.magnitude < 0 or x.magnitude > free_length.magnitude:
        raise ValueError("x must be between 0 and free_length")
    a = (max_pitch * (beta - alfa) + min_pitch * (alfa - 1)) / (free_length ** 2 * (alfa - 1) * alfa)
    b = (max_pitch * (alfa**2 - beta) - min_pitch * (alfa - 1)**2) / (free_length * (alfa - 1) * alfa)
    # minimum_pos = - b.magnitude / (2 * a.magnitude)
    # if x.magnitude < minimum_pos * free_length.magnitude:
    #     x = minimum_pos * ureg.mm
    pitch = a * x ** 2 + b * x + min_pitch
    return pitch


def test_progressive_pitch():
    pitch_values = []
    for i in range(130):
        x = i * ureg.mm
        pitch = cuadratic_pitch(x,
                                min_pitch=2.1 * ureg.mm,
                                max_pitch=8.0 * ureg.mm,
                                free_length=130 * ureg.mm,
                                alfa=0.7,
                                beta=0.3)
        print(f"x: {x}, pitch: {pitch}")
        pitch_values.append([i, pitch.to('mm').magnitude])
    pitch_values = np.array(pitch_values)
    with interactive_backend(plt.show):
        plt.plot(pitch_values[:, 0], pitch_values[:, 1])
        plt.xlabel("Position (mm)")
        plt.ylabel("Pitch (mm)")
        plt.title("Progressive Pitch")
        plt.show()
    material = Material(material_name="SH")
    spring = CompressionSpringGeneral(material=material, wire_diameter=2.0)

    spring.set_geometry(func_D=lambda x: 20 * ureg.mm,
                        func_p=lambda x: cuadratic_pitch(x,
                                                         min_pitch=2.1 * ureg.mm,
                                                         max_pitch=8.0 * ureg.mm,
                                                         free_length=130 * ureg.mm,
                                                         alfa=0.7,
                                                         beta=0.3),
                        free_length=130 * ureg.mm)
    spring.calculate_spring_properties()
    spring_data = spring.get_spring_data()
    print("✅ Progressive-pitch CompressionSpringGeneral test, data of the generic class:")
    for key, value in spring_data.items():
        print(f"{key}: {value}")
    positions = [120, 100]
    for pos in positions:
        spring.add_load_position(length=pos * ureg.mm)

    deflection, force, stiffness = spring.simulate_progressive_compression(
        max_deflection=40 * ureg.mm, steps=20
    )

    # get_progressive_pitch_graph returns a base64 PNG (same convention
    # as CompressionSpring's other graph methods, for embedding in HTML/PDF).
    # For local debugging, decode it and write it to disk to look at it,
    # since plt.show() can't open a window under the "Agg" backend.
    spring.get_progressive_compression_graph(deflection, force, show=True)


if __name__ == "__main__":
    _build_constant_geometry_spring()
    test_theta_max_matches_coil_count()
    test_simulate_progressive_compression()
    test_progressive_pitch()
