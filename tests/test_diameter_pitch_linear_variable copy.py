"""Test linear diameter and pitch functions for a spring."""
import numpy as np
from pint import Quantity

from springcalc.lineal.plotting import interactive_backend
from springcalc.pymodels.material import Material
from springcalc.pymodels.units import ureg
from springcalc.lineal.generic_compression import CompressionSpringGeneral
from matplotlib import pyplot as plt
from springcalc.report.pdf_report import SpringPDFReport
from springcalc.lineal.animation import CompressionAnimator


def linear_diameter(x: Quantity,
                    min_diameter: Quantity,
                    max_diameter: Quantity,
                    free_length: Quantity) -> Quantity:
    if not isinstance(x, Quantity):
        x = x * ureg.mm
    if x.magnitude < 0 or x.magnitude > free_length.magnitude:
        raise ValueError("x must be between 0 and free_length")
    diameter = min_diameter + (max_diameter - min_diameter) * (x / free_length)
    return diameter


def linear_pitch(x: Quantity,
                 min_pitch: Quantity,
                 max_pitch: Quantity,
                 free_length: Quantity) -> Quantity:
    if not isinstance(x, Quantity):
        x = x * ureg.mm
    if x.magnitude < 0 or x.magnitude > free_length.magnitude:
        raise ValueError("x must be between 0 and free_length")
    pitch = max_pitch + (min_pitch - max_pitch) * (x / free_length)
    return pitch


def test_linear_diameter():
    diameter_values = []
    for i in range(130):
        x = i * ureg.mm
        diameter = linear_diameter(x,
                                   min_diameter=20 * ureg.mm,
                                   max_diameter=60 * ureg.mm,
                                   free_length=130 * ureg.mm)
        diameter_values.append([i, diameter.to('mm').magnitude])
    diameter_values = np.array(diameter_values)
    with interactive_backend(plt.show):
        plt.plot(diameter_values[:, 0], diameter_values[:, 1])
        plt.xlabel("Position (mm)")
        plt.ylabel("Diameter (mm)")
        plt.title("Variable Diameter")
        plt.show()

    material = Material(material_name="SH")
    spring = CompressionSpringGeneral(material=material, wire_diameter=2.0)
    spring.set_geometry(f_mean_diameter=lambda x: linear_diameter(x,
                                                         min_diameter=20 * ureg.mm,
                                                         max_diameter=60 * ureg.mm,
                                                         free_length=130 * ureg.mm),
                        f_pitch=lambda x: linear_pitch(x,
                                                      min_pitch=2.5 * ureg.mm,
                                                      max_pitch=5.0 * ureg.mm,
                                                      free_length=130 * ureg.mm),
                        free_length=130 * ureg.mm)
    spring.calculate_spring_properties()
    spring_data = spring.get_spring_data()
    print("✅ Linear-diameter CompressionSpringGeneral test, data of the generic class:")
    for key, value in spring_data.items():
        print(f"{key}: {value}")
    positions = [120, 100]
    for pos in positions:
        spring.add_load_position(length=pos * ureg.mm)

    deflection, force, stiffness = spring.simulate_progressive_compression(
        max_deflection=40 * ureg.mm, steps=20
    )
    spring.get_progressive_compression_graph(deflection, force, show=True)
    spring_data = spring.get_3d_plot(show=True)
    report = SpringPDFReport(spring=spring, title="Variable Diameter and Pitch Spring Report")
    report.build(output_path="linear_diameter_pitch_spring_report.pdf")
    animator = CompressionAnimator(spring)
    animator.create_gif(
        max_deflection=40 * ureg.mm, output_path="progressive_compression.gif",
        steps=40, num_points=300, fps=12,
    )


if __name__ == "__main__":
    test_linear_diameter()
