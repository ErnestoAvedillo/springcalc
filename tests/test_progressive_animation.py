"""Test progressive compression animation of a spring
with variable diameter and pitch."""

from pint import Quantity

from springcalc.pymodels.material import Material
from springcalc.pymodels.units import ureg
from springcalc.lineal.generic_compression import CompressionSpringGeneral
from springcalc.lineal.animation import CompressionAnimator
from springcalc.report.pdf_report import SpringPDFReport


def cuadratic_diameter(x: Quantity,
                       min_diameter: Quantity,
                       max_diameter: Quantity,
                       free_length: Quantity) -> Quantity:
    if not isinstance(x, Quantity):
        x = x * ureg.mm
    if x.magnitude < 0 or x.magnitude > free_length.magnitude:
        raise ValueError("x must be between 0 and free_length")
    aux = 4 * (min_diameter - max_diameter) / free_length
    diameter = aux * (x ** 2 / free_length - x) + min_diameter
    return diameter


def cuadratic_pitch(x: Quantity,
                    min_pitch: Quantity,
                    max_pitch: Quantity,
                    free_length: Quantity) -> Quantity:
    if not isinstance(x, Quantity):
        x = x * ureg.mm
    if x.magnitude < 0 or x.magnitude > free_length.magnitude:
        raise ValueError(f"x is {x} and must be between 0 and free_length")
    aux = 4 * (min_pitch - max_pitch) / free_length
    pitch = aux * (x ** 2 / free_length - x) + min_pitch
    return pitch


def test_progressive_compression_animation():
    """Render the coils closing up under progressive load as an animated GIF.

    For local debugging, the GIF is written next to the test so it can be
    opened directly, since there's no "show" window for an animation the way
    there is for a static plot.
    """
    material = Material(material_name="SH")
    spring = CompressionSpringGeneral(material=material, wire_diameter=2.0)
    spring.set_geometry(f_mean_diameter=lambda x: cuadratic_diameter(x,
                                                            min_diameter=20 * ureg.mm,
                                                            max_diameter=60 * ureg.mm,
                                                            free_length=130 * ureg.mm),
                        f_pitch=lambda x: cuadratic_pitch(x,
                                                         min_pitch=2.5 * ureg.mm,
                                                         max_pitch=5.0 * ureg.mm,
                                                         free_length=130 * ureg.mm),
                        free_length=130 * ureg.mm)
    spring.calculate_spring_properties()

    animator = CompressionAnimator(spring)
    output_path = animator.create_gif(
        max_deflection=40 * ureg.mm, output_path="progressive_compression.gif",
        steps=40, num_points=300, fps=12,
    )
    positions = [120, 100]
    for pos in positions:
        spring.add_load_position(length=pos * ureg.mm)
    print(f"✅ Progressive compression animation written to {output_path}")
    spring_data = spring.get_spring_data()
    print("Data of the generic class:")
    for key, value in spring_data.items():
        print(f"{key}: {value}")
    spring.get_3d_plot(show=True)
    report = SpringPDFReport(spring=spring, title="Progressive Compression Animation Report")
    report.build(output_path="progressive_compression_animation_report.pdf")


if __name__ == "__main__":
    test_progressive_compression_animation()
