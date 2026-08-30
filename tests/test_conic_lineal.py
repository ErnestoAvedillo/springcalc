from springcalc.pymodels.material import Material
from springcalc.lineal.generic_compression import CompressionSpringGeneral
from springcalc.lineal.animation import CompressionAnimator
from springcalc.report.pdf_report import SpringPDFReport
from pint import Quantity


from springcalc.pymodels.units import ureg


def linear(x: Quantity, upper: Quantity, lower: Quantity, free_length: Quantity):
    return lower + (upper - lower) * (x / free_length)


def main():
    material = Material(material_name="DH")
    spring = CompressionSpringGeneral(material=material, wire_diameter=2 * ureg.mm)
    free_length = 70
    spring.set_geometry(f_mean_diameter=lambda x: linear(x,
                                                upper=20 * ureg.mm,
                                                lower=60 * ureg.mm,
                                                free_length=free_length * ureg.mm),
                        f_pitch=lambda x: linear(x,
                                                upper=10 * ureg.mm,
                                                lower=3 * ureg.mm,
                                                free_length=free_length * ureg.mm),
                        free_length=free_length * ureg.mm)
    spring.calculate_spring_properties()
    posiciones = [35, 40, 50, 60]
    for pos in posiciones:
        spring.add_load_position(length=pos * ureg.mm)
    animator = CompressionAnimator(spring=spring)
    animator.create_gif(max_deflection=40 * ureg.mm, output_path="animator.gif")
    report = SpringPDFReport(spring)
    report.build("report.pdf")


if __name__ == "__main__":
    main()
