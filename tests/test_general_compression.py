from springcalc.lineal.generic_compression import CompressionSpringGeneral
from springcalc.lineal.animation import CompressionAnimator
from springcalc import ureg
from springcalc.pymodels.material import Material


Quantity = ureg.Quantity


def Diameter(diameter_init: Quantity,
             diameter_end: Quantity,
             free_length: Quantity,
             h: Quantity) -> Quantity:
    diameter = diameter_init + (diameter_end - diameter_init) * (h.to('mm').magnitude / free_length.to('mm').magnitude)
    return diameter


def get_pitch(pitch_init: Quantity,
              pitch_end: Quantity,
              free_length: Quantity,
              h: Quantity,
              wire_diameter: Quantity,
              ) -> Quantity:
    if h < wire_diameter:
        return wire_diameter
    elif h >= wire_diameter and h < wire_diameter + pitch_init:
        max_dim_sector = pitch_init.to('mm').magnitude
        position_in_sector = h.to('mm').magnitude - wire_diameter.to('mm').magnitude
        return wire_diameter + (pitch_init - wire_diameter) * (position_in_sector) / max_dim_sector
    elif h >= wire_diameter + pitch_init and h < free_length - wire_diameter - pitch_end:
        max_dim_sector = free_length.to('mm').magnitude - 2 * wire_diameter.to('mm').magnitude - pitch_init.to('mm').magnitude - pitch_end.to('mm').magnitude
        position_in_sector = h.to('mm').magnitude - wire_diameter.to('mm').magnitude - pitch_init.to('mm').magnitude
        return pitch_init + (pitch_end - pitch_init) * (position_in_sector) / max_dim_sector
    elif h >= free_length - wire_diameter - pitch_end and h < free_length - wire_diameter:
        max_dim_sector = pitch_end.to('mm').magnitude
        position_in_sector = h.to('mm').magnitude - (free_length.to('mm').magnitude - wire_diameter.to('mm').magnitude - pitch_end.to('mm').magnitude)
        return pitch_end + (wire_diameter - pitch_end) * (position_in_sector) / max_dim_sector
    else:
        return wire_diameter

WIRE_DIAMETER = Quantity(2.9, 'mm')

material = Material(material_name="DH")

spring = CompressionSpringGeneral(material=material,
                                  wire_diameter=WIRE_DIAMETER,
                                  free_length=Quantity(100, 'mm'),
                                  nr_coils=10)

FREE_LENGTH = Quantity(100, 'mm')

spring.set_geometry(f_mean_diameter=lambda h: Diameter(diameter_init=Quantity(30, 'mm'),
                                                       diameter_end=Quantity(60, 'mm'),
                                                       free_length=FREE_LENGTH,
                                                       h=h),
                    f_pitch=lambda h: get_pitch(pitch_init=Quantity(10, 'mm'),
                                                pitch_end=Quantity(20, 'mm'),
                                                free_length=FREE_LENGTH,
                                                h=h,
                                                wire_diameter=WIRE_DIAMETER),
                    free_length=FREE_LENGTH)
POSITION_START = FREE_LENGTH.to('mm').magnitude - Quantity(10, 'mm').to('mm').magnitude
POSITION_END = Quantity(10, 'mm').to('mm').magnitude
for position in range(POSITION_START, POSITION_END, -5):
    spring.add_load_position(length=Quantity(position, 'mm'))

spring.get_forces_vs_position_graph(show=True)
spring.get_3d_plot(num_points=500, show=True)
anim = CompressionAnimator(spring)
anim.create_gif(max_deflection=Quantity(50, 'mm'), output_path="compression_animation.gif")