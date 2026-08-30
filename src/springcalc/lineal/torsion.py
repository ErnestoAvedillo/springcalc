"""Torsion spring calculations."""
from math import pi
import io
import base64
import traceback
import numpy as np
from math import atan
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from springcalc.pymodels.material import Material
from springcalc.pymodels.wire_characteristics import WireCharacteristics
from typing import Optional
from springcalc.pymodels.positions import AngularPositionsTable
from pint import Quantity
from springcalc.pymodels.units import ureg
from pydantic import ConfigDict, field_validator
import matplotlib
from .goodman import GoodmanData, GoodmanAnalyzer
from .plotting import interactive_backend
matplotlib.use('Agg')


def _mag(v):
    """Extract the magnitude from a Quantity, or return the value as-is."""
    return v.magnitude if isinstance(v, Quantity) else v


class TorsionSpring(WireCharacteristics):
    model_config = ConfigDict(arbitrary_types_allowed=True,
                              validate_assignment=True)
    mean_diameter: Quantity = 0.0 * ureg.mm
    inner_diameter: Quantity = 0.0 * ureg.mm
    outer_diameter: Quantity = 0.0 * ureg.mm
    free_angle: Quantity = 0.0 * ureg.degree
    tangency_angle: Quantity = 0.0 * ureg.degree
    angle_travel: Quantity = 0.0 * ureg.degree
    angle_position: Quantity = 0.0 * ureg.degree
    torque_position: Quantity = 0.0 * ureg.N * ureg.mm
    stress_position: Quantity = 0.0 * ureg.MPa
    pitch: Quantity = 0.0 * ureg.mm
    shot_peening: bool = False
    # Number of cycles for fatigue analysis, default 1 million
    number_cycles: int = 1e6
    nr_active_coils: float = 0.0
    nr_coils: int = 0
    spring_width: Quantity = 0.0 * ureg.mm  # in mm
    fixed_leg_radius: Quantity = 0.0 * ureg.mm  # in mm
    fixed_leg_length: Quantity = 0.0 * ureg.mm  # in mm
    mobile_leg_radius: Quantity = 0.0 * ureg.mm  # in mm
    mobile_leg_length: Quantity = 0.0 * ureg.mm  # in mm
    spring_index: float = 0.0  # wahl_factor: float = 0.0  # Wahl factor
    wahl_factor_category: Optional[str] = None  # Wahl factor category
    wahl_factor_eval: Optional[float] = None  # evaluated Wahl factor
    wahl_factor: float = 0.0  # Wahl factor
    wire_length_total: Quantity = 0.0 * ureg.mm  # in mm
    body_wire_length: Quantity = 0.0 * ureg.mm  # in mm
    spring_constant: Quantity = 0.0 * ureg.N * ureg.mm / ureg.rad  # in Nmm/rad
    resisting_moment: Quantity = 0.0 * ureg.mm * ureg.mm * ureg.mm * ureg.mm  # in Nmm
    # List of positions for fatigue analysis
    positions: AngularPositionsTable = AngularPositionsTable()

    @field_validator('mean_diameter',
                     'outer_diameter',
                     'inner_diameter',
                     'pitch',
                     'spring_width',
                     'fixed_leg_radius',
                     'fixed_leg_length',
                     'mobile_leg_radius',
                     'mobile_leg_length',
                     'wire_length_total',
                     'body_wire_length',
                     mode='before')
    @classmethod
    def validate_positive_quantities(cls, v):
        """Validate that the quantities are positive and have the correct units"""
        if v is not None:
            try:
                if isinstance(v, Quantity):
                    quantity = v.to('mm')
                elif isinstance(v, (int, float)):
                    quantity = v * ureg.mm
                else:
                    quantity = ureg(v).to('mm')
                if quantity.magnitude < 0:
                    raise ValueError(f'{quantity} must be a positive value with valid units.')
                return quantity
            except Exception as e:
                raise ValueError(f'Error validating quantity: {e}')
        return quantity

    @field_validator('free_angle', 'tangency_angle', 'angle_position', 'angle_travel', mode='before')
    @classmethod
    def validate_positive_angles(cls, v):
        """Validate that the angles are positive and have the correct units"""
        if v is not None:
            try:
                if isinstance(v, Quantity):
                    quantity = v.to('degree')
                elif isinstance(v, (int, float)):
                    quantity = v * ureg.degree
                else:
                    quantity = ureg(v).to('degree')
                if quantity.magnitude < 0:
                    quantity = 360 * ureg.degree + quantity
                return quantity
            except Exception as e:
                raise ValueError(f'Error validating angle: {e}')
        return quantity

    @field_validator('spring_constant', mode='before')
    @classmethod
    def validate_positive_spring_constant(cls, v):
        """Validate that the spring constant is positive"""
        if v is not None:
            try:
                if isinstance(v, Quantity):
                    quantity = v.to('N*mm/rad')
                elif isinstance(v, (int, float)):
                    quantity = v * ureg('N*mm/rad')
                else:
                    quantity = ureg(v).to('N*mm/rad')
                if quantity.magnitude <= 0:
                    raise ValueError(f'{v} must be a positive value')
                return quantity
            except Exception as e:
                raise ValueError(f'Error validating spring constant: {e}')
        return quantity

    @field_validator('resisting_moment', mode='before')
    @classmethod
    def validate_positive_resisting_moment(cls, v):
        """Validate the resisting moment"""
        if v is not None:
            try:
                if isinstance(v, Quantity):
                    quantity = v.to('mm**4')
                elif isinstance(v, (int, float)):
                    quantity = v * ureg('mm**4')
                else:
                    quantity = ureg(v).to('mm**4')
                if quantity.magnitude <= 0:
                    raise ValueError(f'{v} must be a positive value')
                return quantity
            except Exception as e:
                raise ValueError(f'Error validating resisting moment: {e}')
        return quantity

    @field_validator('torque_position', mode='before')
    @classmethod
    def validate_torque_position(cls, v):
        """Validate the torque position"""
        if v is not None:
            try:
                if isinstance(v, Quantity):
                    quantity = v.to('N*mm')
                elif isinstance(v, (int, float)):
                    quantity = v * ureg('N*mm')
                else:
                    quantity = ureg(v).to('N*mm')
                if quantity.magnitude < 0:
                    raise ValueError(f'{v} must be a positive value')
                return quantity
            except Exception as e:
                raise ValueError(f'Error validating torque position: {e}')
        return quantity

    @field_validator('stress_position', mode='before')
    @classmethod
    def validate_stress_position(cls, v):
        """Validate the stress position"""
        if v is not None:
            try:
                if isinstance(v, Quantity):
                    quantity = v.to('MPa')
                elif isinstance(v, (int, float)):
                    quantity = v * ureg('MPa')
                else:
                    quantity = ureg(v).to('MPa')
                if quantity.magnitude < 0:
                    raise ValueError(f'{v} must be a positive value')
                return quantity
            except Exception as e:
                raise ValueError(f'Error validating stress position: {e}')
        return quantity

    def __init__(self, material: Material, wire_diameter: float, **data):
        data.update({
            'material': material,
            'wire_diameter': wire_diameter,
        })
        super().__init__(**data)
        self.number_cycles = 1e6  # Default value of 1 million cycles
        self.shot_peening = False  # Default value, no shot peening
        self.nr_active_coils = 0.0
        self.nr_coils = 0
        self.spring_index = 0.0  # wahl_factor: float = 0.0  # Wahl factor
        self.wahl_factor_category = None  # Wahl factor category
        self.wahl_factor_eval = None  # evaluated Wahl factor
        self.wahl_factor = 0.0  # Wahl factor
        # List of positions for fatigue analysis
        self.positions = AngularPositionsTable()

    def set_geometry(self,
                     mean_diameter: float = None,
                     outer_diameter: float = None,
                     inner_diameter: float = None,
                     nr_coils: int = None,
                     pitch: float = None,
                     free_angle: float = None,
                     fixed_leg_radius: float = None,
                     mobile_leg_radius: float = None):
        """Configure the torsion spring with the given parameters"""
        self.set_diameter(mean_diameter=mean_diameter, outer_diameter=outer_diameter, inner_diameter=inner_diameter)
        self.set_nr_coils(nr_coils)
        self.set_pitch(pitch)
        self.set_free_angle(free_angle)
        self.set_fixed_leg_radius(fixed_leg_radius)
        self.set_mobile_leg_radius(mobile_leg_radius)
        return self.calculate_spring_properties()

    def calculate_spring_properties(self):
        """Calculate all torsion spring properties based on the given parameters"""
        self.set_tangency_angle(self.free_angle, self.fixed_leg_radius, self.mobile_leg_radius)
        self.set_spring_width(self.nr_coils, self.pitch, self.free_angle)
        self.set_wire_length(self.nr_coils, self.pitch, self.tangency_angle, self.fixed_leg_radius, self.mobile_leg_radius)
        self.calculate_spring_index()
        self.calculate_wahl_factor()
        self.calculate_length_legs(self.fixed_leg_radius, self.mobile_leg_radius)
        self.set_resisting_moment()
        self.calculate_spring_constant()
        return self.get_spring_properties()

    def set_material(self, material: str, wire_diameter: float):
        """Set the torsion spring's material"""
        super().set_material(material, wire_diameter)
        if not isinstance(material, Material):
            raise ValueError("""The material must be an instance of the
                             Material class""")
        if wire_diameter is None:
            raise ValueError("""You must provide the wire diameter to
                            set the material""")

    def set_diameter(
            self,
            mean_diameter: float = None,
            outer_diameter: float = None,
            inner_diameter: float = None,
            ):
        if sum(1 for var in [outer_diameter,
                             inner_diameter,
                             mean_diameter] if var is not None) != 1:
            raise ValueError("""You must provide exactly one of the three
                             diameters: outer, inner, mean""")

        if mean_diameter is not None:
            self.set_mean_diameter(mean_diameter)
        else:
            mean_diameter = self.calculate_mean_diameter(outer_diameter,
                                                         inner_diameter)
            self.set_mean_diameter(mean_diameter)

    def calculate_mean_diameter(self, outer_diameter=None, inner_diameter=None):
        """Calculate the spring's mean diameter"""
        provided = sum(1 for var in [outer_diameter, inner_diameter] if var is not None)

        if provided == 0:
            raise ValueError("You must provide at least one of outer_diameter or inner_diameter")

        if provided == 2:
            self.mean_diameter = (outer_diameter + inner_diameter) / 2
            self.wire_diameter = outer_diameter - self.mean_diameter.magnitude
            return self.mean_diameter

        if self.wire_diameter <= 0 or self.wire_diameter is None:
            raise ValueError("""The wire diameter must be a positive
                             non-null value to calculate the mean diameter""")
        if inner_diameter is not None:
            mean_diameter = inner_diameter + self.wire_diameter.magnitude
        else:
            mean_diameter = outer_diameter - self.wire_diameter.magnitude
        self.set_mean_diameter(mean_diameter)
        return self.mean_diameter

    def set_mean_diameter(self, mean_diameter):
        """Set the torsion spring's mean diameter"""
        if _mag(mean_diameter) <= 0:
            raise ValueError("The mean diameter must be a positive value")
        self.mean_diameter = mean_diameter
        self.inner_diameter = self.mean_diameter - self.wire_diameter
        self.outer_diameter = self.mean_diameter + self.wire_diameter
        return

    def set_nr_coils(self, nr_coils):
        """Set the torsion spring's number of coils"""
        if nr_coils <= 0:
            raise ValueError("The number of coils must be a positive value")
        self.nr_coils = nr_coils
        return self.nr_coils

    def set_pitch(self, pitch):
        """Set the torsion spring's pitch"""
        self.pitch = pitch
        if self.pitch < self.wire_diameter:
            raise ValueError("The pitch must be a greater than the wire diameter")
        return self.pitch

    def set_free_angle(self, free_angle):
        """Set the torsion spring's free angle"""
        if free_angle < 0:
            free_angle = 360 + free_angle
        self.free_angle = free_angle
        return self.free_angle

    def set_fixed_leg_radius(self, fixed_leg_radius):
        """Set the torsion spring's fixed leg radius"""
        self.fixed_leg_radius = fixed_leg_radius
        if self.fixed_leg_radius < self.mean_diameter / 2:
            raise ValueError("The fixed leg radius must be at least half the mean diameter")
        return self.fixed_leg_radius

    def set_mobile_leg_radius(self, mobile_leg_radius):
        """Set the torsion spring's mobile leg radius"""
        self.mobile_leg_radius = mobile_leg_radius
        if self.mobile_leg_radius < self.mean_diameter / 2:
            raise ValueError("The mobile leg radius must be at least half the mean diameter")
        return self.mobile_leg_radius

    def calculate_spring_index(self):
        """Calculate the torsion spring index"""
        self.spring_index = self.mean_diameter / self.wire_diameter
        return self.spring_index

    def calculate_wahl_factor(self):
        """Calculate the Wahl factor for torsion springs"""
        if self.spring_index is None:
            self.calculate_spring_index()

        self.wahl_factor = ((4*self.spring_index**2 - self.spring_index - 1)
                            /
                            (4 * self.spring_index * (self.spring_index - 1))
                            )
        if self.spring_index < 4:
            self.wahl_factor_category = 'Low'
            self.wahl_factor_eval = 1 + 0.5 / self.spring_index
        elif 4 <= self.spring_index < 8:
            self.wahl_factor_category = 'Medium'
            self.wahl_factor_eval = 1 + 0.75 / self.spring_index
        else:
            self.wahl_factor_category = 'High'
            self.wahl_factor_eval = 1 + 1.0 / self.spring_index

        return self.wahl_factor_eval

    def set_spring_width(self, nr_coils, pitch, free_angle):
        """Set the torsion spring's width"""
        if nr_coils is None or pitch is None or free_angle is None:
            raise ValueError("""You must provide the number of active
                             coils, pitch and free angle to set the spring
                             width""")
        self.free_angle = free_angle
        self.pitch = pitch
        self.spring_width = (self.nr_coils + self.free_angle / 2 / pi) * self.pitch
        self.nr_active_coils = (self.nr_coils +
                                self.free_angle / 2 / pi)

        return self.spring_width

    def set_wire_length(self,
                        nr_coils: int,
                        pitch: float,
                        tangency_angle: float,
                        fixed_leg_radius: float,
                        mobile_leg_radius: float):
        """Calculate the torsion spring's wire length"""
        parameters_provided = [nr_coils,
                               pitch,
                               tangency_angle,
                               fixed_leg_radius,
                               mobile_leg_radius]
        if sum(1 for var in parameters_provided if var is None) > 0:
            raise ValueError("""You must provide the number of active
                             coils, pitch, free angle, and leg radii to
                             calculate the wire length""")
        if _mag(self.mean_diameter) <= 0:
            raise ValueError("""The mean diameter must be a positive value
                             to calculate the wire length""")
        self.nr_coils = nr_coils
        self.pitch = pitch
        self.tangency_angle = tangency_angle
        self.fixed_leg_radius = fixed_leg_radius
        self.mobile_leg_radius = mobile_leg_radius
        single_turn_length = np.sqrt((pi * self.mean_diameter) ** 2 +
                                     (self.pitch / (2 * pi)) ** 2)
        self.body_wire_length = ((self.nr_coils + self.tangency_angle / 2 / pi) *
                                 single_turn_length)
        self.calculate_length_legs(self.fixed_leg_radius, self.mobile_leg_radius)
        self.wire_length_total = (self.body_wire_length +
                                  self.fixed_leg_length + self.mobile_leg_length)
        return self.wire_length_total

    def set_tangency_angle(self,
                           free_angle,
                           fixed_leg_radius,
                           mobile_leg_radius):
        """Calculate the torsion spring's tangency angle"""
        parameters_provided = [free_angle,
                               fixed_leg_radius,
                               mobile_leg_radius]
        if sum(1 for var in parameters_provided if var is None) > 0:
            raise ValueError("""You must provide the free angle and leg
                             radii to calculate the tangency angle""")
        self.free_angle = free_angle
        self.fixed_leg_radius = fixed_leg_radius
        self.mobile_leg_radius = mobile_leg_radius
        zero_equivalent_angle = (2 * pi -
                                 atan(self.fixed_leg_radius / self.mean_diameter) -
                                 atan(self.mobile_leg_radius / self.mean_diameter))
        if self.free_angle + zero_equivalent_angle > 2 * pi:
            self.tangency_angle = zero_equivalent_angle + self.free_angle - 2 * pi
        else:
            self.tangency_angle = 2 * pi - zero_equivalent_angle - self.free_angle
        self.nr_active_coils = self.nr_coils + self.tangency_angle / 2 / pi
        return self.tangency_angle

    def calculate_length_legs(self,
                              fixed_leg_radius=None,
                              mobile_leg_radius=None):
        """Calculate the torsion spring's leg lengths"""
        self.fixed_leg_radius = fixed_leg_radius
        self.mobile_leg_radius = mobile_leg_radius
        if self.fixed_leg_radius is None or self.fixed_leg_radius < self.mean_diameter / 2:
            self.fixed_leg_radius = self.mean_diameter / 2
        if self.mobile_leg_radius is None or self.mobile_leg_radius < self.mean_diameter / 2:
            self.mobile_leg_radius = self.mean_diameter / 2
        self.fixed_leg_length = np.sqrt(self.mean_diameter**2 / 4 + self.fixed_leg_radius**2)
        self.mobile_leg_length = np.sqrt(self.mean_diameter**2 / 4 + self.mobile_leg_radius**2)
        return self.fixed_leg_length, self.mobile_leg_length

    def set_number_cycles(self, number_cycles):
        """Set the number of cycles for the torsion spring's fatigue analysis"""
        self.number_cycles = number_cycles
        return self.number_cycles

    def set_shot_peening(self, shot_peening: bool):
        """Set whether the torsion spring has been treated with shot peening"""
        self.shot_peening = shot_peening
        return self.shot_peening

    def calculate_spring_constant(self):
        """Calculate the torsion spring constant"""
        if _mag(self.resisting_moment) <= 0:
            self.set_resisting_moment()
        self.spring_constant = (self.material.young_modulus *
                                self.resisting_moment /
                                self.wire_length_total)
        return self.spring_constant

    def calculate_torque(self, rotation_angle):
        """Calculate the torque applied to the torsion spring for a given
        rotation angle in degrees"""
        if _mag(rotation_angle) <= 0:
            raise ValueError("The rotation angle must be a positive value")
        self.rotation_angle = rotation_angle
        self.torque_position = self.spring_constant * self.rotation_angle
        return self.torque_position

    def set_resisting_moment(self):
        """Calculate the torsion spring's resisting moment for a given
        maximum torque"""
        self.resisting_moment = pi * np.pow(self.wire_diameter, 4) / 64
        return self.resisting_moment

    def calculate_stress(self, torque):
        """Calculate the maximum stress in the torsion spring for a given
        torque"""
        if _mag(torque) <= 0:
            raise ValueError("The torque must be a positive value to \
            calculate the stress")
        stress = ((32 * torque * self.wahl_factor) /
                  (pi * np.pow(self.wire_diameter, 3)))
        return stress

    def add_position(self, angle_travel=None, torque=None):
        """Add a position using a travel angle or torque for fatigue analysis"""
        if angle_travel is None and torque is None:
            raise ValueError("You must provide either an angle or a torque")
        if angle_travel is not None and torque is not None:
            raise ValueError("You must provide only an angle or a torque")
        if angle_travel is not None and angle_travel <= 0:
            raise ValueError("The angle must be a positive value")
        if torque is not None and torque <= 0:
            raise ValueError("The torque must be a positive value")
        if self.spring_constant <= 0:
            self.calculate_spring_constant()
        if angle_travel is not None:
            self.angle_travel = angle_travel
            self.torque_position = self.spring_constant * self.angle_travel
            self.stress_position = self.calculate_stress(self.torque_position)
        else:
            self.torque_position = torque
            self.angle_travel = self.torque_position / self.spring_constant
            self.stress_position = self.calculate_stress(self.angle_travel)
        self.angle_position = self.free_angle - self.angle_travel
        new_tangency_angle = self.tangency_angle + self.angle_travel
        new_mean_diameter = (4 * (self.body_wire_length ** 2 - self.pitch ** 2) /
                             (2 * pi * self.nr_coils + new_tangency_angle) ** 2) ** 0.5
        outer_diameter = new_mean_diameter + self.wire_diameter
        inner_diameter = new_mean_diameter - self.wire_diameter
        if not hasattr(self, 'positions'):
            self.positions = AngularPositionsTable()
        self.positions.add_load_position(self.angle_position,
                                         self.angle_travel,
                                         self.torque_position,
                                         self.stress_position, outer_diameter, inner_diameter)
        return self.positions

    def add_load_position(self, length):
        """Add a load position using an angular travel value (in degrees).

        Mirrors ``CompressionSpring.add_load_position`` so both spring types
        expose the same API for the PDF report.
        """
        return self.add_position(angle_travel=length)

    def get_stress_max(self):
        """Return the maximum stress among the recorded load positions."""
        if not self.positions.positions:
            raise ValueError("No load positions available")
        return max(self.positions.positions, key=lambda x: x.stress).stress

    def get_stress_min(self):
        """Return the minimum stress among the recorded load positions."""
        if not self.positions.positions:
            raise ValueError("No load positions available")
        return min(self.positions.positions, key=lambda x: x.stress).stress

    def get_load_max(self):
        """Return the maximum torque among the recorded load positions."""
        if not self.positions.positions:
            raise ValueError("No load positions available")
        return max(self.positions.positions, key=lambda x: x.load).load

    def get_load_min(self):
        """Return the minimum torque among the recorded load positions."""
        if not self.positions.positions:
            raise ValueError("No load positions available")
        return min(self.positions.positions, key=lambda x: x.load).load

    def create_goodman_diagram(self, show=False):
        """Generate the Goodman diagram for the torsion spring's wire, which is
        loaded in bending. Returns a dictionary containing the base64 image,
        analysis, and calculated stresses, or an 'error'/'traceback' pair."""
        try:
            if not self.positions.positions:
                raise ValueError("No load positions available for Goodman analysis")

            stress_max = self.get_stress_max()
            stress_min = self.get_stress_min()

            goodman_data = GoodmanData(
                material=self.material,
                wire_diameter=self.wire_diameter,
                load_type='flexion',
                number_cycles=int(self.number_cycles)
            )

            analyzer = GoodmanAnalyzer(goodman_data, shot_peening=self.shot_peening)
            if show:
                analyzer.plot_diagram(sigma_max=stress_max, sigma_min=stress_min)
                return

            image_b64 = analyzer.get_diagram_image(sigma_max=stress_max, sigma_min=stress_min)
            analysis = analyzer.get_analysis_summary(stress_max, stress_min)

            return {
                'image': image_b64,
                'analysis': analysis,
                'stresses': {
                    'stress_max': round(stress_max, 2),
                    'stress_min': round(stress_min, 2),
                    'load_max': round(self.get_load_max(), 2),
                    'load_min': round(self.get_load_min(), 2)
                }
            }
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error creating Goodman diagram in TorsionSpring: {e}\n{tb}")
            return {'error': str(e), 'traceback': tb}

    def get_3d_plot(self, num_points: int = 200, show: bool = False, isometric: bool = True) -> str:
        """Render the spring's helical body geometry in 3D (constant mean
        diameter and pitch). Returns a base64-encoded PNG, matching the
        convention used by the other graph methods."""
        theta_max = 2 * np.pi * self.nr_coils
        thetas = np.linspace(0, theta_max, num_points)
        radius = float(self.mean_diameter.to('mm').magnitude) / 2.0
        pitch_mm = float(self.pitch.to('mm').magnitude)

        xs = radius * np.cos(thetas)
        ys = radius * np.sin(thetas)
        zs = pitch_mm * thetas / (2 * np.pi)

        with interactive_backend(show):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot(xs, ys, zs)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title('Spring 3D Model')
            ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
            if isometric:
                ax.set_proj_type('ortho')
                ax.view_init(elev=35.264, azim=45)
            if show:
                plt.show()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close(fig)

        return plot_data

    def clean_positions(self):
        """Clear the positions list for fatigue analysis"""
        self.positions.clear_table()

        return self.positions.positions

    def get_positions(self):
        """Get the positions list for fatigue analysis"""
        return self.positions.positions

    def get_data_positions(self):
        """Return the angular positions table for position curves."""
        return self.positions.positions

    def get_data_travels(self):
        """Return the angular positions table for travel curves."""
        return self.positions.positions

    def get_forces_vs_position_graph(self, show=False):
        """Generate the torque vs angular position curve."""
        def _to_deg_float(value):
            return float(value.to('degree').magnitude) if isinstance(value, Quantity) else float(value)

        def _to_torque_float(value):
            return float(value.to('N*mm').magnitude) if isinstance(value, Quantity) else float(value)

        positions_table = self.positions.positions
        if not positions_table:
            raise ValueError('No positions to plot')

        positions = [_to_deg_float(pc.position) for pc in positions_table]
        torques = [_to_torque_float(pc.load) for pc in positions_table]

        with interactive_backend(show):
            plot = plt.figure()
            plt.plot(positions, torques, marker='o')
            plt.title('Torque vs Angular Position Curve')
            plt.xlabel('Position (degrees)')
            plt.ylabel('Torque (N*mm)')
            plt.grid(True)
            if show:
                plt.show()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close(plot)
        return plot_data

    def get_forces_vs_travel_graph(self, show=False):
        """Generate the torque vs angular travel curve."""
        def _to_deg_float(value):
            return float(value.to('degree').magnitude) if isinstance(value, Quantity) else float(value)

        def _to_torque_float(value):
            return float(value.to('N*mm').magnitude) if isinstance(value, Quantity) else float(value)

        positions_table = self.positions.positions
        if not positions_table:
            raise ValueError('No positions to plot')

        travels = [_to_deg_float(pc.travel) for pc in positions_table]
        torques = [_to_torque_float(pc.load) for pc in positions_table]

        with interactive_backend(show):
            plot = plt.figure()
            plt.plot(travels, torques, marker='o')
            plt.title('Torque vs Angular Travel Curve')
            plt.xlabel('Travel (degrees)')
            plt.ylabel('Torque (N*mm)')
            plt.grid(True)
            if show:
                plt.show()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close(plot)
        return plot_data

    def get_diameter_vs_position_graph(self, show=False):
        """Plot outer diameter vs angular position and the spring cross-section."""
        def _to_deg_float(value):
            return float(value.to('degree').magnitude) if isinstance(value, Quantity) else float(value)

        def _to_mm_float(value):
            return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)

        positions_table = self.positions.positions
        if not positions_table:
            raise ValueError('No positions to plot')

        positions = [_to_deg_float(pc.position) for pc in positions_table]
        diameters = [_to_mm_float(pc.outer_diameter) for pc in positions_table]

        outer_diameter = self.mean_diameter + self.wire_diameter
        inner_diameter = max(self.mean_diameter - self.wire_diameter, 0 * ureg.mm)
        outer_diameter_mm = _to_mm_float(outer_diameter)
        inner_diameter_mm = _to_mm_float(inner_diameter)

        with interactive_backend(show):
            fig, (ax1, ax2) = plt.subplots(
                1,
                2,
                figsize=(10, 4),
                gridspec_kw={'width_ratios': [2, 1]}
            )

            ax1.plot(positions, diameters, marker='o', color='orange')
            ax1.set_title('Outer Diameter vs Angular Position')
            ax1.set_xlabel('Position (degrees)')
            ax1.set_ylabel('Outer Diameter (mm)')
            ax1.grid(True)

            ax2.set_aspect('equal')
            ax2.axis('off')

            outer_radius = outer_diameter_mm / 2.0
            inner_radius = inner_diameter_mm / 2.0
            max_radius = max(outer_radius, inner_radius, 1.0)
            padding = max_radius * 0.25

            ax2.add_patch(Circle((0, 0), outer_radius, fill=False, lw=2, color='tab:green'))
            if inner_radius > 0:
                ax2.add_patch(Circle((0, 0), inner_radius, fill=False, lw=2, color='tab:blue'))

            ax2.plot([-outer_radius, outer_radius], [0, 0], color='tab:green', lw=1)
            ax2.text(0, -padding, f"Dext = {outer_diameter_mm:.2f} mm", ha='center', va='top', fontsize=8)

            if inner_radius > 0:
                ax2.plot([0, 0], [-inner_radius, inner_radius], color='tab:blue', lw=1)
                ax2.text(0, padding, f"Dint = {inner_diameter_mm:.2f} mm", ha='center', va='bottom', fontsize=8)

            ax2.set_xlim(-max_radius - padding, max_radius + padding)
            ax2.set_ylim(-max_radius - padding, max_radius + padding)
            if show:
                plt.show()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close(fig)
        return plot_data

    def get_spring_properties(self) -> dict:
        """Get a dictionary with all the torsion spring properties"""
        return self.get_spring_data()

    def get_spring_data(self) -> dict:
        """Get a dictionary with all the torsion spring properties"""
        return {
            'material': self.material.material_name,
            'young_modulus': self.material.young_modulus,
            'wire_diameter': self.wire_diameter,
            'mean_diameter': self.mean_diameter,
            'inner_diameter': self.inner_diameter,
            'outer_diameter': self.outer_diameter,
            'free_angle': self.free_angle,  # Convert to degrees for output
            'tangency_angle': self.tangency_angle,  # Convert to degrees for output
            'pitch': self.pitch,
            'shot_peening': self.shot_peening,
            'number_cycles': self.number_cycles,
            'nr_active_coils': self.nr_active_coils,
            'nr_coils': self.nr_coils,
            'spring_width': self.spring_width,
            'fixed_leg_radius': self.fixed_leg_radius,
            'fixed_leg_length': self.fixed_leg_length,
            'mobile_leg_radius': self.mobile_leg_radius,
            'mobile_leg_length': self.mobile_leg_length,
            'spring_index': self.spring_index,
            'wahl_factor': self.wahl_factor,
            'wahl_factor_category': self.wahl_factor_category,
            'wahl_factor_eval': self.wahl_factor_eval,
            'wire_length_total': self.wire_length_total,
            'body_wire_length': self.body_wire_length,
            'spring_constant': self.spring_constant,
            'resisting_moment': self.resisting_moment
        }
