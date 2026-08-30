"""Class for calculating a standard compression spring."""
from math import pi
from pydantic import field_validator, ConfigDict
from . import constants as const
from .constants import COMPRESSION_SPRING_END_TYPES
from .goodman import Goodman
import traceback
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import io
import base64
import numpy as np
from .goodman import GoodmanData, GoodmanAnalyzer
from .lineal import LinealSpring
from pint import Quantity
from ..pymodels.units import ureg
from ..pymodels.positions import LinearPositionsTable
from typing import Optional
import matplotlib
from .plotting import interactive_backend
matplotlib.use('Agg')


class CompressionSpring(LinealSpring):
    # Additional CompressionSpring fields
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    type_of_end: str = const.COMPRESSION_SPRING_END_TYPES[const.CLOSED_GROUND]  # ground by default
    # set cold formed by default
    type_conforming: str = 'cold_formed'
    # the wire length should be in mm
    wire_length: Optional[Quantity] = 0.0 * ureg.mm
    nr_coils: Optional[float] = None
    solid_length: Quantity = 0.0 * ureg.mm  # Length at maximum load in mm
    positions: LinearPositionsTable = LinearPositionsTable()

    @field_validator('wire_length', "solid_length", mode='before')
    @classmethod
    def validate_quantities(cls, value):
        if isinstance(value, str):
            try:
                quantity = ureg(value)
                return quantity.to('mm')
            except Exception as e:
                raise ValueError(f"Error converting '{value}' to Quantity: {e}")
        elif isinstance(value, (int, float)):
            return Quantity(value, 'mm')
        elif isinstance(value, Quantity):
            return value.to('mm')
        else:
            raise ValueError(f"Invalid value for Quantity: {value}")

    def __init__(self, material, wire_diameter: float, **data):
        """Initialize the spring variables to 0."""
        # Initialize with default values
        super().__init__(material, wire_diameter, **data)

    def set_material(self, material, wire_diameter):
        return super().set_material(material, wire_diameter)

    def set_geometry(self,
                     mean_diameter: float = None,
                     outer_diameter: float = None,
                     inner_diameter: float = None,
                     nr_coils: float = None,
                     pitch: float = None,
                     free_length: float = None,
                     type_of_end: Optional[str] = None):
        """Set the spring's full geometry in one call.
        Provide the general geometry parameters and characteristics of the spring
        Computes and stores every derived spring property and calls calculate_spring_properties().
        Parameters:
        one of the following must be provided: mean_diameter, outer_diameter, inner_diameter
            mean_diameter: mean diameter of the spring (mm)
            outer_diameter: outer diameter of the spring (mm)
            inner_diameter: inner diameter of the spring (mm)
        two of the following must be provided: nr_coils, pitch, free_length
            nr_coils: number of coils in the spring
            pitch: pitch of the spring (mm)
            free_length: free length of the spring (mm)
        Optional parameters:
            type_of_end: type of end of the spring ('open_ground','closed_ground','open_unground','closed_unground')

        Returns:
        A dictionary with all the spring properties, including the derived ones.
        """
        diameters_provided = sum(1 for var in [mean_diameter, outer_diameter, inner_diameter] if var is not None)
        if diameters_provided != 1:
            raise ValueError("You must provide exactly one of the following variables: mean_diameter, outer_diameter, inner_diameter")

        length_params_provided = sum(1 for var in [nr_coils, pitch, free_length] if var is not None)
        if length_params_provided != 2:
            raise ValueError("You must provide exactly two of the following variables: nr_coils, pitch, free_length")

        if type_of_end is not None:
            self.type_of_end = type_of_end

        self.set_diameter(mean_diameter=mean_diameter,
                          outer_diameter=outer_diameter,
                          inner_diameter=inner_diameter)
        return self.calculate_spring_properties(nr_coils=nr_coils,
                                                pitch=pitch,
                                                free_length=free_length)

    def calculate_spring_properties(self,
                                    nr_coils: float = None,
                                    pitch: float = None,
                                    free_length: float = None):
        """Part three: Calculate all spring properties."""
        parameters_provided = sum(1 for var in [nr_coils, pitch, free_length] if var is not None)
        if parameters_provided != 2:
            raise ValueError("You must provide exactly two of the following variables: nr_coils, pitch, free_length")

        end_factors = const.COMPRESSION_END_FACTORS[self.type_of_end]
        inactive_coils = end_factors['inactive_coils']
        end_length = end_factors['length_factor'] * self.wire_diameter

        if nr_coils is not None and pitch is not None:
            self.nr_coils = nr_coils
            self.pitch = pitch
            self.free_length = (self.nr_coils - inactive_coils) * self.pitch + end_length
        elif nr_coils is not None and free_length is not None:
            self.nr_coils = nr_coils
            self.free_length = free_length
            self.pitch = (self.free_length - end_length) / (self.nr_coils - inactive_coils)
        elif pitch is not None and free_length is not None:
            self.pitch = pitch
            self.free_length = free_length
            self.nr_coils = (self.free_length - end_length) / self.pitch + inactive_coils
        else:
            raise ValueError("You must provide exactly two of the following variables: nr_coils, pitch, free_length")

        if self.pitch <= self.wire_diameter:
            raise ValueError("The pitch must be greater than the wire diameter to avoid interference between coils.")
        self.calculate_active_coils(nr_coils=self.nr_coils)
        self.calculate_wahl_factor()
        self.calculate_spring_constant()
        self.calculate_solid_length()
        self.calculate_wire_length()
        self.calculate_load_at_position(self.solid_length)
        self.calculate_outer_diameter_at_position(self.solid_length)
        properties = self.get_spring_data()
        return properties

    def calculate_active_coils(self, nr_coils):
        """Calculate the number of active spring coils based on the end type."""
        active_coils_offset = const.COMPRESSION_END_FACTORS[self.type_of_end]['active_coils_offset']
        self.nr_active_coils = nr_coils - active_coils_offset
        return self.nr_active_coils

    def calculate_wire_length(self):
        """Calculate the spring wire length."""
        self.wire_length = self.nr_coils * np.sqrt((pi * self.mean_diameter)**2 + self.pitch**2)
        return self.wire_length

    def calculate_pitch(self):
        """Calculate the spring pitch."""
        return self.free_length / self.nr_coils

    def calculate_solid_length(self):
        """Calculate the spring solid length."""
        self.solid_length = self.nr_coils * self.wire_diameter
        return self.solid_length

    def add_load_position(self, length: Quantity):
        """Add a load position to the positions table."""
        if not isinstance(length, Quantity):
            length = length * ureg.mm
        try:
            if length < self.solid_length:
                raise ValueError("The position cannot be smaller than the spring's solid length")
            load = self.calculate_load_at_position(length)
            stress = self.calculate_stress_at_position(load)
            outer_diameter = self.calculate_outer_diameter_at_position(length)
        except ValueError as e:
            raise ValueError(f"Error adding load position at length {length}: {e}")
        self.positions.add_load_position(
            position=self.position_length,
            travel=self.free_length - self.position_length,
            load=load,
            stress=stress,
            outer_diameter=outer_diameter,
            inner_diameter=max(outer_diameter - 2 * self.wire_diameter, 0 * ureg.mm)
        )

    def calculate_outer_diameter_at_position(self, length):
        """Calculate the spring outer diameter."""
        self.position_length = length
        outer_diameter = (self.mean_diameter + self.wire_diameter +
                          self.mean_diameter * self.material.poisson_coef *
                          (self.free_length - self.position_length) / self.free_length)
        return outer_diameter

    def empty_tables(self):
        """Clear the positions, loads, stresses, and outer diameters tables."""
        self.positions.clear_table()

    def get_spring_data(self):
        """Return a dictionary with the main spring data."""
        return {
            "material": self.material.material_name,
            "young_modulus": self.material.young_modulus,
            "shear_modulus": self.material.shear_modulus,
            "elastic_limit_factor": self.material.elastic_limit_factor,
            "poisson_coef": self.material.poisson_coef,
            "RMa_file": self.material.RMa_file,
            "wire_diameter": self.wire_diameter,
            "mean_diameter": self.mean_diameter,
            "free_length": self.free_length,
            "nr_coils": self.nr_coils,
            "nr_active_coils": self.nr_active_coils,
            "spring_constant": self.spring_constant,
            "spring_index": self.spring_index,
            "wahl_factor": self.wahl_factor,
            "wahl_factor_category": self.wahl_factor_category,
            "solid_length": self.solid_length,
            "wire_length": self.wire_length,
            "number_cycles": self.number_cycles,
            "shot_peening": self.shot_peening,
            "coating": self.coating
        }

    def get_data_positions(self):
        """Return the positions, loads, stresses, and outer diameters table."""
        return self.positions.positions

    def get_data_travels(self):
        """Return the positions table for the travel curve."""
        return self.positions.positions

    def get_forces_vs_position_graph(self, show=False):
        def _to_mm_float(value):
            return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)

        def _to_n_float(value):
            return float(value.to('N').magnitude) if isinstance(value, Quantity) else float(value)

        positions_table = self.positions.positions
        positions = [_to_mm_float(pc.position) for pc in positions_table]
        loads = [_to_n_float(pc.load) for pc in positions_table]
        with interactive_backend(show):
            plt.figure()
            plt.plot(positions, loads, marker='o')
            plt.title('Load vs Position Curve')
            plt.xlabel('Position (mm)')
            plt.ylabel('Load (N)')
            plt.grid(True)
            if show:
                plt.show()

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close()

        return plot_data

    def get_forces_vs_travel_graph(self, show=False):
        def _to_mm_float(value):
            return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)

        def _to_n_float(value):
            return float(value.to('N').magnitude) if isinstance(value, Quantity) else float(value)

        positions_table = self.positions.positions
        travels = [_to_mm_float(pc.travel) for pc in positions_table]
        loads = [_to_n_float(pc.load) for pc in positions_table]
        with interactive_backend(show):
            plt.figure()
            plt.plot(travels, loads, marker='o')
            plt.title('Load vs Travel Curve')
            plt.xlabel('Travel (mm)')
            plt.ylabel('Load (N)')
            plt.grid(True)
            # Save the plot to a BytesIO object
            if show:
                plt.show()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close()

        return plot_data

    def get_diameter_graph(self, show=False):
        def _to_mm_float(value):
            return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)

        positions_table = self.positions.positions
        positions = [_to_mm_float(pc.position) for pc in positions_table]
        diameters = [_to_mm_float(pc.outer_diameter) for pc in positions_table]
        with interactive_backend(show):
            plt.figure()
            plt.plot(positions, diameters, marker='o', color='orange')
            plt.title('Outer Diameter vs Position')
            plt.xlabel('Position (mm)')
            plt.ylabel('Outer Diameter (mm)')
            plt.grid(True)
            if show:
                plt.show()

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close()

        return plot_data

    def get_diameter_vs_position_graph(self, show=False):
        """Plot diameter versus position and add a circle diagram."""
        def _to_mm_float(value):
            if isinstance(value, Quantity):
                return float(value.to('mm').magnitude)
            return float(value)

        positions_table = self.positions.positions
        positions = [_to_mm_float(pc.position) for pc in positions_table]
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
                gridspec_kw={"width_ratios": [2, 1]}
            )

            ax1.plot(positions, diameters, marker='o', color='orange')
            ax1.set_title('Outer Diameter vs Position')
            ax1.set_xlabel('Position (mm)')
            ax1.set_ylabel('Outer Diameter (mm)')
            ax1.grid(True)

            ax2.set_aspect('equal')
            ax2.axis('off')

            outer_radius = outer_diameter_mm / 2.0
            inner_radius = inner_diameter_mm / 2.0
            max_radius = max(outer_radius, inner_radius, 1.0)
            padding = max_radius * 0.25

            ax2.add_patch(Circle((0, 0),
                                 outer_radius,
                                 fill=False, lw=2,
                                 color='tab:green'))
            if inner_radius > 0:
                ax2.add_patch(Circle((0, 0),
                                     inner_radius,
                                     fill=False,
                                     lw=2,
                                     color='tab:blue'))

            ax2.plot([-outer_radius, outer_radius],
                     [0, 0],
                     color='tab:green',
                     lw=1)
            ax2.text(0,
                     -padding,
                     f"Dext = {outer_diameter_mm:.2f} mm",
                     ha='center',
                     va='top',
                     fontsize=8)

            if inner_radius > 0:
                ax2.plot([0, 0],
                         [-inner_radius, inner_radius],
                         color='tab:blue',
                         lw=1)
                ax2.text(0,
                         padding,
                         f"Dint = {inner_diameter_mm:.2f} mm",
                         ha='center',
                         va='bottom',
                         fontsize=8)

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

    def get_3d_plot(self, num_points: int = 200, show: bool = False, isometric: bool = True) -> str:
        """Render the spring's helical geometry in 3D (constant mean diameter and pitch).

        Returns a base64-encoded PNG (same convention as the other graph methods).
        Defaults to an orthographic isometric view, matching how spring drawings
        are conventionally presented.
        """
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

    def create_goodman_diagram(self, show=False):
        """Generate the Goodman diagram and return a dictionary containing the
        base64 image, analysis, and calculated stresses. If it fails, return a
        dictionary with the 'error' and 'traceback' keys."""
        try:
            if not self.positions.positions:
                raise ValueError("No load positions available for Goodman analysis")

            # Get stresses from the positions table.
            stress_max = self.get_stress_max()
            stress_min = self.get_stress_min()

            # Prepare data for Goodman.
            goodman_data = GoodmanData(
                material=self.material,
                wire_diameter=self.wire_diameter,
                load_type='torsion',
                number_cycles=int(self.number_cycles)
            )

            analyzer = GoodmanAnalyzer(goodman_data,
                                       shot_peening=self.shot_peening)
            if show:
                analyzer.plot_diagram(sigma_max=stress_max, sigma_min=stress_min)
                return

            image_b64 = analyzer.get_diagram_image(sigma_max=stress_max,
                                                   sigma_min=stress_min)

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
            print(f"Error creating Goodman diagram in LinealSpring: {e}\n{tb}")
            return {'error': str(e), 'traceback': tb}

    def get_goodman_graph(self, goodman: Goodman, show=True):
        """Create the Goodman diagram for the spring."""
        # Placeholder implementation
        # The logic for creating the Goodman diagram would be implemented here.
        stress_min = min(self.positions.positions,
                         key=lambda x: x.stress).stress
        stress_max = max(self.positions.positions,
                         key=lambda x: x.stress).stress
        if show:
            goodman.plot_goodman_graph(stress_max, stress_min)
        else:
            goodman_fig = goodman.get_goodman_graph(stress_max, stress_min)
        return goodman_fig

    def get_goodman_analysis_summary(self, goodman: Goodman):
        """Return a complete summary of the Goodman analysis."""
        stress_min = self.get_stress_min()
        stress_max = self.get_stress_max()
        return goodman.get_analysis_summary(stress_max, stress_min)

    def get_stress_max(self):
        """Return the maximum spring stress."""
        if not self.positions.positions:
            raise ValueError("No load positions available")
        return max(self.positions.positions, key=lambda x: x.stress).stress

    def get_stress_min(self):
        """Return the minimum spring stress."""
        if not self.positions.positions:
            raise ValueError("No load positions available")
        return min(self.positions.positions, key=lambda x: x.stress).stress

    def get_load_max(self):
        """Return the maximum spring load."""
        if not self.positions.positions:
            raise ValueError("No load positions available")
        return max(self.positions.positions, key=lambda x: x.load).load

    def get_load_min(self):
        """Return the minimum spring load."""
        if not self.positions.positions:
            raise ValueError("No load positions available")
        return min(self.positions.positions, key=lambda x: x.load).load
