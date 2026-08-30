from math import pi
from typing import Optional
import io
import base64
import traceback
import numpy as np
from pint import Quantity
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from ..pymodels.units import ureg
from .constants import WAHL_FACTOR_CONSTANTS
from ..pymodels.positions import LinearPositionsTable
from .generic_lineal import VariableLinealSpring
from .goodman import GoodmanData, GoodmanAnalyzer
from .plotting import interactive_backend
matplotlib.use('Agg')

COMPRESSION = 1


class CompressionSpringGeneral(VariableLinealSpring):
    solid_length: Quantity = 0.0 * ureg.mm
    positions: LinearPositionsTable = LinearPositionsTable()
    position_length: Optional[Quantity] = 0.0 * ureg.mm
    position_load: Optional[Quantity] = 0.0 * ureg.N
    position_stress: Optional[Quantity] = 0.0 * ureg.megapascal
    wahl_factor: float = 0.0
    wahl_factor_category: Optional[str] = None
    number_cycles: int = 1_000_000

    def calculate_spring_properties(self, num_points: int = 500) -> dict:
        """
        Calculate all derived properties, including the compression-specific
        ones (solid length, representative Wahl factor) that CompressionSpring
        also reports.
        """
        super().calculate_spring_properties(num_points=num_points)
        self.calculate_wahl_factor_at_position(self.free_length / 2)
        self.calculate_solid_length()
        return self.get_spring_data()

    def get_spring_data(self) -> dict:
        """Return a dictionary with the main spring data, matching CompressionSpring's fields."""
        return {
            "material": self.material.material_name,
            "young_modulus": self.material.young_modulus,
            "shear_modulus": self.material.shear_modulus,
            "elastic_limit_factor": self.material.elastic_limit_factor,
            "poisson_coef": self.material.poisson_coef,
            "RMa_file": self.material.RMa_file,
            "wire_diameter": self.wire_diameter,
            "mean_diameter": self.f_mean_diameter(self.free_length / 2),
            "free_length": self.free_length,
            "nr_coils": self.nr_coils,
            "spring_constant": self.spring_constant,
            "spring_index": self.spring_index,
            "wahl_factor": self.wahl_factor,
            "wahl_factor_category": self.wahl_factor_category,
            "solid_length": self.solid_length,
            "wire_length": self.wire_length,
            "number_cycles": self.number_cycles,
            "shot_peening": self.shot_peening,
            "coating": self.coating,
        }

    def calculate_solid_length(self) -> Quantity:
        """
        In a variable-geometry spring (such as a conical one), the coils can
        nest inside one another (telescoping).
        """
        # If the spring nests flat (very different diameters): solid_length = wire_diameter
        # If it's a standard compression spring where coils collide: solid_length = nr_coils * wire_diameter
        # Add a basic check for "nesting" or "telescoping"
        # H_val = self.free_length.to('mm').magnitude
        D_start = self.f_mean_diameter(0 * ureg.mm).to('mm').magnitude
        D_end = self.f_mean_diameter(self.free_length).to('mm').magnitude
        d_wire = self.wire_diameter.to('mm').magnitude

        # If the radius difference is greater than the wire diameter per coil, it "nests"
        if abs(D_start - D_end) >= (self.nr_coils * d_wire):
            # Perfect telescoping limit case (every coil fits inside another)
            self.solid_length = d_wire * ureg.mm
        else:
            # Conventional case: coils stack on top of one another
            self.solid_length = self.nr_coils * self.wire_diameter

        return self.solid_length

    def calculate_load_at_position(self, length: Quantity, spring_class: int = COMPRESSION) -> Quantity:
        """Calculate the load at a given position from the contact-aware progressive
        compression simulation, rather than assuming a single constant stiffness
        across the whole travel. This captures the stiffening caused by coil-to-coil
        collisions and by coils bottoming out against the floor or the moving plate
        as the position approaches solid height.
        """
        self.position_length = length
        travel = self.free_length - self.position_length

        deflection_history, force_history, _ = self.simulate_progressive_compression(
            max_deflection=self.free_length, steps=500,
        )
        force_n = np.interp(
            travel.to('mm').magnitude,
            deflection_history.to('mm').magnitude,
            force_history.to('N').magnitude,
        )
        self.position_load = spring_class * force_n * ureg.N
        self.calculate_stress_at_position(self.position_load)
        return self.position_load

    def calculate_wahl_factor_at_position(self, length: Quantity):
        """Calculate the Wahl factor using the local spring index at a given position."""
        spring_index = self.calculate_spring_index_local(length)
        self.wahl_factor = (4 * spring_index - 1) / (4 * spring_index - 4) + 0.615 / spring_index
        if spring_index < WAHL_FACTOR_CONSTANTS['red'][1]:
            self.wahl_factor_category = 'red'
        elif spring_index < WAHL_FACTOR_CONSTANTS['orange'][1]:
            self.wahl_factor_category = 'orange'
        else:
            self.wahl_factor_category = 'green'
        return self.wahl_factor, self.wahl_factor_category

    def calculate_stress_at_position(self, load: Quantity) -> Quantity:
        """Calculate the stress in the spring wire at a given predefined position."""
        self.calculate_wahl_factor_at_position(self.position_length)
        local_diameter = self.f_mean_diameter(self.position_length)
        self.position_stress = (8 * local_diameter * load) / (pi * self.wire_diameter**3) * self.wahl_factor
        return self.position_stress

    def calculate_outer_diameter_at_position(self, length: Quantity) -> Quantity:
        """Calculate the spring outer diameter at a given position."""
        self.position_length = length
        local_diameter = self.f_mean_diameter(length)
        outer_diameter = local_diameter + self.wire_diameter + \
            local_diameter * self.material.poisson_coef * \
            (self.free_length - self.position_length) / self.free_length
        return outer_diameter

    def add_load_position(self, length: Quantity):
        """Add a load position to the positions table."""
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

    def empty_tables(self):
        """Clear the positions table."""
        self.positions.clear_table()

    def get_data_positions(self):
        """Return the positions, loads, stresses, and outer diameters table."""
        return self.positions.positions

    def get_data_travels(self):
        """Return the positions table for the travel curve."""
        return self.positions.positions

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

    def create_goodman_diagram(self, show=False):
        """Generate the Goodman diagram and return a dictionary containing the
        base64 image, analysis, and calculated stresses. If it fails, return a
        dictionary with the 'error' and 'traceback' keys."""
        try:
            if not self.positions.positions:
                raise ValueError("No load positions available for Goodman analysis")

            stress_max = self.get_stress_max()
            stress_min = self.get_stress_min()

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
            print(f"Error creating Goodman diagram in CompressionSpringGeneral: {e}\n{tb}")
            return {'error': str(e), 'traceback': tb}

    # Resolution used for the load curve underlying both graphs below. Matches
    # calculate_load_at_position's call so the two share the same simulate_
    # progressive_compression() cache entry instead of recomputing it.
    _LOAD_CURVE_STEPS = 500

    def _get_load_curve(self):
        """Dense (travel_mm, force_n) arrays spanning the full compressible
        travel, from the contact-aware progressive simulation (coil-to-coil,
        floor, and top-plate contact), not just the handful of manually added
        load positions."""
        deflection_history, force_history, _ = self.simulate_progressive_compression(
            max_deflection=self.free_length, steps=self._LOAD_CURVE_STEPS,
        )
        return deflection_history.to('mm').magnitude, force_history.to('N').magnitude

    def get_forces_vs_position_graph(self, show=False):
        def _to_mm_float(value):
            return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)

        def _to_n_float(value):
            return float(value.to('N').magnitude) if isinstance(value, Quantity) else float(value)

        travel_mm, force_n = self._get_load_curve()
        position_mm = self.free_length.to('mm').magnitude - travel_mm

        positions_table = self.positions.positions
        marked_positions = [_to_mm_float(pc.position) for pc in positions_table]
        marked_loads = [_to_n_float(pc.load) for pc in positions_table]
        with interactive_backend(show):
            plt.figure()
            plt.plot(position_mm, force_n)
            if marked_positions:
                plt.plot(marked_positions, marked_loads, 'o', color='tab:red', label='Load positions')
                plt.legend()
            plt.title('Load vs Position Curve')
            plt.xlabel('Position (mm)')
            plt.ylabel('Load (N)')
            plt.grid(True)
            if show:
                plt.show()

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

        travel_mm, force_n = self._get_load_curve()

        positions_table = self.positions.positions
        marked_travels = [_to_mm_float(pc.travel) for pc in positions_table]
        marked_loads = [_to_n_float(pc.load) for pc in positions_table]
        with interactive_backend(show):
            plt.figure()
            plt.plot(travel_mm, force_n)
            if marked_travels:
                plt.plot(marked_travels, marked_loads, 'o', color='tab:red', label='Load positions')
                plt.legend()
            plt.title('Load vs Travel Curve')
            plt.xlabel('Travel (mm)')
            plt.ylabel('Load (N)')
            plt.grid(True)
            if show:
                plt.show()

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

        mid_diameter = self.f_mean_diameter(self.free_length / 2)
        outer_diameter = mid_diameter + self.wire_diameter
        inner_diameter = max(mid_diameter - self.wire_diameter, 0 * ureg.mm)
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

    def get_progressive_compression_graph(self, deflection: Quantity, force: Quantity, show: bool = False) -> str:
        """Plot the deflection/force curve from simulate_progressive_compression.

        Returns a base64-encoded PNG, ready to embed in HTML or a PDF report
        (same convention as CompressionSpring's graph methods). The module is
        pinned to the non-interactive "Agg" backend for headless web/PDF use,
        so show=True temporarily switches to a real interactive backend for
        this call (blocking until the window is closed) and restores "Agg"
        afterwards.
        """
        deflection_mm = deflection.to('mm').magnitude
        force_n = force.to('N').magnitude

        with interactive_backend(show):
            plt.figure()
            plt.plot(deflection_mm, force_n, marker='o')
            plt.title('Progressive Compression Simulation')
            plt.xlabel('Deflection (mm)')
            plt.ylabel('Force (N)')
            plt.grid(True)
            if show:
                plt.show()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close()

        return plot_data

    def get_3d_plot(self, num_points: int = 500, show: bool = False, isometric: bool = True) -> str:
        """Render the spring's helical centerline in 3D.

        Reuses the same (theta, z, diameter) development as calculate_wire_length,
        so variable diameter/pitch geometries show up as a non-uniform helix.
        Returns a base64-encoded PNG (same convention as the other graph methods).
        Defaults to an orthographic isometric view, matching how spring drawings
        are conventionally presented.
        """
        thetas, zs = self.get_h_theta_development(num_points)
        Ds = np.array([self.f_mean_diameter(z * ureg.mm).to('mm').magnitude for z in zs])
        radii = Ds / 2.0

        xs = radii * np.cos(thetas)
        ys = radii * np.sin(thetas)

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
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close()

        return plot_data
