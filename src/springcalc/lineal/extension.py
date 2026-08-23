from numpy import sqrt, pi
from typing import List, Optional
import traceback
import io
import base64
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from .constants import WAHL_FACTOR_CONSTANTS, EXTENSION_SPRING_END_TYPES
from .goodman import Goodman, GoodmanData, GoodmanAnalyzer
from ..pymodels.material import Material
from .lineal import LinealSpring, TENSION
from pint import Quantity
from springcalc.pymodels.units import ureg
from pydantic import field_validator
from .plotting import interactive_backend

matplotlib.use('Agg')


class ExtensionSpring(LinealSpring):
    type_of_end: str = EXTENSION_SPRING_END_TYPES[1]
    wire_length: Optional[Quantity] = 0.0 * ureg.mm
    nr_coils: Optional[float] = None
    initial_force: Optional[Quantity] = 0.0 * ureg.N
    initial_stress: Optional[Quantity] = 0.0 * ureg.MPa
    length_btw_hooks: Optional[Quantity] = 0.0 * ureg.mm

    @field_validator("wire_length", mode="before")
    @classmethod
    def validate_wire_length(cls, value):
        if isinstance(value, str):
            try:
                return ureg(value)
            except Exception as e:
                raise ValueError(f"Error converting wire_length to Quantity: {e}")
        elif isinstance(value, (int, float)):
            return value * ureg.mm
        elif isinstance(value, Quantity):
            return value
        else:
            raise ValueError("wire_length must be a string, a number, or a Quantity")

    @field_validator("initial_force", mode="before")
    @classmethod
    def validate_initial_force(cls, value):
        if isinstance(value, str):
            try:
                return ureg(value)
            except Exception as e:
                raise ValueError(f"Error converting initial_force to Quantity: {e}")
        elif isinstance(value, (int, float)):
            return value * ureg.N
        elif isinstance(value, Quantity):
            return value
        else:
            raise ValueError("initial_force must be a string, a number, or a Quantity")

    @field_validator("initial_stress", mode="before")
    @classmethod
    def validate_initial_stress(cls, value):
        if isinstance(value, str):
            try:
                return ureg(value)
            except Exception as e:
                raise ValueError(f"Error converting initial_stress to Quantity: {e}")
        elif isinstance(value, (int, float)):
            return value * ureg.MPa
        elif isinstance(value, Quantity):
            return value
        else:
            raise ValueError("initial_stress must be a string, a number, or a Quantity")

    def __init__(self, material, wire_diameter: float, **data):
        super().__init__(material, wire_diameter, **data)
        self.number_cycles = int(1e6)
        self.shot_peening = False
        self.initial_stress = float(data.get("initial_stress", 0.0))
        self.initial_force = float(data.get("initial_force", 0.0))

    def set_material(self, material: str, wire_diameter: float):
        super().set_material(material, wire_diameter)
        if not isinstance(material, Material):
            raise ValueError("The material must be an instance of the Material class")
        if wire_diameter is None:
            raise ValueError("""You must provide the wire
                    diameter to set the material""")

    def set_diameter(
            self,
            outer_diameter: float = None,
            inner_diameter: float = None,
            mean_diameter: float = None,
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

        self.calculate_wahl_factor()

    def set_geometry(self,
                     mean_diameter: float = None,
                     outer_diameter: float = None,
                     inner_diameter: float = None,
                     nr_coils: float = None,
                     pitch: float = None,
                     free_length: float = None):
        """Set the spring's full geometry in one call.

        Provide exactly one of mean_diameter, outer_diameter, inner_diameter,
        and exactly two of nr_coils, pitch, free_length. Computes and stores
        every derived spring property, same as calling set_diameter() followed
        by calculate_spring_properties().
        """
        diameters_provided = sum(1 for var in [mean_diameter, outer_diameter, inner_diameter] if var is not None)
        if diameters_provided != 1:
            raise ValueError("You must provide exactly one of the following variables: mean_diameter, outer_diameter, inner_diameter")

        length_params_provided = sum(1 for var in [nr_coils, pitch, free_length] if var is not None)
        if length_params_provided != 3:
            raise ValueError("You must provide all three of the following variables: nr_coils, pitch, free_length")

        self.set_diameter(mean_diameter=mean_diameter,
                          outer_diameter=outer_diameter,
                          inner_diameter=inner_diameter)
        self.calculate_spring_properties(nr_coils=nr_coils,
                                         pitch=pitch,
                                         free_length=free_length)
        return self.get_spring_data()

    def calculate_spring_properties(self,
                                    nr_coils: float = None,
                                    pitch: float = None,
                                    free_length: float = None):
        if pitch is None:
            pitch = self.wire_diameter
            self.pitch = self.wire_diameter
        self.set_nr_coils(nr_coils=nr_coils,
                          pitch=pitch,
                          free_length=free_length)
        self.calculate_spring_constant()
        self.calculate_wire_length()

    def calculate_positions_table(self, step: List):
        for position in step:
            self.add_load_position(position)

    def set_mean_diameter(self, mean_diameter: float):
        if mean_diameter <= 0:
            raise ValueError("The mean diameter must be positive")
        self.mean_diameter = mean_diameter

    def calculate_mean_diameter(self, outer_diameter=None, inner_diameter=None):
        """Calculate the spring's mean diameter"""
        provided = sum(1 for var in [outer_diameter, inner_diameter] if var is not None)

        if provided == 0:
            raise ValueError("You must provide at least one of outer_diameter or inner_diameter")

        # Route through the pydantic-validated outer_diameter/inner_diameter
        # fields (same pattern as LinealSpring.set_diameter) so a plain
        # number or a Quantity both end up as a proper mm Quantity before
        # any arithmetic with self.wire_diameter -- mixing a bare number
        # with a Quantity in the same expression raises a pint
        # DimensionalityError instead of treating the number as mm.
        if outer_diameter is not None:
            self.outer_diameter = outer_diameter
            outer_diameter = self.outer_diameter
        if inner_diameter is not None:
            self.inner_diameter = inner_diameter
            inner_diameter = self.inner_diameter

        if provided == 2:
            self.mean_diameter = (outer_diameter + inner_diameter) / 2
            self.wire_diameter = outer_diameter - self.mean_diameter
            return self.mean_diameter

        if self.wire_diameter <= 0 or self.wire_diameter is None:
            raise ValueError("""The wire diameter must be a positive
                             non-null value to calculate the mean diameter""")
        if inner_diameter is not None:
            mean_diameter = inner_diameter + self.wire_diameter
        else:
            mean_diameter = outer_diameter - self.wire_diameter
        self.set_mean_diameter(mean_diameter)
        return self.mean_diameter

    def calculate_spring_index(self):
        self.spring_index = self.mean_diameter / self.wire_diameter
        return self.spring_index

    def calculate_active_coils(self):
        if self.type_of_end == "open":
            self.nr_active_coils = self.nr_coils - 0.5
        elif self.type_of_end == "closed":
            self.nr_active_coils = self.nr_coils - 1
        elif self.type_of_end == "semi-closed":
            self.nr_active_coils = self.nr_coils - 0.75
        else:
            self.nr_active_coils = self.nr_coils - 1

        if self.nr_active_coils <= 0:
            raise ValueError("The number of active coils must be greater than zero")
        return self.nr_active_coils

    def set_nr_coils(self,
                     nr_coils=None,
                     pitch=None,
                     free_length=None):

        self.free_length = nr_coils * pitch
        self.nr_coils = nr_coils
        self.pitch = pitch
        self.length_btw_hooks = self.free_length

        self.calculate_active_coils()

    def set_number_cycles(self, number_cycles):
        self.number_cycles = int(number_cycles)
        return self.number_cycles

    def set_initial_stress(self, initial_stress: float):
        if initial_stress < 0:
            raise ValueError("The initial stress cannot be negative")
        self.initial_stress = initial_stress
        return self.initial_stress

    def set_pitch_from_length(self, length: float):
        self.pitch = length / self.nr_coils
        return self.pitch

    def calculate_wire_length(self):
        print(f"Calculating wire length with mean_diameter={self.mean_diameter}, pitch={self.pitch}, nr_coils={self.nr_coils}")
        print(f"Formula used: wire_length = {self.nr_coils * sqrt((pi * self.mean_diameter) ** 2 + self.pitch**2)}")
        self.wire_length = self.nr_coils * sqrt((pi * self.mean_diameter) ** 2 + self.pitch**2)
        return self.wire_length

    def calculate_wahl_factor(self):
        if self.spring_index == 0:
            self.calculate_spring_index()

        self.wahl_factor = (4 * self.spring_index - 1) / (4 * self.spring_index - 4) + 0.615 / self.spring_index

        if self.spring_index < WAHL_FACTOR_CONSTANTS["red"][1]:
            self.wahl_factor_category = "red"
        elif WAHL_FACTOR_CONSTANTS["orange"][0] <= self.spring_index < WAHL_FACTOR_CONSTANTS["orange"][1]:
            self.wahl_factor_category = "orange"
        else:
            self.wahl_factor_category = "green"

        return self.wahl_factor

    def calculate_spring_constant(self):
        try:
            self.spring_constant = (
                self.material.shear_modulus * self.wire_diameter**4
            ) / (8 * self.mean_diameter**3 * self.nr_active_coils)
        except Exception as e:
            raise ValueError(f"Cannot calculate the spring constant: {e}")
        return self.spring_constant

    def calculate_pitch(self):
        return self.free_length / self.nr_coils

    def calculate_load_at_position(self, length: float):
        if self.spring_constant == 0:
            self.calculate_spring_constant()

        if length < self.length_btw_hooks.magnitude:
            raise ValueError("For an extension spring, the calculation length cannot be smaller than the free length")

        extension = (length - self.length_btw_hooks.magnitude) * ureg.mm
        print(f"Calculating load at position with initial_force={self.initial_force}, spring_constant={self.spring_constant}, extension={extension}")
        load = self.initial_force + self.spring_constant * extension
        return load

    def calculate_stress_at_position(self, load: float):
        try:
            # load = self.calculate_load_at_position(length, TENSION)
            stress = (8 * self.mean_diameter * load) / (3.1416 * self.wire_diameter**3) * self.wahl_factor
            return stress
        except ValueError as e:
            raise ValueError(f"Error calculating stress at position {length}: {e}")

    def calculate_outer_diameter_at_position(self, length: float):
        self.position_length = length
        extension = self.position_length - self.free_length
        outer_diameter = (
            self.mean_diameter
            + self.wire_diameter
            - self.mean_diameter * self.material.poisson_coef * extension / self.free_length
        )
        return outer_diameter

    def add_load_position(self, length: float):
        try:
            self.position_length = length
            load = self.calculate_load_at_position(length)
            stress = self.calculate_stress_at_position(load)
            outer_diameter = self.calculate_outer_diameter_at_position(self.position_length)
        except ValueError as e:
            raise ValueError(f"Error adding load position at length {length}: {e}")

        self.positions.add_load_position(
            position=self.position_length,
            travel=self.position_length - self.free_length,
            load=load,
            stress=stress,
            outer_diameter=outer_diameter,
            inner_diameter=max(outer_diameter - 2 * self.wire_diameter, 0)
        )

    def empty_tables(self):
        self.positions.clear_table()

    def get_spring_data(self):
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
            "wahl_factor": self.wahl_factor,
            "wahl_factor_category": self.wahl_factor_category,
            "wire_length": self.wire_length,
            "number_cycles": self.number_cycles,
            "shot_peening": self.shot_peening,
            "spring_index": self.spring_index,
            "pitch": self.pitch,
            "initial_stress": self.initial_stress,
        }

    def get_data_positions(self):
        return self.positions.positions

    def get_data_travels(self):
        return self.positions.positions

    def get_forces_vs_position_graph(self, show=False):
        def _to_mm_float(value):
            return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)

        def _to_n_float(value):
            return float(value.to('N').magnitude) if isinstance(value, Quantity) else float(value)

        positions_table = self.positions.positions
        if not positions_table:
            raise ValueError('No positions to plot')

        positions = [_to_mm_float(pc.position) for pc in positions_table]
        loads = [_to_n_float(pc.load) for pc in positions_table]

        with interactive_backend(show):
            plot = plt.figure()
            plt.plot(positions, loads, marker='o')
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
            plt.close(plot)
        return plot_data

    def get_forces_vs_travel_graph(self, show=False):
        def _to_mm_float(value):
            return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)

        def _to_n_float(value):
            return float(value.to('N').magnitude) if isinstance(value, Quantity) else float(value)

        positions_table = self.positions.positions
        if not positions_table:
            raise ValueError('No positions to plot')

        travels = [_to_mm_float(pc.travel) for pc in positions_table]
        loads = [_to_n_float(pc.load) for pc in positions_table]

        with interactive_backend(show):
            plot = plt.figure()
            plt.plot(travels, loads, marker='o')
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
            plt.close(plot)

        return plot_data

    def get_diameter_graph(self, show=False):
        def _to_mm_float(value):
            return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)

        positions_table = self.positions.positions
        positions = [_to_mm_float(pc.position) for pc in positions_table]
        diameters = [_to_mm_float(pc.outer_diameter) for pc in positions_table]

        with interactive_backend(show):
            plot = plt.figure()
            plt.plot(positions, diameters, marker="o", color="orange")
            plt.title("Outer Diameter vs Position")
            plt.xlabel("Position (mm)")
            plt.ylabel("Outer Diameter (mm)")
            plt.grid(True)
            if show:
                plt.show()
                return None
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close()

        return plot_data

    def get_diameter_vs_position_graph(self, show=False):
        def _to_mm_float(value):
            return float(value.to('mm').magnitude) if isinstance(value, Quantity) else float(value)

        positions_table = self.positions.positions
        positions = [_to_mm_float(pc.position) for pc in positions_table]
        diameters = [_to_mm_float(pc.outer_diameter) for pc in positions_table]

        outer_diameter = self.mean_diameter + self.wire_diameter
        inner_diameter = max(self.mean_diameter - self.wire_diameter, 0)
        outer_diameter_mm = _to_mm_float(outer_diameter)
        inner_diameter_mm = _to_mm_float(inner_diameter)

        with interactive_backend(show):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})

            ax1.plot(positions, diameters, marker="o", color="orange")
            ax1.set_title("Outer Diameter vs Position")
            ax1.set_xlabel("Position (mm)")
            ax1.set_ylabel("Outer Diameter (mm)")
            ax1.grid(True)

            ax2.set_aspect("equal")
            ax2.axis("off")

            outer_radius = outer_diameter_mm / 2.0
            inner_radius = inner_diameter_mm / 2.0
            max_radius = max(outer_radius, inner_radius, 1.0)
            padding = max_radius * 0.25

            ax2.add_patch(Circle((0, 0), outer_radius, fill=False, lw=2, color="tab:green"))
            if inner_radius > 0:
                ax2.add_patch(Circle((0, 0), inner_radius, fill=False, lw=2, color="tab:blue"))

            ax2.plot([-outer_radius, outer_radius], [0, 0], color="tab:green", lw=1)
            ax2.text(0, -padding, f"Dext = {outer_diameter_mm:.2f} mm", ha="center", va="top", fontsize=8)

            if inner_radius > 0:
                ax2.plot([0, 0], [-inner_radius, inner_radius], color="tab:blue", lw=1)
                ax2.text(0, padding, f"Dint = {inner_diameter_mm:.2f} mm", ha="center", va="bottom", fontsize=8)

            ax2.set_xlim(-max_radius - padding, max_radius + padding)
            ax2.set_ylim(-max_radius - padding, max_radius + padding)

            if show:
                plt.show()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode()
            buf.close()
            plt.close(fig)

        return plot_data

    def create_goodman_diagram(self, show=False):
        try:
            stress_max = self.get_stress_max()
            stress_min = self.get_stress_min()

            goodman_data = GoodmanData(
                material=self.material,
                diameter=self.wire_diameter,
                load_type="torsion",
                cycles=int(self.number_cycles),
            )

            analyzer = GoodmanAnalyzer(goodman_data, shot_peening=self.shot_peening)
            if show:
                analyzer.plot_diagram(stress_max, stress_min, show_plot=True)
                return

            fig = analyzer.plot_diagram(stress_max, stress_min, show_plot=False)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            image_b64 = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig)

            analysis = analyzer.get_analysis_summary(stress_max, stress_min)

            return {
                "image": image_b64,
                "analysis": analysis,
                "stresses": {
                    "stress_max": round(stress_max, 2),
                    "stress_min": round(stress_min, 2),
                    "load_max": round(self.get_load_max(), 2),
                    "load_min": round(self.get_load_min(), 2),
                },
            }
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error creating Goodman diagram in ExtensionSpring: {e}\n{tb}")
            return {"error": str(e), "traceback": tb}

    def get_goodman_graph(self):
        goodman_diagram = Goodman(
            material=self.material,
            diameter=self.wire_diameter,
            load_type="torsion",
            number_cycles=self.number_cycles,
            shot_peening=self.shot_peening,
        )
        stress_min = min(self.positions.positions, key=lambda x: x.stress).stress
        stress_max = max(self.positions.positions, key=lambda x: x.stress).stress
        goodman_fig = goodman_diagram.get_goodman_graph(stress_max, stress_min)
        return goodman_fig

    def get_stress_max(self):
        return max(self.positions.positions, key=lambda x: x.stress).stress

    def get_stress_min(self):
        return min(self.positions.positions, key=lambda x: x.stress).stress

    def get_load_max(self):
        return max(self.positions.positions, key=lambda x: x.load).load

    def get_load_min(self):
        return min(self.positions.positions, key=lambda x: x.load).load

    def plot_diagram(self):
        goodman_diagram = Goodman(
            material=self.material,
            diameter=self.wire_diameter,
            load_type="torsion",
            number_cycles=self.number_cycles,
            shot_peening=self.shot_peening,
        )
        stress_min = self.get_stress_min()
        stress_max = self.get_stress_max()
        goodman_diagram.plot_diagram(stress_max, stress_min)

    def get_goodman_analysis_summary(self):
        goodman_diagram = Goodman(
            material=self.material,
            diameter=self.wire_diameter,
            load_type="torsion",
            number_cycles=self.number_cycles,
            shot_peening=self.shot_peening,
        )
        stress_min = self.get_stress_min()
        stress_max = self.get_stress_max()
        return goodman_diagram.get_analysis_summary(stress_max, stress_min)
