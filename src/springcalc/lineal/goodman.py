import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pydantic import BaseModel, computed_field, field_validator
from ..pymodels.wire_characteristics import WireCharacteristics
from ..pymodels.material import Material
from springcalc.regresiones.factor_f.usar_modelo_factor_f import ModelFactorF
from math import log10
import io
import base64
from pint import Quantity
from ..pymodels.units import ureg
from .plotting import interactive_backend

class GoodmanData(BaseModel):
    """Data model for the Goodman diagram - validation and data only"""
    material: Material
    diameter: float
    load_type: str = "axial"
    cycles: int = 1e6  # Number of cycles for fatigue analysis, default 1 million

    @field_validator('diameter', mode='before')
    @classmethod
    def validate_diameter(cls, v):
        if isinstance(v, Quantity):
            return float(v.to('mm').magnitude)
        if isinstance(v, (int, float)):
            return float(v)
        return float(ureg(v).to('mm').magnitude)

    @computed_field
    @property
    def wire_characteristics(self) -> WireCharacteristics:
        """Wire characteristics computed automatically"""
        return WireCharacteristics(material=self.material, wire_diameter=self.diameter)


class GoodmanAnalyzer:
    """Service for Goodman analysis - business logic and calculations"""

    def __init__(self, data: GoodmanData, shot_peening: bool = False):
        self.data = data
        self.wire_char = data.wire_characteristics
        self.shot_peening = shot_peening
        self._calculate_factors()

    def _calculate_factors(self):
        """Calculate all strength factors according to Shigley"""
        # 1. Ultimate tensile strength (RMa_min) of the material
        self.Sut = self.wire_char.RMa_min

        # 2. Maximum shear strength (Ssu) - approximated as 0.75 * RMa_min for steels
        self.Ssu = 0.67 * self.Sut
        # 3. Torsional yield strength (Ssy ≈ 0.45 * Sut to 0.56 * Sut)
        self.Ssy = 0.45 * self.Sut
        # 4. Uncorrected shear fatigue limit (Sse')
        # For steel springs, Sse' ≈ 0.5 * Sut is often used for infinite life
        self.Sse_prime = self.wire_char.material.elastic_limit_factor * self.Sut

        # Shigley Mechanical Engineering Design, 9th edition, section 6-8, p. 274
        # Calculation of the fatigue-strength correction factors (Marin factors)
        # surface factor
        if self.shot_peening:
            # Shot peening improves fatigue strength, so a correction factor of 1 can be used
            self.k_a = 1
        else:
            self.k_a = 4.51 * self.Sut**(-0.265)

        # size factor
        if self.data.load_type in ["torsion", "flexion"]:
            # Shigley size factor formula using diameter in mm (2.79 mm <= d <= 51 mm)
            if 2.79 <= self.data.diameter <= 51:
                self.k_b = 1.24 * (self.data.diameter ** -0.107)
            elif self.data.diameter > 51:
                self.k_b = 1.51 * (self.data.diameter ** -0.157)
            else:
                self.k_b = 1.0
        else:
            self.k_b = 1

        # load factor
        if self.data.load_type == "flexion":
            self.k_c = 1.0
        elif self.data.load_type == "axial":
            self.k_c = 0.85
        else:
            self.k_c = 1.0

        # temperature factor
        self.k_d = 1.0
        # reliability factor
        self.k_e = 1.0
        # fatigue factor
        # Apply all Marin modification factors to the uncorrected endurance limit Sse'
        self.Sse = self.k_a * self.k_b * self.k_c * self.k_d * self.k_e * self.Sse_prime
        factor_f_model = ModelFactorF()
        self.factor_f = factor_f_model.predict(self.Ssu)
        if self.data.cycles <= 1e3:
            # Low-cycle fatigue strength approximation
            self.Ssf = self.Sut * (self.data.cycles ** (log10(self.factor_f) / 3))
        elif self.data.cycles >= 1e6:
            # Infinite life region
            self.Ssf = self.Sse
        else:
            # Finite-life region (10^3 < N < 10^6 cycles) using corrected endurance limit (Sse)
            # S_f = a * N^b, where S_f(10^3) = f * Sut and S_f(10^6) = Sse
            a = ((self.factor_f * self.Sut) ** 2) / self.Sse
            b = -log10((self.factor_f * self.Sut) / self.Sse) / 3
            self.Ssf = a * (self.data.cycles ** b)
    @staticmethod
    def _to_mpa_float(value) -> float:
        if isinstance(value, Quantity):
            return float(value.to('MPa').magnitude)
        return float(value)

    def plot_diagram(self, sigma_max: float, sigma_min: float, show_plot: bool = True):
        """
        Plot the Goodman diagram with the operating point marked

        Args:
            sigma_max: Maximum stress of the load cycle
            sigma_min: Minimum stress of the load cycle
            show_plot: Whether to show the plot immediately

        Returns:
            matplotlib Figure for further flexibility
        """
        sigma_max = self._to_mpa_float(sigma_max)
        sigma_min = self._to_mpa_float(sigma_min)

        with interactive_backend(show_plot):
            fig, ax = plt.subplots(figsize=(10, 8))

            # Effective fatigue strength Sn capped at ultimate tensile strength Sut
            raw_sn = self.Ssf if self.data.cycles < 1e6 else self.Sse
            Sn = min(raw_sn, self.Sut)

            # Intersection point calculation bounded to non-negative values
            if Sn >= self.Ssy:
                sm_yield = 0.0
            else:
                denom = 1.0 - (Sn / self.Sut) if self.Sut != Sn else 1e-6
                sm_yield = max(0.0, min((self.Ssy - Sn) / denom, self.Ssy))

            # Upper boundary line (Max. stress)
            max_line_x = [0, sm_yield, self.Ssy]
            max_line_y = [min(Sn, self.Ssy), self.Ssy, self.Ssy]

            # Lower boundary line (Min. stress)
            min_line_x = [0, sm_yield, self.Ssy]
            min_line_y = [-min(Sn, self.Ssy), 2 * sm_yield - self.Ssy, self.Ssy]

            # Plot boundaries
            ax.plot(max_line_x, max_line_y, 'b-', linewidth=2, label='Max stress boundary')
            ax.plot(min_line_x, min_line_y, 'b-', linewidth=2, label='Min stress boundary')

            # Fill safe region
            envelope_x = max_line_x + min_line_x[::-1]
            envelope_y = max_line_y + min_line_y[::-1]
            ax.fill(envelope_x, envelope_y, alpha=0.25, color='lightblue', label='Safe region')

            # Midrange reference line (45 degrees)
            ax.plot([0, self.Ssy], [0, self.Ssy], 'k--', linewidth=1, alpha=0.7, label='Midrange line (45°)')

            # Operating point
            mean_tension = (sigma_max + sigma_min) / 2
            amplitude = (sigma_max - sigma_min) / 2

            ax.plot([mean_tension, mean_tension], [sigma_min, sigma_max],
                    'ro-', linewidth=2, markersize=8, label='Operating point')
            ax.plot(mean_tension, mean_tension, 'go', markersize=10, label=f'σₘ={mean_tension:.1f}, σₐ={amplitude:.1f}')

            # Plot configuration
            ax.set_title(f'Goodman Diagram - Material: {self.data.material.material_name}')
            ax.set_xlabel('Mean Tension σₘ (MPa)')
            ax.set_ylabel('Alternating Tension σₐ (MPa)')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add technical info
            info_text = f"""Goodman Factors:
        Nr of cycles: {self.data.cycles:.1e}
        Correction factors
        kₐ = {self.k_a:.3f}
        k_b = {self.k_b:.3f}
        k_c = {self.k_c:.3}
        Sut = {self.Ssu:.1f} MPa
        Se = {self.Sse:.1f} MPa
        Sy = {self.Ssy:.1f} MPa
        Sf = {self.Ssf:.1f} MPa
        Security factor: {self.calculate_safety_factor(sigma_max, sigma_min):.2f}"""

            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            if show_plot:
                plt.show()

        return fig

    def get_diagram_image(self, sigma_max: float, sigma_min: float):
        """Return the Goodman diagram image in base64"""
        fig = self.plot_diagram(sigma_max, sigma_min, show_plot=False)
        # Save the diagram as base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        goodman_image = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return goodman_image

    def calculate_safety_factor(self, sigma_max: float, sigma_min: float) -> float:
        """
        Calculate the safety factor of the operating point. Shigley 6.12, p. 275

        Args:
            sigma_max: Maximum stress of the cycle
            sigma_min: Minimum stress of the cycle

        Returns:
            Safety factor
        """
        sigma_max = self._to_mpa_float(sigma_max)
        sigma_min = self._to_mpa_float(sigma_min)

        mean_tension = (sigma_max + sigma_min) / 2
        amplitude = (sigma_max - sigma_min) / 2

        # sigma_max/sigma_min are shear stresses for torsion/flexion loading,
        # so the ultimate strength used against them must be the shear
        # ultimate (Ssu), matching the shear yield (Ssy) used below.
        ultimate = self.Ssu if self.data.load_type in ("torsion", "flexion") else self.Sut

        # 1. Pure static load case (no cyclic amplitude)
        if amplitude <= 0:
            if mean_tension <= 0:
                return float('inf')
            return ultimate / mean_tension

        # 2. Select fatigue limit according to target cycle count
        # If cycles < 1e6 se we have to use finite life (Ssf), otherwise Sse.
        Sn = self.Ssf if self.data.cycles < 1e6 else self.Sse

        # 3. Fatigue factor of safety (Modified Goodman criterion)
        n_fatigue = 1 / ((amplitude / Sn) + (mean_tension / ultimate))

        # 4. Static yield factor of safety (Langer yield guard line)
        n_yield = self.Ssy / (amplitude + mean_tension)

        # Governing safety factor is the minimum between fatigue and yield
        return max(0.0, min(n_fatigue, n_yield))

    def get_analysis_summary(self, sigma_max: float, sigma_min: float) -> dict:
        """
        Return a complete summary of the Goodman analysis

        Returns:
            Dictionary with all calculated parameters
        """
        sigma_max = self._to_mpa_float(sigma_max)
        sigma_min = self._to_mpa_float(sigma_min)

        return {
            'material': self.data.material.material_name,
            'diameter': self.data.diameter,
            'load_type': self.data.load_type,
            'factors': {
                'k_a': self.k_a,
                'k_b': self.k_b,
                'k_c': self.k_c,
                'k_d': self.k_d,
                'k_e': self.k_e
            },
            'strengths': {
                'Se_MPa': self.Sse,
                'Sf_MPa': self.Ssf,
                'RMa_min_MPa': self.wire_char.RMa_min,
                'RMa_max_MPa': self.wire_char.RMa_max
            },
            'operation_point': {
                'sigma_max_MPa': sigma_max,
                'sigma_min_MPa': sigma_min,
                'mean_tension_MPa': (sigma_max + sigma_min) / 2,
                'amplitude_MPa': (sigma_max - sigma_min) / 2
            },
            'safety_factor': self.calculate_safety_factor(sigma_max, sigma_min)
        }


# Backwards compatibility: keep the original interface so existing code doesn't break
class Goodman(GoodmanAnalyzer):
    """Backwards-compatibility class - uses the new architecture internally"""

    def __init__(self, material: Material, diameter: float, load_type: str = "axial", number_cycles: int = 1e6, shot_peening: bool = False):
        data = GoodmanData(material=material, diameter=diameter, load_type=load_type, cycles=number_cycles)
        super().__init__(data, shot_peening=shot_peening)

    def plot_goodman_graph(self, sigma_max: float, sigma_min: float):
        """Original method kept for backwards compatibility"""
        return self.plot_diagram(sigma_max, sigma_min, show_plot=True)

    def get_goodman_graph(self, sigma_max: float, sigma_min: float):
        """Original method kept for backwards compatibility"""
        return self.plot_diagram(sigma_max, sigma_min, show_plot=False)
