from .constants import WAHL_FACTOR_CONSTANTS
from ..pymodels.wire_characteristics import WireCharacteristics
from ..pymodels.material import Material
from ..pymodels.positions import LinearPositionsTable
from typing import Optional
from pint import Quantity
from ..pymodels.units import ureg
from pydantic import field_validator, ConfigDict
import numpy as np
"""Class for calculating a linear spring"""
COMPRESSION = 1
TENSION = -1


class LinealSpring(WireCharacteristics):
    # Additional LinealSpring fields
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    mean_diameter: Quantity = 0.0 * ureg.mm  # in mm
    outer_diameter: Quantity = 0.0 * ureg.mm  # in mm
    inner_diameter: Quantity = 0.0 * ureg.mm   # in mm
    free_length: Quantity = 0.0 * ureg.mm  # in mm
    nr_active_coils: float = 0.0
    pitch: Quantity = 0.0 * ureg.mm  # in mm
    shot_peening: bool = False
    coating: Optional[str] = None
    spring_index: float = 0.0
    positions: LinearPositionsTable = LinearPositionsTable()
    spring_constant: Quantity = 0.0 * ureg.N / ureg.mm  # in N/mm
    wahl_factor: float = 0.0  # Wahl factor
    wahl_factor_category: Optional[str] = None  # Wahl factor category
    wahl_factor_eval: Optional[float] = None  # evaluated Wahl factor
    number_cycles: int = 1e6  # Number of cycles for fatigue analysis, default 1 million
    position_length: Optional[Quantity] = 0.0 * ureg.mm  # Length at which the load is applied for fatigue analysis
    position_load: Optional[Quantity] = 0.0 * ureg.N  # Load at the position length for fatigue analysis
    position_stress: Optional[Quantity] = 0.0 * ureg.megapascal  # Stress from the load for fatigue analysis

    @field_validator('free_length',
                     'position_length',
                     'outer_diameter',
                     'inner_diameter',
                     'mean_diameter',
                     'pitch',
                     mode='before')
    @classmethod
    def validate_dimensions(cls, v):
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

    @field_validator('spring_constant', mode='before')
    @classmethod
    def validate_spring_constant(cls, v):
        if v is not None:
            try:
                if isinstance(v, Quantity):
                    quantity = v.to('N/mm')
                elif isinstance(v, (int, float)):
                    quantity = v * ureg.N / ureg.mm
                else:
                    quantity = ureg(v).to('N/mm')
                if quantity.magnitude < 0:
                    raise ValueError(f'{quantity} must be a positive value with valid units.')
                return quantity
            except Exception as e:
                raise ValueError(f'Error validating quantity: {e}')
        return quantity

    @field_validator('position_load', mode='before')
    @classmethod
    def validate_position_load(cls, v):
        if v is not None:
            try:
                if isinstance(v, Quantity):
                    quantity = v.to('N')
                elif isinstance(v, (int, float)):
                    quantity = v * ureg.N
                else:
                    quantity = ureg(v).to('N')
                if quantity.magnitude < 0:
                    raise ValueError(f'{quantity} must be a positive value with valid units.')
                return quantity
            except Exception as e:
                raise ValueError(f'Error validating quantity: {e}')
        return quantity

    @field_validator('position_stress', mode='before')
    @classmethod
    def validate_position_stress(cls, v):
        if v is not None:
            try:
                if isinstance(v, Quantity):
                    quantity = v.to('MPa')
                elif isinstance(v, (int, float)):
                    quantity = v * ureg.megapascal
                else:
                    quantity = ureg(v).to('MPa')
                if quantity.magnitude < 0:
                    raise ValueError(f'{quantity} must be a positive value with valid units.')
                return quantity
            except Exception as e:
                raise ValueError(f'Error validating quantity: {e}')
        return quantity

    def __init__(self, material:Material, wire_diameter: float, **data):
        """initialize the spring variables to 0"""
        # Initialize with default values

        data.update({
            'material': material,
            'wire_diameter': wire_diameter,
            'positions': LinearPositionsTable()
        })
        super().__init__(**data)
        self.number_cycles = 1e6  # Default value of 1 million cycles for fatigue analysis
        self.shot_peening = False  # Default value, no shot peening

    def set_material(self, material:str, wire_diameter:float):
        """First part: set the spring's material"""
        super().set_material(material, wire_diameter)
        if not isinstance(material, Material):
            raise ValueError("The material must be an instance of the Material class")
        if wire_diameter is None:
            raise ValueError("You must provide the wire diameter to set the material")

    def set_diameter(self, mean_diameter=None, outer_diameter=None, inner_diameter=None):
        """Second part: validate and set the spring diameters"""
        parametres_provided = sum(1 for var in [outer_diameter, inner_diameter, mean_diameter] if var is not None)
        if parametres_provided != 1:
            raise ValueError("""You must provide exactly one of the following variables:
                             outer_diameter, inner_diameter, mean_diameter""")
        if outer_diameter is not None:
            self.outer_diameter = outer_diameter
            self.mean_diameter = self.outer_diameter - self.wire_diameter
            self.inner_diameter = self.outer_diameter - 2 * self.wire_diameter
        elif inner_diameter is not None:
            self.inner_diameter = inner_diameter
            self.mean_diameter = self.inner_diameter + self.wire_diameter
            self.outer_diameter = self.inner_diameter + 2 * self.wire_diameter
        elif mean_diameter is not None:
            self.mean_diameter = mean_diameter
            self.outer_diameter = self.mean_diameter + self.wire_diameter
            self.inner_diameter = self.mean_diameter - self.wire_diameter
        self.calculate_wahl_factor()

    def calculate_spring_index(self):
        """Calculate the spring index"""
        self.spring_index = self.mean_diameter / self.wire_diameter
        return self.spring_index

    def set_nr_active_coils(self, nr_active_coils=None, pitch=None, free_length=None):
        """Set the number of active spring coils"""
        numero_variables = sum(1 for var in [nr_active_coils, pitch, free_length] if var is not None)
        if numero_variables != 2:
            raise ValueError("You must provide exactly two of the three variables: nr_active_coils, pitch, free_length")
        if not pitch:
            self.pitch = free_length / nr_active_coils
            self.free_length = free_length
        elif not nr_active_coils:
            self.nr_active_coils = free_length / pitch
            self.pitch = pitch
            self.free_length = free_length
        elif not free_length:
            self.free_length = nr_active_coils * pitch
            self.pitch = pitch

    def set_number_cycles(self, number_cycles):
        """Set the number of cycles for fatigue analysis"""
        self.number_cycles = number_cycles
        return self.number_cycles

    def calculate_pitch(self, length:float):
        """Calculate the spring pitch"""
        self.free_length = length
        self.pitch = length / self.nr_active_coils
        return self.pitch

    def calculate_wahl_factor(self):
        """Calculate the Wahl factor"""
        if self.spring_index == 0:
            self.calculate_spring_index()
        self.wahl_factor = (4 * self.spring_index - 1) / (4 * self.spring_index - 4) + 0.615 / self.spring_index
        """Classify the spring index into a manufacturability category"""
        if self.spring_index < WAHL_FACTOR_CONSTANTS['red'][1]:
            self.wahl_factor_category = 'red'
        elif WAHL_FACTOR_CONSTANTS['orange'][0] <= self.spring_index < WAHL_FACTOR_CONSTANTS['orange'][1]:
            self.wahl_factor_category = 'orange'
        else:
            self.wahl_factor_category = 'green'
        return self.wahl_factor, self.wahl_factor_category

    def calculate_spring_constant(self):
        """Calculate the spring constant (N/mm)"""
        try:
            self.spring_constant = (self.material.shear_modulus * np.power(self.wire_diameter,4)) / (8 * np.power(self.mean_diameter,3) * self.nr_active_coils)
        except ValueError:
            raise ValueError("Cannot calculate the spring constant without the mean diameter")
        return self.spring_constant

    def calculate_load_at_position(self, length:float, spring_class=COMPRESSION):
        """Calculate the load at a given position using the spring constant"""
        if self.spring_constant == 0:
            self.calculate_spring_constant()
        self.position_length = length
        self.position_load = spring_class * self.spring_constant * (self.free_length - self.position_length)
        self.calculate_stress_at_position(self.position_load)
        return self.position_load

    def calculate_stress_at_position(self, load: float):
        """Calculate the stress in the spring wire at a given predefined position"""
        try:
            stress = (8 * self.mean_diameter * load) / (3.1416 * self.wire_diameter**3) * self.wahl_factor
            return stress
        except ValueError as e:
            raise ValueError(f"Error calculating stress at position with load {load}: {e}")
