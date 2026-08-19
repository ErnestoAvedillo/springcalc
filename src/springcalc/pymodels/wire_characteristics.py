from pydantic import BaseModel, field_validator, model_validator, ConfigDict
from .material import Material
import os
from pandas import read_csv
from pint import Quantity
from .units import ureg


def _diameter_to_mm_value(wire_diameter) -> float:
    """Convert the wire diameter to a scalar value in millimeters."""
    if isinstance(wire_diameter, Quantity):
        return float(wire_diameter.to('mm').magnitude)
    if isinstance(wire_diameter, (int, float)):
        return float(wire_diameter)
    return float(ureg.Quantity(wire_diameter).to('mm').magnitude)


def get_standard_wire_diameters() -> list:
    """Return the standard wire diameter series (mm) listed in DIAMETRO_TOLERANCIAS.csv."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    material_dir = os.path.join(os.path.dirname(current_dir), 'material')
    file = os.path.join(material_dir, "DIAMETRO_TOLERANCIAS.csv")
    df = read_csv(file)
    return sorted(float(d) for d in df['diameter'])


def get_wire_tolerance(wire_diameter: float) -> float:
    """Get the wire diameter tolerance based on the diameter"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    material_dir = os.path.join(os.path.dirname(current_dir), 'material')
    file = os.path.join(material_dir, "DIAMETRO_TOLERANCIAS.csv")

    try:
        wire_diameter_mm = _diameter_to_mm_value(wire_diameter)
        pd = read_csv(file)
        if pd.iloc[0]['diameter'] > wire_diameter_mm:
            return pd.iloc[0]['tolerance']
        if wire_diameter_mm > pd.iloc[len(pd) - 1]['diameter']:
            return pd.iloc[len(pd) - 1]['tolerance']
        last_row = pd.iloc[0]
        for index, row in pd.iterrows():
            if row['diameter'] >= wire_diameter_mm:
                return last_row['tolerance']
            last_row = row
        return pd.iloc[len(pd) - 1]['tolerance']
    except Exception as e:
        print(f"Error getting tolerance: {e}")
        return 0.1  # Default value

def get_RMa_range(material, wire_diameter: float) -> tuple:
    """Get the RMa range for a given material and wire diameter"""
    try:
        wire_diameter_mm = _diameter_to_mm_value(wire_diameter)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        material_dir = os.path.join(os.path.dirname(current_dir), 'material')
        file = os.path.join(material_dir, material.RMa_file)

        if not os.path.exists(file):
            print(f"RMa file not found: {file}")
            return (1000.0, 1200.0)  # Default values

        df = read_csv(file)
        df.columns = df.columns.str.strip()  # Strip whitespace from column names

        # If the diameter is smaller than the first value, use the first one
        if wire_diameter_mm <= df.iloc[0]['diameter']:
            return (float(df.iloc[0]['RMa_min']), float(df.iloc[0]['RMa_max']))

        # If the diameter is larger than the last value, use the last one
        if wire_diameter_mm >= df.iloc[len(df) - 1]['diameter']:
            return (float(df.iloc[len(df) - 1]['RMa_min']), float(df.iloc[len(df) - 1]['RMa_max']))

        # Find between which values it falls and return the lower value
        prev_row = df.iloc[0]
        for index, row in df.iterrows():
            if row['diameter'] >= wire_diameter_mm:
                return (float(row['RMa_min']), float(row['RMa_max']))
            prev_row = row

        return (1000.0, 1200.0)  # Default values

    except Exception as e:
        print(f"Error getting RMa range: {e}")
        return (1000.0, 1200.0)  # Default values

class WireCharacteristics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    material: Material
    wire_diameter: Quantity
    diameter_tolerance: Quantity = None
    RMa_min: Quantity = None
    RMa_max: Quantity = None

    @field_validator('wire_diameter', 'diameter_tolerance', mode='before')
    @classmethod
    def validate_wire_diameter(cls, v):
        """Validate that the wire diameter is positive"""
        if v is None:
            return v
        try:
            if isinstance(v, Quantity):
                quantity = v.to('mm')
            else:
                quantity = (v * ureg.mm)
            if quantity.magnitude <= 0:
                raise ValueError("The wire diameter must be positive")
            return quantity
        except Exception as e:
            raise ValueError(f"The wire diameter must be positive: {e}")

    @field_validator('RMa_min', 'RMa_max', mode='before')
    @classmethod
    def validate_RMa(cls, v):
        """Validate that RMa_min and RMa_max are positive"""
        if v is not None:
            try:
                quantity = ureg(v).to('MPa')
                if quantity.magnitude <= 0:
                    raise ValueError("RMa_min and RMa_max must be positive values with valid MPa units.")
                return quantity
            except Exception as e:
                raise ValueError(f"Error validating RMa: {e}")
        return v

    def __str__(self):
        return (f"Material: {self.material.material_name}, "
                f"Wire Diameter: {self.wire_diameter} mm, "
                f"Diameter Tolerance: {self.diameter_tolerance} mm, "
                f"RMa Min: {self.RMa_min} MPa, "
                f"RMa Max: {self.RMa_max} MPa")

    @model_validator(mode='after')
    def assign_wire_characteristics(self):
        """Automatically assign the wire characteristics based on material and diameter"""
        if self.wire_diameter is not None:
            object.__setattr__(
                self,
                'diameter_tolerance',
                get_wire_tolerance(self.wire_diameter),
            )

        if self.material and self.wire_diameter and (self.RMa_min is None or self.RMa_max is None):
            rma_min, rma_max = get_RMa_range(self.material, self.wire_diameter)
            object.__setattr__(self, 'RMa_min', rma_min)
            object.__setattr__(self, 'RMa_max', rma_max)

        return self

    def set_material(self, material:str, wire_diameter:float):
        """Set the spring's material"""
        if not isinstance(material, Material):
            raise ValueError("The material must be an instance of the Material class")
        if wire_diameter is None:
            raise ValueError("You must provide the wire diameter to set the material")
        self.material = material
        self.wire_diameter = wire_diameter
        self.diameter_tolerance = get_wire_tolerance(wire_diameter)
        self.RMa_min, self.RMa_max = get_RMa_range(material, wire_diameter)
