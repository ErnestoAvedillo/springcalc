from pydantic import (BaseModel,
                      field_validator,
                      model_validator,
                      ConfigDict,
                      ValidationInfo)
import pandas as pd
import os
from typing import List, Optional, Sequence, Tuple, Union
from pint import Quantity
from .units import ureg


def get_material_dir() -> str:
    """Get the directory where materials.csv and RMa_file tables live"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(current_dir), 'material')


def get_materials_csv_path() -> str:
    """Get the path to materials.csv"""
    return os.path.join(get_material_dir(), 'materials.csv')


def get_materials_dataframe() -> pd.DataFrame:
    """Get the full materials DataFrame from materials.csv"""
    csv_path = get_materials_csv_path()

    try:
        df = pd.read_csv(csv_path)
        # Strip whitespace from all text columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        return df
    except Exception as e:
        print(f"Error reading materials.csv: {e}")
        return pd.DataFrame()


def get_available_materials() -> List[str]:
    """Get the list of available materials from materials.csv"""
    df = get_materials_dataframe()
    if not df.empty:
        return df['denomination'].tolist()
    return []


class Material(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    material_name: str
    young_modulus: Optional[Quantity] = None
    shear_modulus: Optional[Quantity] = None
    elastic_limit_factor: Optional[Quantity] = None
    poisson_coef: Optional[Quantity] = None
    RMa_file: Optional[str] = None

    @field_validator('material_name')
    @classmethod
    def validate_material_name(cls, v):
        """Validate that the material name is in the
        list of available materials"""
        available_materials = get_available_materials()
        if v not in available_materials:
            err_msg = f'Material "{v}" is not valid. \n'
            err_msg += f'Available materials: {", ".join(available_materials)}'
            raise ValueError(err_msg)
        return v

    @field_validator('young_modulus',
                     'shear_modulus',
                     'elastic_limit_factor',
                     'poisson_coef', mode='before')
    @classmethod
    def validate_positive_quantities(cls, v, info: ValidationInfo):
        """Validate that the quantities are positive
        and have the correct units"""
        if v is not None:
            try:
                quantity = ureg(v).to('MPa') if info.field_name in ['young_modulus', 'shear_modulus'] else ureg(v)
                if quantity.magnitude <= 0:
                    error_msg = f'{info.field_name} must be a positive value with valid units: '
                    error_msg += 'Pascal (Pa) for young_modulus and shear_modulus, '
                    error_msg += 'unitless for elastic_limit_factor, and dimensionless for poisson_coef.'
                    raise ValueError(error_msg)
                return quantity
            except Exception as e:
                raise ValueError(f'Error validating {info.field_name}: {e}')
        return v

    @model_validator(mode='after')
    def assign_material_properties(self):
        """Automatically assign material properties based on materials.csv"""
        if self.material_name:
            df = get_materials_dataframe()
            if not df.empty:
                # Strip whitespace from column names
                df.columns = df.columns.str.strip()

                # Find the row matching the material
                material_row = df[df['denomination'] == self.material_name]

                if not material_row.empty:
                    row = material_row.iloc[0]

                    # Assign properties automatically 
                    # (only if not already set and the value is not empty)
                    if self.young_modulus is None and 'young_modulus' in df.columns:
                        val = row.get('young_modulus')
                        if pd.notna(val) and str(val).strip() != '':
                            self.young_modulus = float(val) * ureg.MPa

                    if self.shear_modulus is None and 'shear_modulus' in df.columns:
                        val = row.get('shear_modulus')
                        if pd.notna(val) and str(val).strip() != '':
                            self.shear_modulus = float(val) * ureg.MPa

                    if self.elastic_limit_factor is None and 'elastic_limit_factor' in df.columns:
                        val = row.get('elastic_limit_factor')
                        if pd.notna(val) and str(val).strip() != '':
                            self.elastic_limit_factor = float(val) * ureg.dimensionless

                    if self.poisson_coef is None and 'poisson_coef' in df.columns:
                        val = row.get('poisson_coef')
                        if pd.notna(val) and str(val).strip() != '':
                            self.poisson_coef = float(val) * ureg.dimensionless

                    if self.RMa_file is None and 'RMa_file' in df.columns:
                        val = row.get('RMa_file')
                        if pd.notna(val) and str(val).strip() != '':
                            self.RMa_file = str(val).strip()

        return self

    @classmethod
    def material_exists(cls, material_name: str) -> bool:
        """Check whether a material is already registered in materials.csv"""
        return material_name in get_available_materials()

    @classmethod
    def list_available_materials(cls) -> List[str]:
        """Get the list of available material names"""
        return get_available_materials()

    @staticmethod
    def _to_magnitude(value: Union[str, float, int, Quantity],
                      unit: Optional[str]) -> float:
        """Resolve a value (string, number or Quantity)
        to a plain positive magnitude in `unit`"""
        if isinstance(value, Quantity):
            quantity = value
        elif isinstance(value, str):
            quantity = ureg(value)
        else:
            quantity = value * ureg(unit) if unit else value * ureg.dimensionless
        if unit:
            quantity = quantity.to(unit)
        if quantity.magnitude <= 0:
            raise ValueError(f'Value must be positive: {value}')
        return quantity.magnitude

    @classmethod
    def create_material(
        cls,
        material_name: str,
        young_modulus: Union[str, float, int, Quantity],
        shear_modulus: Union[str, float, int, Quantity],
        elastic_limit_factor: Union[str, float, int, Quantity],
        poisson_coef: Union[str, float, int, Quantity],
        description: str = "",
        RMa_file: Optional[str] = None,
        RMa_data: Optional[Sequence[Tuple[float, float, float]]] = None,
        overwrite: bool = False,
    ) -> "Material":
        """Register a new material in materials.csv and return it as a Material instance.

        young_modulus/shear_modulus may be given as strings with units (e.g. "206000 MPa"),
        plain numbers (assumed MPa) or pint Quantities; elastic_limit_factor and poisson_coef
        are treated as dimensionless. RMa_data, when provided, is a sequence of
        (diameter, RMa_min, RMa_max) rows sorted by ascending diameter, written to RMa_file
        (auto-named "<material_name>_RMa.csv" when RMa_file is not given).
        """
        if not overwrite and cls.material_exists(material_name):
            raise ValueError(f'Material "{material_name}" already exists. Pass overwrite=True to replace it.')

        young_modulus_value = cls._to_magnitude(young_modulus, 'MPa')
        shear_modulus_value = cls._to_magnitude(shear_modulus, 'MPa')
        elastic_limit_factor_value = cls._to_magnitude(elastic_limit_factor, None)
        poisson_coef_value = cls._to_magnitude(poisson_coef, None)

        if RMa_data is not None and RMa_file is None:
            RMa_file = f"{material_name}_RMa.csv"

        csv_path = get_materials_csv_path()
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df = df[df['denomination'].str.strip() != material_name]

        new_row = {
            'denomination': material_name,
            'young_modulus': young_modulus_value,
            'shear_modulus': shear_modulus_value,
            'elastic_limit_factor': elastic_limit_factor_value,
            'poisson_coef': poisson_coef_value,
            'RMa_file': RMa_file or '',
            'description': description,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

        if RMa_data is not None:
            rma_df = pd.DataFrame(list(RMa_data), columns=['diameter', 'RMa_min', 'RMa_max'])
            rma_df = rma_df.sort_values('diameter').reset_index(drop=True)
            rma_df.to_csv(os.path.join(get_material_dir(), RMa_file), index=False)

        return cls(material_name=material_name)
