"""springcalc — spring calculation library (compression, extension, and torsion).

Main public API:
    - CompressionSpring, ExtensionSpring, TorsionSpring: calculations per spring type.
    - Material, get_available_materials: material data.
    - GoodmanData, GoodmanAnalyzer, Goodman: fatigue analysis (Goodman diagram).
"""

from springcalc.lineal.compresion import (CompressionSpring,
                                          COMPRESSION_SPRING_END_TYPES)
from springcalc.lineal.generic_compression import CompressionSpringGeneral
from springcalc.lineal.extension import ExtensionSpring
from springcalc.lineal.torsion import TorsionSpring
from springcalc.lineal.animation import CompressionAnimator
from springcalc.lineal.goodman import Goodman, GoodmanAnalyzer, GoodmanData
from springcalc.pymodels.material import Material, get_available_materials, get_materials_dataframe
from springcalc.pymodels.positions import (
    LinearPosition,
    AngularPosition,
    LinearLoadPosition,
    AngularLoadPosition,
    LinearPositionsTable,
    AngularPositionsTable,
)
from springcalc.pymodels.wire_characteristics import get_wire_tolerance, get_RMa_range
from springcalc.plots.goodman_diagram import generate_goodman_diagram
from springcalc.report import (
    SpringPDFReport,
    SupplierSpecSheet,
    spring_geometry_to_dict,
    spring_geometry_to_json,
    spring_geometry_to_csv,
    spring_geometry_profile,
    spring_geometry_to_step,
)
from springcalc.pymodels.units import ureg
from springcalc.lineal.plotting import interactive_backend
from springcalc.persistence import (
    export_model,
    import_model,
    model_to_json,
    model_from_json,
    save_model,
    load_model,
)
from importlib.metadata import version as _version
from springcalc.inverse_calc.lineal_comp_inv import (
    CompressionSpringInverseDesigner
)
from springcalc.inverse_calc.conical_comp_inv import (
    ConicalCompressionSpringInverseDesigner,
)
from springcalc.inverse_calc.conical_curve_comp_inv import (
    ConicalCurveCompressionSpringInverseDesigner,
)
__version__ = _version("springcalc")

__all__ = [
    "CompressionSpring",
    "CompressionSpringGeneral",
    "ExtensionSpring",
    "TorsionSpring",
    "CompressionAnimator",
    "Material",
    "get_available_materials",
    "get_materials_dataframe",
    "LinearPosition",
    "AngularPosition",
    "LinearLoadPosition",
    "AngularLoadPosition",
    "LinearPositionsTable",
    "AngularPositionsTable",
    "get_wire_tolerance",
    "get_RMa_range",
    "GoodmanData",
    "GoodmanAnalyzer",
    "Goodman",
    "generate_goodman_diagram",
    "SpringPDFReport",
    "SupplierSpecSheet",
    "spring_geometry_to_dict",
    "spring_geometry_to_json",
    "spring_geometry_to_csv",
    "spring_geometry_profile",
    "spring_geometry_to_step",
    "ureg",
    "interactive_backend",
    "export_model",
    "import_model",
    "model_to_json",
    "model_from_json",
    "save_model",
    "load_model",
    "CompressionSpringInverseDesigner",
    "ConicalCompressionSpringInverseDesigner",
    "ConicalCurveCompressionSpringInverseDesigner",
    "__version__",
]
