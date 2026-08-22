# springcalc

A Python library for **spring calculations**: compression, extension, and
torsion springs. Includes material data, wire characteristic calculations,
and fatigue analysis via the Goodman diagram.

Visit the library in my guithub https://github.com/ErnestoAvedillo/springcalc  and clone my repository using:

    git clone git@github.com:ErnestoAvedillo/springcalc.git

## Structure

```
src/springcalc/         Library package
├── lineal/             Calculation engine (compression, extension, torsion, Goodman,
│                       3D visualization, progressive-compression animation)
├── pymodels/           Data models (pydantic): material, units, wire, positions
├── material/           Material and tolerance tables (CSV, package data)
├── regresiones/        Fitted models loaded at runtime
│   └── factor_f/       Shigley's factor f (plain JSON coefficients + loader)
├── plots/              Goodman diagram generation
├── report/             PDF report generation (SpringPDFReport)
└── inverse_calc/       Inverse design: find spring geometry from a target rate or curve

tests/                  Tests (pytest)
scripts/regresiones/    Training scripts that regenerate the JSON coefficients (not runtime)
docs/                   Reference material (spreadsheet, figures)
```

## Installation

I recommend to use uv to install the library (https://docs.astral.sh/uv/):

```bash
# create the environment and install dependencies
uv init
uv add springcalc
```

## Usage

```python
from springcalc import Material, CompressionSpring

material = Material(material_name="SH")
spring = CompressionSpring(material=material, wire_diameter=1.0)
spring.set_geometry(outer_diameter=10.0, free_length=50.0, nr_coils=10)
properties = compression_spring.get_spring_data()
    for key, value in properties.items():
        print(f"{key}: {value}")                       )
```

Generating a PDF report (spring data, load/travel/diameter curves, and the
Goodman fatigue diagram):

```python
from springcalc.report import SpringPDFReport

report = SpringPDFReport(spring, title="Spring XYZ-123")
report.build("spring_report.pdf")
```

## API Reference

- [Material data](#material-data) — `Material`, `get_available_materials()`
- [Wire characteristics](#wire-characteristics) — `WireCharacteristics`
- [Compression springs](#compression-springs) — `CompressionSpring`
- [Extension springs](#extension-springs) — `ExtensionSpring`
- [Torsion springs](#torsion-springs) — `TorsionSpring`
- [Fatigue analysis (Goodman diagram)](#fatigue-analysis-goodman-diagram) — `GoodmanData`, `GoodmanAnalyzer`, `Goodman`, `generate_goodman_diagram()`
- [Position tables](#position-tables) — `LinearPositionsTable`, `AngularPositionsTable`
- [PDF reports](#pdf-reports) — `SpringPDFReport`
- [Advanced: variable-geometry springs](#advanced-variable-geometry-springs) — `VariableLinealSpring`, `CompressionSpringGeneral`
- [Animating progressive compression](#animating-progressive-compression) — `CompressionAnimator`
- [Inverse design](#inverse-design) — `CompressionSpringInverseDesigner`, `ConicalCompressionSpringInverseDesigner`, `ConicalCurveCompressionSpringInverseDesigner`

All physical quantities are [`pint`](https://pint.readthedocs.io/) `Quantity`
objects (a number with a unit, e.g. `20.0 millimeter`). Plain numbers passed
into a field are usually interpreted in that field's default unit (mm, N,
degrees...), but a few methods require an explicit `Quantity` — those are
called out below. Import the shared unit registry with:

```python
from springcalc.pymodels.units import ureg

length = 20 * ureg.mm
angle = 90 * ureg.degree
```

### Material data

`Material` (`springcalc.pymodels.material.Material`) is a pydantic model that
looks up a named material's mechanical properties from `material/materials.csv`
and auto-fills any field you don't pass explicitly.

| Member | Description |
|---|---|
| `Material(material_name, young_modulus=None, shear_modulus=None, elastic_limit_factor=None, poisson_coef=None, RMa_file=None)` | Construct a material by name; unset fields are auto-filled from `materials.csv`. |
| `.young_modulus` | Young's modulus E, as a `Quantity` in MPa. |
| `.shear_modulus` | Shear modulus G, as a `Quantity` in MPa. |
| `.elastic_limit_factor` | Dimensionless factor used to derive the fatigue limit. |
| `.poisson_coef` | Poisson's ratio (dimensionless). |
| `.RMa_file` | Name of the CSV (in `material/`) with tensile-strength ranges by wire diameter. |
| `get_available_materials()` | Module-level function: list of valid `material_name` values. |
| `Material.list_available_materials()` | Classmethod: list of valid `material_name` values (same as `get_available_materials()`). |
| `Material.material_exists(material_name)` | Classmethod: whether a material is already registered. |
| `Material.create_material(material_name, young_modulus, shear_modulus, elastic_limit_factor, poisson_coef, description="", RMa_file=None, RMa_data=None, overwrite=False)` | Classmethod: register a new material in `materials.csv` (and optionally its `RMa_file` table) and return it. Raises `ValueError` if the name already exists, unless `overwrite=True`. |

```python
from springcalc import Material, get_available_materials

print(get_available_materials())
# ['SL', 'SM', 'DM', 'SH', 'DH', 'TDC', 'TDCrV', ...]

material = Material(material_name="SH")
print(material.young_modulus)    # 206000.0 megapascal
print(material.shear_modulus)    # 81500.0 megapascal
print(material.poisson_coef)     # 0.3 dimensionless
```

Adding a new material — `young_modulus`/`shear_modulus` accept a unit string, a
plain number (assumed MPa), or a `Quantity`; `elastic_limit_factor` and
`poisson_coef` are dimensionless. Once created, the material is loadable by
name like any built-in one:

```python
from springcalc import Material

custom = Material.create_material(
    material_name="CustomSteel",
    young_modulus="210000 MPa",
    shear_modulus=81000,
    elastic_limit_factor=0.5,
    poisson_coef=0.3,
    description="Custom steel for a special order",
    RMa_data=[(1.0, 1600, 1800), (2.0, 1500, 1700)],  # (diameter mm, RMa_min, RMa_max)
)

# From now on it behaves like any other material
same_material = Material(material_name="CustomSteel")
```

### Wire characteristics

`WireCharacteristics` (`springcalc.pymodels.wire_characteristics.WireCharacteristics`)
is the base class of every spring type. Given a material and a wire diameter,
it looks up the wire diameter tolerance and the tensile-strength range (RMa)
for that diameter.

| Member | Description |
|---|---|
| `WireCharacteristics(material, wire_diameter)` | `wire_diameter` may be a plain number (interpreted in mm) or a `Quantity`. |
| `.diameter_tolerance` | Manufacturing tolerance for this diameter (from `DIAMETRO_TOLERANCIAS.csv`). |
| `.RMa_min` / `.RMa_max` | Tensile-strength range for this material and diameter, from `material.RMa_file`. |
| `.set_material(material, wire_diameter)` | Re-assign the material/diameter and refresh the derived fields. |

```python
from springcalc import Material
from springcalc.pymodels.wire_characteristics import WireCharacteristics

material = Material(material_name="SH")
wire = WireCharacteristics(material=material, wire_diameter=2.0)
print(wire.diameter_tolerance)          # 0.025
print(wire.RMa_min, wire.RMa_max)       # 1980.0 2200.0
```

### Compression springs

`CompressionSpring` (`springcalc.lineal.compresion.CompressionSpring`, also
exported as `springcalc.CompressionSpring`) is the main entry point for
helical compression springs. It extends `LinealSpring`
(`springcalc.lineal.lineal.LinealSpring`), the shared calculation engine also
used by `ExtensionSpring`.

| Method | Description |
|---|---|
| `CompressionSpring(material, wire_diameter, **data)` | Create the spring. |
| `.set_geometry(mean_diameter=None, outer_diameter=None, inner_diameter=None, nr_coils=None, pitch=None, free_length=None, type_of_end=None, type_conforming=None)` | Set the full geometry in one call: exactly one diameter and exactly two of `nr_coils`/`pitch`/`free_length`. `type_of_end` (one of `constants.COMPRESSION_SPRING_END_TYPES`, e.g. `"open_ground"`) and `type_conforming` (one of `constants.FORMING_TYPES`, e.g. `"cold_formed"`) are optional and, if given, override the spring's defaults before the active-coil count is computed. Equivalent to calling `.set_diameter()` followed by `.calculate_spring_properties()`. Returns `.get_spring_data()`. |
| `.set_diameter(mean_diameter=None, outer_diameter=None, inner_diameter=None)` | Set exactly one of the three diameters; derives the others and the spring index/Wahl factor. |
| `.calculate_spring_properties(nr_coils=None, pitch=None, free_length=None)` | Provide exactly two of the three; computes coils, active coils, Wahl factor, spring constant, solid length, and wire length. |
| `.add_load_position(length)` | Record the load/stress/outer-diameter at a given compressed length, for the load-position table and fatigue analysis. |
| `.empty_tables()` | Clear the recorded load positions. |
| `.get_spring_data()` | `dict` with all computed spring properties (material, diameters, constants, Wahl factor, etc.). |
| `.get_data_positions()` / `.get_data_travels()` | List of `LinearLoadPosition` recorded via `add_load_position`. |
| `.get_forces_vs_position_graph(show=False)` | Load vs. absolute position curve; returns a base64-encoded PNG. |
| `.get_forces_vs_travel_graph(show=False)` | Load vs. travel (compression from free length) curve; returns a base64 PNG. |
| `.get_diameter_graph(show=False)` | Outer diameter vs. position curve; returns a base64 PNG. |
| `.get_diameter_vs_position_graph(show=False)` | Outer diameter curve plus a to-scale cross-section diagram; returns a base64 PNG. |
| `.get_3d_plot(num_points=200, show=False, isometric=True)` | Renders the coiled wire geometry in 3D (constant mean diameter and pitch); returns a base64 PNG. Defaults to an orthographic isometric view, matching how spring drawings are conventionally presented. |
| `.create_goodman_diagram(show=False)` | Runs the fatigue (Goodman) analysis from the recorded positions; returns `{"image", "analysis", "stresses"}` or `{"error", "traceback"}`. |
| `.get_stress_max()` / `.get_stress_min()` | Max/min stress across recorded positions (raises if none are recorded). |
| `.get_load_max()` / `.get_load_min()` | Max/min load across recorded positions. |
| `.calculate_solid_length()` | Coils-stacked solid (fully compressed) length. |
| `.calculate_wire_length()` | Total wire length needed to wind the spring. |
| `.set_number_cycles(number_cycles)` | Design life, in cycles, used by the fatigue analysis (default 1e6). |

```python
from springcalc import Material, CompressionSpring

material = Material(material_name="SL")
spring = CompressionSpring(material=material, wire_diameter=2.5)
spring.set_geometry(outer_diameter=30, pitch=20, free_length=100)  # mm

# Equivalent to calling separately:
#   spring.set_diameter(outer_diameter=30)
#   spring.calculate_spring_properties(pitch=20, free_length=100)

for length_mm in [30, 40, 50, 60, 70, 80, 90, 100]:
    spring.add_load_position(length=length_mm)

data = spring.get_spring_data()
print(data["spring_constant"])          # ~7.09 N / mm
print(data["wahl_factor_category"])     # 'green' -> C=11 is in a normal manufacturable range

# Graphs (base64 PNGs, ready to embed in HTML or a PDF)
load_vs_position_png = spring.get_forces_vs_position_graph()
diameter_png = spring.get_diameter_vs_position_graph()

# Fatigue analysis from the recorded positions
result = spring.create_goodman_diagram()
print(result["analysis"]["safety_factor"])
```

### Extension springs

`ExtensionSpring` (`springcalc.lineal.extension.ExtensionSpring`, also exported
as `springcalc.ExtensionSpring`) models helical extension springs. It extends
`LinealSpring` directly (not `CompressionSpring`) and its diameter/length
setters require explicit `Quantity` values rather than plain numbers.

| Method | Description |
|---|---|
| `ExtensionSpring(material, wire_diameter, **data)` | Create the spring. |
| `.set_diameter(outer_diameter=None, inner_diameter=None, mean_diameter=None)` | Set exactly one diameter. **Must be a `Quantity`** (e.g. `15 * ureg.mm`), not a plain number. |
| `.calculate_spring_properties(nr_coils=None, pitch=None, free_length=None)` | Provide exactly two of the three; computes active coils, spring constant, and wire length. |
| `.add_load_position(length)` | Record the load/stress/outer-diameter at a given extended length (must be ≥ free length). |
| `.calculate_positions_table(step: list)` | Convenience: call `add_load_position` for each length in `step`. |
| `.empty_tables()` | Clear the recorded load positions. |
| `.get_spring_data()` | `dict` with all computed spring properties. |
| `.get_data_positions()` / `.get_data_travels()` | List of recorded `LinearLoadPosition`. |
| `.get_forces_vs_position_graph(show=False)` / `.get_forces_vs_travel_graph(show=False)` | Load curves; return base64 PNGs. |
| `.get_diameter_graph()` / `.get_diameter_vs_position_graph()` | Diameter curves (no `show` parameter on this class); return base64 PNGs. |
| `.create_goodman_diagram()` | Fatigue analysis from the recorded positions; returns `{"image", "analysis", "stresses"}` or `{"error", "traceback"}` (no `show` parameter). |
| `.get_stress_max()` / `.get_stress_min()` / `.get_load_max()` / `.get_load_min()` | Extremes across recorded positions. |
| `.set_number_cycles(number_cycles)` | Design life in cycles for the fatigue analysis. |
| `.set_initial_stress(initial_stress)` | Set the spring's initial tension (pre-load) stress. |

```python
from springcalc import Material, ExtensionSpring
from springcalc.pymodels.units import ureg

material = Material(material_name="SH")
spring = ExtensionSpring(material=material, wire_diameter=1.5)
spring.set_diameter(outer_diameter=15 * ureg.mm)   # note: needs a Quantity, unlike CompressionSpring
spring.calculate_spring_properties(nr_coils=10, free_length=60)

spring.calculate_positions_table([65, 70, 75, 80])  # extend beyond the free length

data = spring.get_spring_data()
print(data["spring_constant"])   # ~2.33 N / mm

result = spring.create_goodman_diagram()
print(result["analysis"]["safety_factor"])
```

### Torsion springs

`TorsionSpring` (`springcalc.lineal.torsion.TorsionSpring`, also exported as
`springcalc.TorsionSpring`) models helical torsion springs, tracking angular
position/travel and torque instead of linear load. It extends
`WireCharacteristics` directly and has no Goodman/fatigue integration.

| Method | Description |
|---|---|
| `TorsionSpring(material, wire_diameter, **data)` | Create the spring. |
| `.set_geometry(mean_diameter, nr_coils, pitch, free_angle, fixed_leg_radius, mobile_leg_radius)` | One-call setup: sets geometry and computes every derived property. Returns `.get_spring_properties()`. |
| `.calculate_spring_properties()` | Re-run the derived-property calculations after changing an input. |
| `.add_position(angle_travel=None, torque=None)` | Record a working position from either an angular travel or a torque (exactly one). |
| `.clean_positions()` | Clear the recorded positions. |
| `.get_positions()` / `.get_data_positions()` / `.get_data_travels()` | List of recorded `AngularLoadPosition`. |
| `.get_spring_properties()` | `dict` with all computed properties (diameters, angles, leg lengths, spring constant, Wahl factor, etc.). |
| `.calculate_torque(rotation_angle)` | Torque required for a given rotation angle. |
| `.calculate_stress(torque)` | Max wire stress for a given torque. |
| `.get_forces_vs_position_graph(show=False)` / `.get_forces_vs_travel_graph(show=False)` | Torque vs. angular position/travel curves; return base64 PNGs. |
| `.get_diameter_vs_position_graph(show=False)` | Outer diameter curve plus a cross-section diagram; returns a base64 PNG. |
| `.set_number_cycles(number_cycles)` / `.set_shot_peening(shot_peening)` | Fatigue-related inputs (stored but not yet used by a Goodman analysis for this class). |

```python
from springcalc import Material, TorsionSpring
from springcalc.pymodels.units import ureg

material = Material(material_name="SH")
spring = TorsionSpring(material=material, wire_diameter=1.0)
spring.set_geometry(
    mean_diameter=10 * ureg.mm,
    nr_coils=8,
    pitch=1.2 * ureg.mm,
    free_angle=180 * ureg.degree,
    fixed_leg_radius=15 * ureg.mm,
    mobile_leg_radius=15 * ureg.mm,
)
print(spring.spring_constant)   # ~35 mm*N/rad

spring.add_position(angle_travel=30 * ureg.degree)
for position in spring.get_positions():
    print(position)
```

### Fatigue analysis (Goodman diagram)

`springcalc.lineal.goodman` implements the modified-Goodman fatigue check
(Shigley, ch. 6/10) for spring wire in torsion, axial, or flexural loading.
`CompressionSpring.create_goodman_diagram()` and
`ExtensionSpring.create_goodman_diagram()` use this internally, but it can
also be used directly.

| Member | Description |
|---|---|
| `GoodmanData(material, diameter, load_type="axial", cycles=1e6)` | Pydantic input model. `load_type` is `"axial"`, `"torsion"`, or `"flexion"`. |
| `GoodmanAnalyzer(data, shot_peening=False)` | Computes the Marin correction factors, the corrected endurance limit `Sse` and fatigue strength `Ssf`. |
| `.calculate_safety_factor(sigma_max, sigma_min)` | Modified-Goodman safety factor for an operating stress cycle. |
| `.get_analysis_summary(sigma_max, sigma_min)` | `dict` with the correction factors, strengths, operating point, and safety factor. |
| `.plot_diagram(sigma_max, sigma_min, show_plot=True)` | Returns a matplotlib `Figure` with the Goodman envelope and the operating point plotted. |
| `.get_diagram_image(sigma_max, sigma_min)` | Same diagram, returned as a base64 PNG string. |
| `Goodman(material, diameter, load_type="axial", number_cycles=1e6, shot_peening=False)` | Backwards-compatible wrapper around `GoodmanAnalyzer` with the same methods (`plot_goodman_graph`, `get_goodman_graph`, etc.). |
| `generate_goodman_diagram(spring, initial_length, final_length, shot_peening=False, number_cycles=1e6)` | Module function in `springcalc.plots`: derives max/min load and stress for a spring compressed between two lengths, and returns the same `{"image", "analysis", "stresses"}` dict. **`initial_length`/`final_length` must be `Quantity` values** (they're subtracted directly from `spring.free_length`). |

```python
from springcalc import Material, GoodmanData, GoodmanAnalyzer

material = Material(material_name="DH")
data = GoodmanData(material=material, diameter=1.0, load_type="torsion", cycles=1e5)
analyzer = GoodmanAnalyzer(data)

sigma_max, sigma_min = 400, 100   # MPa
print(analyzer.calculate_safety_factor(sigma_max, sigma_min))
summary = analyzer.get_analysis_summary(sigma_max, sigma_min)
print(summary["strengths"]["Se_MPa"], summary["strengths"]["Sf_MPa"])
```

Using the standalone helper directly on a `CompressionSpring` instance (see
the [Compression springs](#compression-springs) example for how `spring` was
built), instead of calling `spring.create_goodman_diagram()`:

```python
from springcalc.plots import generate_goodman_diagram
from springcalc.pymodels.units import ureg

result = generate_goodman_diagram(spring, initial_length=100 * ureg.mm, final_length=40 * ureg.mm)
print(result["analysis"]["safety_factor"])
```

### Position tables

Every spring stores its recorded working positions in a pydantic list model
under `.positions`, populated via each spring's `add_load_position`/
`add_position` method rather than built by hand — but they're documented here
since `.get_data_positions()` returns their contents.

| Model | Used by | Fields |
|---|---|---|
| `LinearLoadPosition` (`springcalc.pymodels.positions`) | `CompressionSpring`, `ExtensionSpring` | `.position`, `.travel`, `.load` (N), `.stress` (MPa), `.outer_diameter`, `.inner_diameter` — all `Quantity`. |
| `LinearPositionsTable` | same | `.positions`: `list[LinearLoadPosition]`. `.add_load_position(...)`, `.clear_table()`. |
| `AngularLoadPosition` | `TorsionSpring` | Same fields as above, with `.position`/`.travel` in degrees and `.load` in N·mm (torque). |
| `AngularPositionsTable` | `TorsionSpring` | `.positions`: `list[AngularLoadPosition]`. `.add_load_position(...)`, `.clear_table()`. |

```python
# `spring` is any CompressionSpring/ExtensionSpring/TorsionSpring instance
# with load positions already recorded via add_load_position/add_position.
for position in spring.get_data_positions():
    print(position.position, position.load, position.stress)
```

### PDF reports

`SpringPDFReport` (`springcalc.report.SpringPDFReport`, also exported as
`springcalc.SpringPDFReport`) renders a `CompressionSpring`'s data, a 3D
isometric view, curves, and the Goodman diagram into a printable PDF using
`reportlab`.

| Method | Description |
|---|---|
| `SpringPDFReport(spring, title=None)` | Wrap a `CompressionSpring` (with load positions already added via `add_load_position`, for the fullest report). |
| `.build(output_path)` | Render the report and write it to `output_path`. Returns the path. Includes the spring data table, a 3D isometric view (from `spring.get_3d_plot()`), the load/geometry curves, the load-position table, and the Goodman diagram. Degrades gracefully (with a placeholder message) if a graph or the Goodman analysis can't be generated, e.g. no load positions recorded yet. |

```python
from springcalc.report import SpringPDFReport

# `spring` is a CompressionSpring instance (see the Compression springs example)
report = SpringPDFReport(spring, title="Spring XYZ-123")
report.build("spring_report.pdf")
```

### Advanced: variable-geometry springs

`VariableLinealSpring` and `CompressionSpringGeneral`
(`springcalc.lineal.generic_lineal` / `springcalc.lineal.generic_compression`)
model compression springs whose mean diameter and/or pitch vary along their
length (e.g. conical or barrel springs), by numerically integrating along the
helix instead of using the constant-geometry closed-form equations. They are
not exported from the top-level `springcalc` package — import them from their
modules directly. For a constant-diameter, constant-pitch spring they agree
with the closed-form `CompressionSpring` results.

| Method | Description |
|---|---|
| `CompressionSpringGeneral(material, wire_diameter, **data)` | Create the spring. Set `.mean_diameter_init`, `.pitch_constant`, and `.free_length` for a constant-geometry spring, or... |
| `.establish_geometrical_function(func_D, func_p)` | ...inject custom functions `h -> mean_diameter` and `h -> pitch` (both `Quantity -> Quantity`) for a true variable-geometry spring. |
| `.set_geometry(func_D, func_p, free_length=None, type_of_end=None, type_conforming=None)` | One-call setup: calls `.establish_geometrical_function(func_D, func_p)`, sets `.free_length`, and optionally `type_of_end` (one of `constants.COMPRESSION_SPRING_END_TYPES`, e.g. `"open_ground"`) and `type_conforming` (one of `constants.FORMING_TYPES`, e.g. `"cold_formed"`) — both default to the spring's current value when omitted. |
| `.calculate_theta_max()` | Total helix rotation angle (rad) needed to reach `free_length`; also updates `.nr_coils`. |
| `.calculate_active_coils()` | Number of active coils (`.nr_active_coils`), discounting the ground/squared end coils that don't deform, based on `type_of_end`/`type_conforming` — same formula as `CompressionSpring`. |
| `.calculate_spring_constant(num_points=500)` | Equivalent stiffness, integrating the local flexibility along the helix over the active coils only (end coils excluded per `.calculate_active_coils()`). |
| `.calculate_wire_length(num_points=500)` | Total wire length, integrating the 3D arc length along the helix. |
| `.calculate_solid_length()` | Solid (fully compressed) length, accounting for coil telescoping/nesting when the diameter varies enough. |
| `.get_3d_plot(num_points=500, show=False, isometric=True)` | Renders the helix centerline in 3D, following the actual `f_mean_diameter`/`f_pitch` functions (so variable geometries show up as a non-uniform helix); returns a base64 PNG. Defaults to an orthographic isometric view. |
| `.simulate_progressive_compression(max_deflection, steps=100, num_points=500, capture_geometry=False)` | Step-by-step compression simulation that detects coil-to-coil (oblique) contact; returns `(deflection, force, instantaneous_stiffness)` arrays. With `capture_geometry=True`, also returns a 4th value: `{"thetas", "z_history"}`, the instantaneous coil shape at every step (used by `CompressionAnimator`, see [below](#animating-progressive-compression)). |

```python
from springcalc import Material
from springcalc.pymodels.units import ureg
from springcalc.lineal.generic_compression import CompressionSpringGeneral

material = Material(material_name="SH")
spring = CompressionSpringGeneral(material=material, wire_diameter=2.0)
spring.set_geometry(
    func_D=lambda h: 20 * ureg.mm,
    func_p=lambda h: 6 * ureg.mm,
    free_length=60 * ureg.mm,
    type_of_end="open_ground",     # optional; this is the default
    type_conforming="cold_formed", # optional; this is the default
)

spring.calculate_theta_max()
print(spring.calculate_spring_constant())   # matches G*d^4/(8*D^3*n_active) for constant geometry
print(spring.nr_active_coils)               # 7.7 (10 total coils minus the non-deforming ground ends)

deflection, force, stiffness = spring.simulate_progressive_compression(max_deflection=20 * ureg.mm, steps=20)
print(force[-1])   # ~52.92 N
```

### Animating progressive compression

`CompressionAnimator` (`springcalc.lineal.animation.CompressionAnimator`) renders
`simulate_progressive_compression`'s result as an animated GIF of the coils
closing up under load, reusing the same helix geometry as `get_3d_plot`. It
takes any `VariableLinealSpring` (e.g. `CompressionSpringGeneral`). Saving is
done with matplotlib's `PillowWriter`, so no extra system dependency (like
`ffmpeg`) is required.

| Method | Description |
|---|---|
| `CompressionAnimator(spring)` | Wrap a `CompressionSpringGeneral` (or other `VariableLinealSpring`) instance. |
| `.create_gif(max_deflection, output_path="compression.gif", steps=60, num_points=300, fps=12, isometric=True)` | Runs `simulate_progressive_compression(capture_geometry=True)` internally and writes the resulting animation to `output_path`. Returns `output_path`. Axis limits are fixed from the free-state geometry so the camera doesn't jump between frames. |

```python
from springcalc import Material
from springcalc.pymodels.units import ureg
from springcalc.lineal.generic_compression import CompressionSpringGeneral
from springcalc.lineal.animation import CompressionAnimator

material = Material(material_name="SH")
spring = CompressionSpringGeneral(material=material, wire_diameter=2.0)
spring.set_geometry(func_D=lambda h: 20 * ureg.mm, func_p=lambda h: 6 * ureg.mm, free_length=60 * ureg.mm)

animator = CompressionAnimator(spring)
animator.create_gif(max_deflection=25 * ureg.mm, output_path="compression.gif")
```

### Inverse design

`springcalc.inverse_calc` (not exported from the top-level `springcalc`
package — import from its modules directly) works backwards from a target
spring *rate* or a target force-displacement *curve* to a buildable geometry,
instead of computing properties from geometry you already chose. All three
designers search the standard wire diameter series
(`get_standard_wire_diameters()`), score candidates against a target fatigue
safety factor (via `GoodmanAnalyzer`), and rebuild/verify the winning design
with the library's normal spring classes before returning it — so the
returned `.spring` behaves exactly like a spring you built by hand.

#### Cylindrical spring from two (or more) rate points

`CompressionSpringInverseDesigner`
(`springcalc.inverse_calc.lineal_comp_inv`) finds the wire diameter and mean
diameter of a standard (constant-diameter, constant-pitch) `CompressionSpring`
that reproduces a target rate while landing the safety factor as close as
possible to a target value. The number of coils is always solved so the rate
matches exactly; for each standard wire diameter, the mean diameter that hits
the safety-factor target is found by 1-D root finding.

| Member | Description |
|---|---|
| `Requirements(material, security_factor, length1=None, length2=None, force1=None, force2=None, csv_path=None)` | Design brief. Give either `length1`/`length2`/`force1`/`force2` (two exact length/force points) or `csv_path` (a CSV with `length`,`force` columns, mm/N, fit by least squares) — not both. Also used by the conical designers below. |
| `CompressionSpringInverseDesigner(requirements, type_of_end=..., type_conforming=..., spring_index_bounds=(4.5, 12.0), wire_diameter_bounds=(0.0, inf), min_active_coils=2.0, number_cycles=1_000_000, shot_peening=False)` | Construct the designer. |
| `.design()` | Runs the search and returns an `InverseCompressionDesign`. |
| `InverseCompressionDesign.spring` | The winning, fully built `CompressionSpring` (load positions already added at the two extreme lengths). |
| `.wire_diameter` / `.mean_diameter` / `.free_length` / `.nr_coils` / `.spring_index` / `.spring_constant` | Resulting geometry, as `Quantity`/`float`. |
| `.safety_factor` / `.safety_factor_target` / `.safety_factor_error` | Achieved vs. target safety factor (`error` is positive when more conservative than requested). |
| `.candidates` | `list[CandidateDesign]`, one per standard wire diameter tried (valid or not), for traceability. |

```python
from springcalc.inverse_calc.lineal_comp_inv import CompressionSpringInverseDesigner, Requirements
from springcalc.pymodels.material import Material

material = Material(material_name="SL")
requirements = Requirements(
    material=material, security_factor=1.5,
    length1=60, force1=200,   # more compressed, higher-force point
    length2=90, force2=50,    # less compressed, lower-force point
)
result = CompressionSpringInverseDesigner(requirements).design()

print(result.wire_diameter, result.mean_diameter, result.nr_coils)
print(result.safety_factor, result.safety_factor_error)
```

#### Conical (tapered) spring from two (or more) rate points

`ConicalCompressionSpringInverseDesigner`
(`springcalc.inverse_calc.conical_comp_inv`) is the same idea, but for a
linearly-tapered (diameter *and* pitch) conical spring built on
`CompressionSpringGeneral`. The rate and safety-factor equations alone leave
the taper shape (`tau_D` = D_end/D_start, `tau_p` = p_end/p_start) free, so
the extra degree of freedom is resolved by minimizing solid length — i.e.
preferring the most compact (most telescoping) taper, the usual reason to
choose a conical spring at all. For each standard wire diameter, the taper
shape is searched on a grid and locally polished (Nelder-Mead).

| Member | Description |
|---|---|
| `ConicalCompressionSpringInverseDesigner(requirements, type_of_end=..., type_conforming=..., spring_index_bounds=(4.5, 12.0), taper_ratio_bounds=(0.3, 1.0), pitch_ratio_bounds=(0.3, 3.0), wire_diameter_bounds=(0.0, inf), min_coils=2.0, shape_grid_resolution=9, number_cycles=1_000_000, shot_peening=False)` | Construct the designer with the same `Requirements` used above. |
| `.design()` | Runs the search and returns a `ConicalInverseCompressionDesign`. |
| `ConicalInverseCompressionDesign.spring` | The winning, fully built `CompressionSpringGeneral`. |
| `.diameter_start` / `.diameter_end` / `.pitch_start` / `.pitch_end` / `.free_length` / `.nr_coils` / `.solid_length` / `.spring_constant` | Resulting geometry, as `Quantity`/`float`. |
| `.safety_factor` / `.safety_factor_target` / `.safety_factor_error` | Achieved vs. target safety factor. |
| `.candidates` | `list[ShapeCandidate]`, best taper shape per standard wire diameter tried. |
| `linear_profile(start_mm, end_mm, free_length_mm)` | Module function: builds a `func_D`/`func_p`-compatible closure for a linearly-varying quantity — handy for feeding the winning geometry into `CompressionSpringGeneral.set_geometry()` directly. |

```python
from springcalc.inverse_calc.conical_comp_inv import ConicalCompressionSpringInverseDesigner
from springcalc.inverse_calc.lineal_comp_inv import Requirements
from springcalc.pymodels.material import Material

material = Material(material_name="SL")
requirements = Requirements(material=material, security_factor=1.2,
                            length1=60, force1=200, length2=90, force2=50)
result = ConicalCompressionSpringInverseDesigner(requirements).design()

print(result.diameter_start, result.diameter_end)
print(result.pitch_start, result.pitch_end)
print(result.solid_length)
```

#### Conical (tapered) spring fit to a full force-displacement curve

`ConicalCurveCompressionSpringInverseDesigner`
(`springcalc.inverse_calc.conical_curve_comp_inv`) fits a conical spring's
wire diameter, free length, and diameter/pitch taper to an entire target
force-vs-displacement curve (rather than just two points), so it can capture
progressive stiffening from coil-to-coil contact. There's no closed form for
this, so it's a regression: `scipy.optimize.differential_evolution` searches
the 6-parameter space (wire diameter, free length, D_start, D_end, p_start,
p_end), minimizing curve RMSE plus a soft safety-factor term plus a geometry
penalty, using a fast closed-form contact simulation; the winning wire
diameter is then snapped to the standard series and the rest locally
re-fit before the final design is rebuilt with the real, general
`CompressionSpringGeneral` machinery.

| Member | Description |
|---|---|
| `CompressionCurveRequirements(material, security_factor, csv_path)` | Design brief: `csv_path` is a CSV with `displacement`,`load` columns (mm, N) — `displacement` is travel from the free length, not an absolute position. |
| `load_target_curve(csv_path)` | Module function: reads and sorts the target curve; returns `(displacement, load)` numpy arrays. |
| `ConicalCurveCompressionSpringInverseDesigner(requirements, type_of_end=..., type_conforming=..., spring_index_bounds=(4.5, 12.0), wire_diameter_bounds=(0.3, 10.0), diameter_bounds=(3.0, 150.0), pitch_bounds=(0.3, 40.0), free_length_margin=(1.05, 3.0), min_coils=2.0, safety_factor_weight=1.0, penalty_weight=0.05, search_num_points=60, search_steps=60, final_num_points=500, final_steps=500, maxiter=60, popsize=15, seed=None, number_cycles=1_000_000, shot_peening=False)` | Construct the designer. `seed` makes the regression reproducible. |
| `.design()` | Runs the regression and returns a `ConicalCurveInverseDesign`. |
| `ConicalCurveInverseDesign.spring` | The winning, fully built `CompressionSpringGeneral`. |
| `.diameter_start` / `.diameter_end` / `.pitch_start` / `.pitch_end` / `.free_length` / `.nr_coils` / `.solid_length` | Resulting geometry, as `Quantity`/`float`. |
| `.safety_factor` / `.safety_factor_target` / `.safety_factor_error` | Achieved vs. target safety factor. |
| `.curve_rmse` / `.curve_rmse_relative` | Fit quality: RMSE between the simulated and target curves (`Quantity` in N), and that RMSE relative to the target curve's typical load magnitude. |
| `.target_displacement` / `.target_load` / `.simulated_displacement` / `.simulated_load` | Both curves as numpy arrays, ready to plot against each other. |

```python
from springcalc.inverse_calc.conical_curve_comp_inv import (
    CompressionCurveRequirements, ConicalCurveCompressionSpringInverseDesigner,
)
from springcalc.pymodels.material import Material

material = Material(material_name="SL")
requirements = CompressionCurveRequirements(
    material=material, security_factor=1.3, csv_path="target_curve.csv",  # displacement,load columns (mm, N)
)
result = ConicalCurveCompressionSpringInverseDesigner(requirements, seed=0).design()

print(result.curve_rmse, result.curve_rmse_relative)
print(result.diameter_start, result.diameter_end, result.pitch_start, result.pitch_end)
```

## Tests

```bash
uv run pytest
```

## Retraining the regression models

The included JSON coefficient files are already fitted. To regenerate them:

```bash
uv run python scripts/regresiones/factor_f/factor_f.py
```
