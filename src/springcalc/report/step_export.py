"""STEP (ISO 10303) 3D export of a spring's coil geometry, for colleagues
doing mechanical CAD who need to check fit/interference in an assembly.

Requires the optional 'step' extra (`pip install springcalc[step]`, or
`uv add --optional step cadquery` in a springcalc checkout) -- OpenCASCADE
bindings are a large binary dependency the rest of the library doesn't
need, so cadquery stays out of the base install and is imported lazily
here.

Why this doesn't sweep a profile along a spline path (the "obvious" CAD
approach): a spring's centerline turns through many full rotations in a
short axial span, and both a generic BSpline fit through sampled helix
points and a sweep with a Frenet-frame path through it turned out to be
numerically unreliable for that -- verified empirically (not just in
theory) against this library's own get_h_theta_development output: at
different point counts the swept "solid"'s volume came back as roughly
5x, 20x, or even negative multiples of the wire's actual volume
(pi * wire_radius**2 * wire_length), i.e. silently self-intersecting
garbage that LOOKS like a valid solid (export succeeds, no exception)
but isn't. Chaining short, straight cylinder segments between consecutive
centerline points -- with a sphere at each joint to fill the wedge a bend
leaves on a polyline-of-cylinders' outer edge -- has no frame to lose
track of, so it can't accumulate that failure mode; each segment's volume
checks out against the analytic expectation to within discretization
error (see test in the module's development notes).
"""
import numpy as np

from ..pymodels.units import ureg

_IMPORT_ERROR_MESSAGE = (
    "STEP export needs the optional 'step' extra: pip install springcalc[step] "
    "(or `uv add --optional step cadquery` in a springcalc checkout)."
)


def _to_mm(value) -> float:
    if hasattr(value, "to"):
        return float(value.to("mm").magnitude)
    return float(value)


def spring_centerline_points(design, num_points: int = 300, length=None, steps: int = 100) -> np.ndarray:
    """(N, 3) array of the wire centerline's x, y, z coordinates in mm.

    With `length=None` this is the free-state helix. With `length` (a
    `Quantity`, or a plain number in mm) it is the spring compressed to that
    overall length: the shape comes from `simulate_progressive_compression`
    run to a deflection of `free_length - length` in `steps` steps, so coils
    that touch stay locked against each other. Each wire point keeps its
    free-state radius and only its height changes, the same convention as
    `CompressionAnimator`. `length` must lie between the solid length and the
    free length.
    """
    spring = getattr(design, "spring", design)

    if length is None:
        thetas, zs = spring.get_h_theta_development(num_points)
        radii_mm = np.array([spring.f_mean_diameter(z * ureg.mm).to("mm").magnitude for z in zs]) / 2.0
    else:
        length_mm = _to_mm(length)
        free_length_mm = _to_mm(spring.free_length)
        solid_length_mm = _to_mm(spring.calculate_solid_length())
        if not solid_length_mm <= length_mm <= free_length_mm:
            raise ValueError(
                f"length must be between the solid length ({solid_length_mm:.3f} mm) "
                f"and the free length ({free_length_mm:.3f} mm), got {length_mm:.3f} mm")
        deflection_mm = free_length_mm - length_mm
        if deflection_mm == 0:
            return spring_centerline_points(spring, num_points=num_points)
        _, _, _, geometry = spring.simulate_progressive_compression(
            max_deflection=deflection_mm * ureg.mm,
            steps=steps,
            num_points=num_points,
            capture_geometry=True,
        )
        thetas = np.asarray(geometry["thetas"], dtype=float)
        zs_free = np.asarray(geometry["z_history"][0], dtype=float)
        zs = np.asarray(geometry["z_history"][-1], dtype=float)
        radii_mm = np.array([spring.f_mean_diameter(z * ureg.mm).to("mm").magnitude for z in zs_free]) / 2.0

    thetas = np.asarray(thetas, dtype=float)
    zs = np.asarray(zs, dtype=float)
    return np.column_stack([radii_mm * np.cos(thetas), radii_mm * np.sin(thetas), zs])


def spring_geometry_to_step(design, output_path: str, num_points: int = 300, length=None,
                            include_centerline: bool = True, steps: int = 100) -> str:
    """Export the spring's coil wire as a STEP file. `design` is a spring
    (`CompressionSpringGeneral`) or a design result carrying one as
    `design.spring`. Returns `output_path`.

    `length` (a `Quantity`, or a plain number in mm) exports the spring
    compressed to that overall length instead of at its free length; see
    `spring_centerline_points` for how the compressed shape is obtained
    (`steps` is the number of simulation steps used for it). With
    `include_centerline=True` (the default) the wire's centerline is added
    to the file as a spline edge alongside the solid, e.g. to use as a
    sweep path or a reference in CAD.

    The free-state centerline reuses `spring.get_h_theta_development()` and
    `f_mean_diameter()` -- the same development `get_3d_plot()` draws --
    so the model matches what's already shown/reported, for a constant,
    linear-taper, or arbitrary diameter/pitch profile alike. `num_points`
    trades fidelity for generation time: `get_h_theta_development` solves
    one fsolve per point for a general profile, so a few hundred points
    (the default) typically takes single-digit seconds; the CAD build and
    STEP write themselves are fast (well under a second even at 1000+
    points) since they're plain cylinder/sphere primitives, not a boolean
    fuse.

    The result is a `Compound` of many small overlapping cylinder and
    sphere primitives approximating the coil -- not a single fused,
    watertight solid (boolean-fusing that many primitives is impractically
    slow -- tested at over two minutes for just 150 points). That's fine
    for what this is for: a visual/dimensional reference a colleague loads
    into a CAD assembly to check fit and interference against surrounding
    parts. It is NOT suitable for mass-property queries (volume/CoG) or as
    an input to further boolean CAD operations, since the overlapping
    primitives double-count material at every segment joint. End faces
    (ground/squared ends) aren't modeled -- same level of detail as
    `get_3d_plot()`.
    """
    try:
        import cadquery as cq
    except ImportError as exc:
        raise ImportError(_IMPORT_ERROR_MESSAGE) from exc

    spring = getattr(design, "spring", design)

    centerline = spring_centerline_points(spring, num_points=num_points, length=length, steps=steps)
    points = [tuple(p) for p in centerline.tolist()]

    wire_radius_mm = float(spring.wire_diameter.to("mm").magnitude) / 2.0

    pieces = []
    for i in range(len(points) - 1):
        start = cq.Vector(*points[i])
        end = cq.Vector(*points[i + 1])
        segment = end - start
        if segment.Length < 1e-9:
            continue
        pieces.append(cq.Solid.makeCylinder(wire_radius_mm, segment.Length, start, segment))
        pieces.append(cq.Solid.makeSphere(wire_radius_mm, start))
    pieces.append(cq.Solid.makeSphere(wire_radius_mm, cq.Vector(*points[-1])))

    if include_centerline:
        # Drop coincident consecutive points (e.g. a locked, fully closed end
        # coil) -- spline interpolation can't pass through the same point twice.
        spline_points = [cq.Vector(*points[0])]
        for p in points[1:]:
            vector = cq.Vector(*p)
            if (vector - spline_points[-1]).Length > 1e-9:
                spline_points.append(vector)
        if len(spline_points) >= 2:
            pieces.append(cq.Edge.makeSpline(spline_points))

    compound = cq.Compound.makeCompound(pieces)
    cq.exporters.export(cq.Workplane(obj=compound), output_path)
    return output_path
