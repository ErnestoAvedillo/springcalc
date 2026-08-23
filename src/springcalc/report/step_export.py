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


def spring_geometry_to_step(design, output_path: str, num_points: int = 300) -> str:
    """Export the spring's coil wire as a STEP file. `design` is a spring
    (`CompressionSpringGeneral`) or a design result carrying one as
    `design.spring`. Returns `output_path`.

    The centerline reuses `spring.get_h_theta_development()` and
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

    thetas, zs = spring.get_h_theta_development(num_points)
    diameters_mm = np.array([spring.f_mean_diameter(z * ureg.mm).to("mm").magnitude for z in zs])
    radii_mm = diameters_mm / 2.0
    xs = radii_mm * np.cos(thetas)
    ys = radii_mm * np.sin(thetas)
    points = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))

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

    compound = cq.Compound.makeCompound(pieces)
    cq.exporters.export(cq.Workplane(obj=compound), output_path)
    return output_path
