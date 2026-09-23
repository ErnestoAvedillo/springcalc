from typing import Callable, Optional
from pint import Quantity
import numpy as np
from pydantic import ConfigDict, PrivateAttr, field_validator
from ..pymodels.units import ureg
from .constants import COMPRESSION_SPRING_END_TYPES
from .constants import OPEN_GROUND
from ..pymodels.wire_characteristics import WireCharacteristics
from scipy.integrate import quad, cumulative_simpson, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

# (Keep your other imports: ureg, WireCharacteristics, Material, etc.)


class VariableLinealSpring(WireCharacteristics):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # --- Base parameters of the generic spring ---
    free_length: Quantity = 0.0 * ureg.mm
    nr_coils: float = 0.0
    shot_peening: bool = False
    coating: Optional[str] = None
    spring_index: float = 0.0  # Note: for variable geometries this index varies locally.
    spring_constant: Quantity = 0.0 * ureg.N / ureg.mm

    # --- NEW: functional attributes for variable geometry ---
    # Store functions that receive the height 'h' (Quantity) and return a Quantity
    f_mean_diameter: Callable[[Quantity], Quantity] = lambda h: 0.0 * ureg.mm
    f_pitch: Callable[[Quantity], Quantity] = lambda h: 0.0 * ureg.mm

    # Keep physical bounds for quick initializations
    mean_diameter_init: Quantity = 0.0 * ureg.mm
    pitch_constant: Quantity = 0.0 * ureg.mm
    wire_length: Quantity = 0.0 * ureg.mm
    type_of_end: str = COMPRESSION_SPRING_END_TYPES[OPEN_GROUND]  # open_ground by default
    type_conforming: str = 'cold_formed'
    theta_max: float = 0.0  # Total helix rotation angle (radians)

    # Cache of the last simulate_progressive_compression() call: {"key": (...),
    # "result": (...)}. Callers (report graphs, position lookups) tend to ask
    # for the same (max_deflection, steps, num_points) repeatedly within one
    # report, and the simulation is expensive (a root-solve per point plus a
    # per-step contact scan), so avoid recomputing it every time. Invalidated
    # by set_geometry whenever the underlying geometry changes.
    _progressive_compression_cache: Optional[dict] = PrivateAttr(default=None)

    # --- Your validators stay the same (trimmed here for brevity) ---
    @field_validator('free_length', mode='before')
    @classmethod
    def validate_dimensions(cls, v):
        if v is not None:
            if isinstance(v, Quantity):
                return v.to('mm')
            return float(v) * ureg.mm
        return v

    def __init__(self, material, wire_diameter: float, **data):
        data.update({'material': material, 'wire_diameter': wire_diameter})
        super().__init__(**data)
        """
        Initialize the VariableLinealSpring with a material and wire diameter.
        The geometry functions (f_mean_diameter and f_pitch) can be set later using set_geometry or establish_geometrical_function.
        """
        # By default, if no custom functions are given, initialize as a constant linear spring
        if 'f_mean_diameter' not in data:
            self.f_mean_diameter = lambda h: self.mean_diameter_init
        if 'f_pitch' not in data:
            self.f_pitch = lambda h: self.pitch_constant

    def set_geometry(self,
                     f_mean_diameter: Callable[[Quantity], Quantity],
                     f_pitch: Callable[[Quantity], Quantity],
                     free_length: Quantity,
                     type_of_end: Optional[str] = None,
                     ):
        """Sets the initial geometry for a variable-diameter, variable-pitch spring.
        Parameters:
            f_mean_diameter: function that takes height (h) and returns mean diameter at that height
            f_pitch: function that takes height (h) and returns pitch at that height.
            free_length: optional free length of the spring
            type_of_end: optional end type (e.g., open, closed, ground)"""
        self.establish_geometrical_function(f_mean_diameter, f_pitch)
        self.free_length = free_length.to('mm')
        self.free_length = free_length if free_length is not None else self.free_length
        if type_of_end is not None:
            self.type_of_end = type_of_end

        # Invalidate everything derived from the previous geometry, so the
        # next calculation recomputes against f_mean_diameter/f_pitch/free_length
        # instead of silently reusing stale numbers from a prior call (e.g.
        # if this spring instance is reconfigured and reused).
        self.theta_max = 0.0
        self.nr_coils = 0.0
        self.spring_constant = 0.0 * ureg.N / ureg.mm
        self.wire_length = 0.0 * ureg.mm
        self._progressive_compression_cache = None

    def establish_geometrical_function(self,
                                       f_mean_diameter: Callable[[Quantity], Quantity],
                                       f_pitch: Callable[[Quantity], Quantity]):
        """Allows injecting any variable geometry into the spring"""
        self.f_mean_diameter = f_mean_diameter
        self.f_pitch = f_pitch

    def calculate_spring_index_local(self, h: Quantity) -> float:
        """The spring index now depends on which part (h) of the spring you measure"""
        D_local = self.f_mean_diameter(h)
        local_index = D_local.to('mm').magnitude / self.wire_diameter.to('mm').magnitude
        if local_index < 3:
            raise ValueError("The spring index must be greater than 3.")
        return float(local_index)

    def calculate_theta_max(self) -> float:
        """
        Determines theta_max by integrating: d_theta = (2*pi / p(h)) * dh
        from h = 0 to free_length.
        """
        H_val = self.free_length.to('mm').magnitude

        # Integrand function: extracts the magnitude in mm of the local pitch
        def integrand(h_mm):
            local_pitch = self.f_pitch(h_mm * ureg.mm).to('mm').magnitude
            if local_pitch <= 0:
                raise ValueError("The pitch at any point of h must be greater than zero.")
            return (2 * np.pi) / local_pitch
        try:
            theta_max, _ = quad(integrand, 0, H_val)
        except Exception as e:
            raise ValueError(f"Error calculating impossible or invalid theta_max: {e}")
        self.theta_max = theta_max
        # Update the total number of coils
        self.nr_coils = theta_max / (2 * np.pi)
        return self.theta_max

    def get_h_theta_development(self, num_points=500, theta_start=0.0 * ureg.rad, theta_end=None):
        """
        Solves the correspondence between the angle theta and the height h,
        over [theta_start, theta_end] (defaults to the full [0, theta_max] span).
        Returns numpy arrays in millimeters.

        theta(h) = integral_0^h (2*pi / p(h')) dh' is monotonically increasing
        (p(h) > 0 everywhere), so rather than root-solving each output point
        individually -- which re-integrates the pitch from scratch on every
        trial step of the root solve -- this builds that cumulative curve once
        on a dense h grid (Simpson's rule) and inverts it with a monotonic
        cubic (PCHIP) interpolant: a single pass over the (potentially
        expensive, pint-based) pitch function instead of thousands of
        repeated quad/root-solve evaluations, yet still accurate to a few
        parts in 1e-10 thanks to Simpson's O(dh^4) error and PCHIP tracking
        the curve's actual curvature instead of linearly chording between
        grid points.
        """
        try:
            if self.theta_max == 0:
                self.calculate_theta_max()
        except Exception as e:
            raise ValueError(f"Error calculating theta_max in development: {e}")
        if theta_end is None:
            theta_end = self.theta_max

        # theta_start/theta_end may arrive as a plain float (radians) or as a pint
        # Quantity (e.g. the default 0.0 * ureg.rad), depending on the caller.
        if isinstance(theta_start, Quantity):
            theta_start = theta_start.to('rad').magnitude
        if isinstance(theta_end, Quantity):
            theta_end = theta_end.to('rad').magnitude

        thetas = np.linspace(theta_start, theta_end, num_points)

        H_val = self.free_length.to('mm').magnitude

        # Grid the cumulative angle curve is built on, independent of num_points
        # so the interpolation stays accurate even when few output points are
        # requested; 5x the requested resolution with a floor is a comfortable margin.
        n_grid = max(2000, 5 * num_points)
        h_grid = np.linspace(0.0, H_val, n_grid)
        integrand_grid = np.array([(2 * np.pi) / self.f_pitch(h * ureg.mm).to('mm').magnitude for h in h_grid])
        theta_grid = cumulative_simpson(integrand_grid, x=h_grid, initial=0.0)

        zs = PchipInterpolator(theta_grid, h_grid)(thetas)
        return thetas, np.clip(zs, 0.0, H_val)

    def calculate_wire_length(self, num_points=500) -> Quantity:
        """Calculate the wire length by integrating the arc-length differential (ds)"""
        try:
            thetas, zs = self.get_h_theta_development(num_points)
        except Exception as e:
            raise ValueError(f"Error calculating theta_max before wire length: {e}")

        if self.theta_max == 0:
            raise ValueError("theta_max is zero after attempting to calculate it.")

        # Get diameters at each z step
        Ds = np.array([self.f_mean_diameter(z * ureg.mm).to('mm').magnitude for z in zs])
        radii = Ds / 2.0

        # 3D cartesian coordinates of the axis
        xs = radii * np.cos(thetas)
        ys = radii * np.sin(thetas)

        # Numerical derivatives with respect to theta
        dtheta = thetas[1] - thetas[0]
        dx_dtheta = np.gradient(xs, dtheta)
        dy_dtheta = np.gradient(ys, dtheta)
        dz_dtheta = np.gradient(zs, dtheta)

        # ds = sqrt( dx^2 + dy^2 + dz^2 )
        ds = np.sqrt(dx_dtheta**2 + dy_dtheta**2 + dz_dtheta**2)

        # Total integrated length
        L_mm = np.trapezoid(ds, thetas)
        self.wire_length = L_mm * ureg.mm
        return self.wire_length

    def calculate_spring_constant(self, num_points=500) -> Quantity:
        """
        Calculate the equivalent spring stiffness (K) considering the coils in series.
        1/K = integral_0^theta_max [ 8 * D(theta)^3 / (G * d^4 * 2*pi) ] dtheta
        """
        try:
            if self.theta_max == 0:
                self.calculate_theta_max()
        except Exception as e:
            raise ValueError(f"Error calculating theta_max before spring constant: {e}")
        try:
            thetas, zs = self.get_h_theta_development(num_points,
                                                      theta_start=0.0,
                                                      theta_end=self.theta_max)
        except Exception as e:
            raise ValueError(f"Error generating h-theta development: {e}")
        Ds = np.array([self.f_mean_diameter(z * ureg.mm).to('mm').magnitude for z in zs])

        d_val = self.wire_diameter.to('mm').magnitude
        # Access the material's shear modulus (ensure it's in MPa or N/mm²)
        G_val = self.material.shear_modulus.to('N/mm**2').magnitude

        # Integrand function for the local flexibility (1/dK)
        local_flexibility = (8 * Ds**3) / (G_val * (d_val**4) * 2 * np.pi)

        # Numerical integration using the trapezoidal rule
        total_flexibility = np.trapezoid(local_flexibility, thetas)

        K_val = 1.0 / total_flexibility  # N/mm
        self.spring_constant = K_val * (ureg.N / ureg.mm)
        return self.spring_constant

    def calculate_spring_properties(self, num_points: int = 500) -> dict:
        """
        Calculate all derived properties of the variable-geometry spring:
        total helix angle, coil count, wire length, spring constant, and a
        representative spring index evaluated at mid free length.
        """
        try:
            self.calculate_theta_max()
            self.calculate_wire_length(num_points=num_points)
            self.calculate_spring_constant(num_points=num_points)
            self.spring_index = self.calculate_spring_index_local(self.free_length / 2)
            return self.get_spring_data()
        except Exception as e:
            print(f"Error calculating spring properties: {e}")
            return {}

    def get_spring_data(self) -> dict:
        """Return a dictionary with the main spring data."""
        return {
            "material": self.material.material_name,
            "wire_diameter": self.wire_diameter,
            "free_length": self.free_length,
            "nr_coils": self.nr_coils,
            "spring_constant": self.spring_constant,
            "spring_index": self.spring_index,
            "wire_length": self.wire_length,
            "theta_max": self.theta_max,
            "shot_peening": self.shot_peening,
            "coating": self.coating,
        }

    def simulate_progressive_compression(self, max_deflection: Quantity,
                                         steps: int = 100,
                                         num_points: int = 500,
                                         capture_geometry: bool = False):
        """
        Simulates step-by-step compression accounting for oblique contact between coils.
        Returns the force vs. deflection curve and the evolution of the instantaneous stiffness.
        Parameters:
            max_deflection: maximum deflection to simulate (Quantity)
            steps: number of discrete deflection steps to simulate
            num_points: number of discretization points along the spring for calculations
            capture_geometry: if True, also record the instantaneous height (z) of every
                discretization point at each step (needed to animate the coil closing up).
                Off by default since it's only needed for that use case.
        Returns:
            deflection_history: numpy array of deflections (mm)
            force_history: numpy array of forces (N)
            stiffness_history: numpy array of instantaneous stiffness (N/mm)
            geometry: only when capture_geometry=True, a dict {"thetas": ndarray,
                "z_history": list of ndarrays, one per recorded step} describing the
                coil shape (in mm) at every step. Radii per point stay constant across
                steps since f_mean_diameter is evaluated at each point's free-state
                (undeformed) position, not its instantaneous height.
        """
        cache_key = (round(max_deflection.to('mm').magnitude, 9), steps, num_points, capture_geometry)
        cached = self._progressive_compression_cache
        if cached is not None and cached["key"] == cache_key:
            return cached["result"]

        # 1. Get the initial free-state (unloaded) trajectory, spanning the
        # complete winding: end coils are not excluded, so the geometry
        # captured here (and the animation built from it) renders the full
        # spring from z=0.
        try:
            if self.theta_max == 0:
                self.calculate_theta_max()
            thetas, zs_free = self.get_h_theta_development(num_points,
                                                           theta_start=0.0,
                                                           theta_end=self.theta_max)
        except Exception as e:
            raise ValueError(f"Error generating h-theta development: {e}")
        Ds = np.array([self.f_mean_diameter(z * ureg.mm).to('mm').magnitude for z in zs_free])
        radii = Ds / 2.0

        d_val = self.wire_diameter.to('mm').magnitude
        G_val = self.material.shear_modulus.to('N/mm**2').magnitude

        # We'll store the compression state at each step
        deflection_history = []
        force_history = []
        stiffness_history = []
        z_history = [] if capture_geometry else None

        # Angular step for one full turn (to compare adjacent coils)
        # Find how many discretization points equal 2*pi radians
        dtheta = thetas[1] - thetas[0]
        points_per_turn = int(round((2 * np.pi) / dtheta))

        # Initialize the accumulated force and deformation
        current_force = 0.0  # Newtons
        # Deformation of each point along the z axis
        delta_y = np.zeros_like(zs_free)

        target_max_deflection = max_deflection.to('mm').magnitude
        deflection_step = target_max_deflection / steps

        for step in range(steps + 1):
            current_deflection = step * deflection_step

            # --- OBLIQUE CONTACT DETECTION ---
            # Build a mask of active zones (1.0 = active, 0.0 = collided/locked)
            is_active = np.ones_like(thetas)

            for i in range(len(thetas) - points_per_turn):
                # Index of the adjacent upper coil
                i_sup = i + points_per_turn

                # Current vertical distance in mm
                z_inf_actual = zs_free[i] - delta_y[i]
                z_sup_actual = zs_free[i_sup] - delta_y[i_sup]
                Pz_actual = abs(z_sup_actual - z_inf_actual)

                # Radius difference (variable geometry)
                delta_R = abs(radii[i_sup] - radii[i])

                # Collision condition
                if delta_R < d_val:
                    # Oblique physical collision limit
                    Pz_limite = np.sqrt(d_val**2 - delta_R**2)
                    if Pz_actual <= Pz_limite:
                        # If they collide, both sections and everything in between are deactivated
                        is_active[i:i_sup+1] = 0.0
                else:
                    # If delta_R >= d, there is telescoping: the coils nest inside one
                    # another instead of colliding directly. Nothing above stops that
                    # section from deforming, so it's handled by the floor/top-plate
                    # contact check below instead.
                    pass

            # --- FLOOR CONTACT DETECTION ---
            # The spring rests on a fixed base at z = 0. In a variable-pitch/diameter
            # (conical) spring, an interior point can sink faster than its neighbours
            # and reach the floor before the base anchor's own neighbourhood does --
            # that point has nowhere left to go and must lock flat, same as an oblique
            # coil-on-coil collision.
            #
            # NOTE: a symmetric check against the moving top plate was tried here
            # (locking any point whose height reached or exceeded the last point's,
            # i.e. z_actual[-1]) and removed: it's unsound once an interior span has
            # already locked via coil-on-coil collision. A locked span's flexibility
            # goes to 0, so cumulative_trapezoid stops advancing across it and it
            # goes *nearly* stationary -- it does not get carried down as a rigid
            # unit by the points above it the way a real jammed stack would. Meanwhile
            # the last point's height is, by construction (see the cumulative_
            # deformation comment below), always forced down by exactly one
            # deflection_step per step regardless of any locking elsewhere. Once an
            # interior span is nearly stationary while the last point relentlessly
            # keeps dropping, the last point's height *will* eventually fall below
            # the stalled span -- a false "poking through the plate" reading, not a
            # real one. Locking those points on that basis only shrinks total_flex
            # further, which concentrates the remaining deflection increment onto an
            # ever-smaller sliver and drives the last point down even faster: a
            # feedback loop that was observed to collapse a spring to a fraction of
            # its true stiffening deflection. Properly modeling top-plate contact
            # would require carrying locked spans down as rigid bodies (translating
            # with whatever is directly above them), not just zeroing their local
            # flexibility -- a bigger change than a boundary check.
            z_actual = zs_free - delta_y
            touches_floor = z_actual <= 0.0
            # The base anchor point itself *defines* the floor position (it's always
            # exactly 0 there for the unmargined case), so comparing it against
            # itself is tautological -- exclude it.
            touches_floor[0] = False
            is_active[touches_floor] = 0.0

            # --- INSTANTANEOUS STIFFNESS CALCULATION (K_inst) ---
            # Local differential flexibility: if not active, its flexibility is 0 (infinite stiffness)
            local_flexibility = (8 * Ds**3) / (G_val * (d_val**4) * 2 * np.pi)
            active_local_flexibility = local_flexibility * is_active

            total_flex = np.trapezoid(active_local_flexibility, thetas)

            if total_flex <= 1e-9:
                # The spring has reached full lock-up (solid height)
                K_inst = float('inf')
            else:
                K_inst = 1.0 / total_flex

            # --- UPDATE FORCE AND DEFORMATION ---
            if step > 0:
                # dF = K_inst * d_deflection
                force_increment = K_inst * deflection_step if K_inst != float('inf') else 0.0
                current_force += force_increment

                # Distribute the differential deformation locally.
                # Each point's displacement is the *cumulative* flexibility
                # from the fixed base (theta_start) up to that point, not a
                # flat per-point shift: a flat shift moves every point by the
                # same amount, so the gap between two coils one turn apart
                # never changes and no collision could ever be detected. The
                # cumulative form makes every turn's gap close by the same
                # amount when diameter (and thus flexibility) is uniform,
                # matching a real spring's uniform per-turn wind-up under a
                # constant torque, while still weighting by local flexibility
                # where diameter varies.
                if K_inst != float('inf'):
                    deformation_factor = active_local_flexibility / total_flex
                    cumulative_deformation = cumulative_trapezoid(deformation_factor, thetas, initial=0.0)
                    delta_y += cumulative_deformation * deflection_step

            deflection_history.append(current_deflection)
            force_history.append(current_force)
            stiffness_history.append(K_inst)
            if capture_geometry:
                # Clamp to 0: a point can cross the floor within the same step that
                # locks it (it's deactivated for the *next* step's deformation), so
                # without this its last recorded position could dip slightly negative.
                z_history.append(np.maximum(zs_free - delta_y, 0.0))

            if K_inst == float('inf'):
                # If the spring is fully locked, end the simulation
                break

        if capture_geometry:
            result = (np.array(deflection_history) * ureg.mm,
                      np.array(force_history) * ureg.N,
                      np.array(stiffness_history) * (ureg.N / ureg.mm),
                      {"thetas": thetas, "z_history": z_history})
        else:
            result = (np.array(deflection_history) * ureg.mm,
                      np.array(force_history) * ureg.N,
                      np.array(stiffness_history) * (ureg.N / ureg.mm))

        self._progressive_compression_cache = {"key": cache_key, "result": result}
        return result
