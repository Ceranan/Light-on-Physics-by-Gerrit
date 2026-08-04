"""
Problem 9.11 -- symbolic derivation with sympy
================================================
Shows:
  1. f(v), the normalized Maxwell-Boltzmann speed distribution, and its
     normalization check.
  2. Number(v) -- the CORRECTED flux/loading integrand (1/4) N v f(v) A
     (your original version was missing the f(v) weighting).
  3. The EXACT definite integral of Number(v) from 0 to v_c -- this turns
     out to have a clean elementary closed form (no erf!), because the
     integrand is an odd power of v times a Gaussian.
  4. The small-v_c (v_c << v_p) limit, recovering the textbook result
     R_load = N*A*v_c**4 / (4*sqrt(pi)*v_p**3) as a Taylor series check.
  5. The equilibrium atom number N_eq, showing the background density N
     cancel automatically -- the symbolic version of "quirk #3".
"""

import sympy as sp

# ----------------------------------------------------------------------
# symbols  (all positive, physical quantities)
# ----------------------------------------------------------------------
v, vc, vp, N, A, vbar, sigma = sp.symbols(
    'v v_c v_p N A v_bar sigma', positive=True
)

# a = M/(2 k_B T) = 1/v_p**2   (written in terms of v_p to avoid carrying
# M, k_B, T around separately -- v_p is what actually enters the physics)
a = 1 / vp**2


# ----------------------------------------------------------------------
# (1) normalized Maxwell-Boltzmann speed distribution, f(v)
# ----------------------------------------------------------------------
def f(v):
    """Normalized 3D Maxwell-Boltzmann speed distribution (Foot eqn 8.3)."""
    return 4 * sp.pi * v**2 * (a / sp.pi) ** sp.Rational(3, 2) * sp.exp(-a * v**2)


norm_check = sp.integrate(f(v), (v, 0, sp.oo))
print("Normalization check  ∫₀^∞ f(v) dv  =", sp.simplify(norm_check))
# -> 1  (confirms f(v) is correctly normalized)


# ----------------------------------------------------------------------
# (2) flux / loading-rate integrand -- CORRECTED version of your Number(v)
#     (this is what was missing f(v) in your notebook)
# ----------------------------------------------------------------------
def Number(v):
    """Kinetic-theory flux density: (1/4) N v f(v) A."""
    return sp.Rational(1, 4) * N * v * f(v) * A


print("\nNumber(v) =")
sp.pprint(sp.simplify(Number(v)))


# ----------------------------------------------------------------------
# (3) EXACT loading rate: integrate 0 -> v_c, no approximation yet
# ----------------------------------------------------------------------
R_exact = sp.integrate(Number(v), (v, 0, vc))
R_exact = sp.simplify(R_exact)

print("\nExact R_load  (no small-v_c approximation) =")
sp.pprint(R_exact)
# Elementary closed form -- no erf, because the integrand v*f(v) ~ v^3 e^{-a v^2}
# has an odd power of v, so the substitution u = v^2 reduces it to a
# standard ∫ u e^{-au} du, which is elementary (unlike ∫ v^2 e^{-av^2} dv,
# which *does* need erf -- that's the integral you'd hit computing <v^2>
# type quantities, but it never appears here).


# ----------------------------------------------------------------------
# (4) small-v_c (v_c << v_p) limit -- recover the textbook result
# ----------------------------------------------------------------------
x = sp.symbols('x', positive=True)          # x = (v_c / v_p)^2, our small parameter
R_exact_x = R_exact.subs(vc, sp.sqrt(x) * vp)

R_series = sp.series(R_exact_x, x, 0, 3).removeO()   # keep leading nonzero order
R_series = sp.simplify(R_series.subs(x, vc**2 / vp**2))

print("\nLeading-order (v_c << v_p) approximation:")
sp.pprint(R_series)
# -> N*A*v_c**4 / (4*sqrt(pi)*v_p**3)
#    matches the hand-derived boxed result exactly.

# sanity check: ratio of exact to approximate should -> 1 as v_c/v_p -> 0
ratio = sp.simplify(R_exact / R_series)
ratio_limit = sp.limit(ratio.subs(vc, sp.sqrt(x) * vp), x, 0)
print("\nlim_(v_c/v_p -> 0)  [R_exact / R_approx] =", ratio_limit)  # -> 1


# ----------------------------------------------------------------------
# (5) Equilibrium atom number (part b): N_eq = R_load / (N v_bar sigma)
#     -- shows the background density N cancel automatically
# ----------------------------------------------------------------------
N_eq = sp.simplify(R_series / (N * vbar * sigma))

print("\nEquilibrium N_eq  (note: N has cancelled) =")
sp.pprint(N_eq)
# -> A*v_c**4 / (4*sqrt(pi)*v_p**3*v_bar*sigma)
#    N no longer appears anywhere in the expression -- this is the
#    symbolic proof of pressure-independence from part (b), sympy
#    doesn't even need to be told to cancel it, it falls out automatically.


# ----------------------------------------------------------------------
# (6) optional: numeric evaluation, plug in your Rb / T=300K values
# ----------------------------------------------------------------------
vals = {
    A: 6 * 0.02**2,      # m^2   (D = 2 cm cube)
    vc: 25,               # m/s
    vp: 239.4,             # m/s   (computed from sqrt(2 kB T / M) for Rb-87, 300K)
    vbar: 270.2,            # m/s
    sigma: 2e-17,          # m^2   (assumed trap-loss cross-section)
}
print("\nNumeric R_load/N (m^3/s):", float(R_series.subs(vals) / N.subs(vals, 1) if False else (R_series/N).subs(vals)))
print("Numeric N_eq (atoms):     ", float(N_eq.subs(vals)))