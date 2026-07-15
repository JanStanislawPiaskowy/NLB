import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

# ── 1. Import gas ─────────────────────────────────────────────────────────────
gas = ct.Solution('h2_dissociation_nasa9.yaml', 'h2_dissociation_RK')

iH2 = gas.species_index('H2')
iH  = gas.species_index('H')

# ── 2. Geometry ───────────────────────────────────────────────────────────────
ft_to_m = 30.48 / 100
L = 6 * ft_to_m  # m

R3 = 24.45 / 100
R_in = 0.911 * ft_to_m
R_out = 1.320 * ft_to_m

A0 = np.pi * (R_in**2 - R3**2)
b = (R_out - R_in) / L

def R(x):
    return R_in + x * b

def A(x):
    r_x = R(x)
    return np.pi * (r_x**2 - R3**2)

def dAdx(x):
    return 2 * np.pi * R(x) * b

# ── 3. Heat input ─────────────────────────────────────────────────────────────
POWER_TO_PROPELLANT = 0.8375
Q_total = 4600 / 7 * 1_000_000 * POWER_TO_PROPELLANT

# Global multiplier on Q_total. Left at 1.0 for the baseline run; mutated
# by the power-perturbation sweep at the bottom of this script.
Q_SCALE = 1.0
#
# def q(x):
#     return Q_total * np.pi / (2 * L) * np.sin(np.pi / L * x)

def q(x):
    return Q_SCALE * Q_total / L

# ── 4. Inlet conditions ──────────────────────────────────────────────────────
T0 = 2250   # K
p0 = 500 * 101325  # Pa
X0 = 'H2:1'
mdot = 19.3 / 7  # kg/s

# --- Equilibrium inlet ---
gas.TPX = T0, p0, X0
gas.equilibrate('TP')
print(ct.get_data_directories())

rho0_eq = gas.density
Y0_eq   = gas.Y.copy()
cp0_eq  = gas.cp_mass
gam0_eq = gas.cp_mass / gas.cv_mass
a0_eq   = (gam0_eq * p0 / rho0_eq) ** 0.5
u0_eq   = mdot / (rho0_eq * A0)
M0_eq   = u0_eq / a0_eq

# --- Frozen inlet (pure H2, no equilibration) ---
gas.TPX = T0, p0, X0
rho0_fr = gas.density
Y0_fr   = gas.Y.copy()
cp0_fr  = gas.cp_mass
gam0_fr = gas.cp_mass / gas.cv_mass
a0_fr   = (gam0_fr * p0 / rho0_fr) ** 0.5
u0_fr   = mdot / (rho0_fr * A0)
M0_fr   = u0_fr / a0_fr

print("=" * 60)
print("INLET CONDITIONS")
print("=" * 60)
print(f"{'':30s} {'Equilibrium':>14s} {'Frozen':>14s}")
print(f"{'Density [kg/m³]':30s} {rho0_eq:14.3f} {rho0_fr:14.3f}")
print(f"{'Velocity [m/s]':30s} {u0_eq:14.2f} {u0_fr:14.2f}")
print(f"{'Mach number':30s} {M0_eq:14.4f} {M0_fr:14.4f}")
print(f"{'Speed of sound [m/s]':30s} {a0_eq:14.2f} {a0_fr:14.2f}")
print(f"{'Gamma':30s} {gam0_eq:14.4f} {gam0_fr:14.4f}")
print(f"{'cp [J/(kg·K)]':30s} {cp0_eq:14.2f} {cp0_fr:14.2f}")

if M0_eq >= 1.0:
    raise ValueError(f"Equilibrium inlet is supersonic (Ma={M0_eq:.3f})")
if M0_fr >= 1.0:
    raise ValueError(f"Frozen inlet is supersonic (Ma={M0_fr:.3f})")


# ── 5. Property helpers ──────────────────────────────────────────────────────

def equilibrium_properties(gas, T, p, dT=1.0, dp_frac=1e-4):
    """Equilibrium (dissociation) partial derivatives via central differences."""
    dp = p * dp_frac

    gas.TP = T, p;  gas.equilibrate('TP')
    rho = gas.density

    gas.TP = T + dT, p;  gas.equilibrate('TP')
    rho_Tp, h_Tp = gas.density, gas.enthalpy_mass
    gas.TP = T - dT, p;  gas.equilibrate('TP')
    rho_Tm, h_Tm = gas.density, gas.enthalpy_mass
    rho_T = (rho_Tp - rho_Tm) / (2 * dT)
    cp_eq = (h_Tp - h_Tm) / (2 * dT)

    gas.TP = T, p + dp;  gas.equilibrate('TP')
    rho_pp, h_pp = gas.density, gas.enthalpy_mass
    gas.TP = T, p - dp;  gas.equilibrate('TP')
    rho_pm, h_pm = gas.density, gas.enthalpy_mass
    rho_p = (rho_pp - rho_pm) / (2 * dp)
    h_p   = (h_pp - h_pm) / (2 * dp)

    gas.TP = T, p;  gas.equilibrate('TP')
    return rho, rho_T, rho_p, cp_eq, h_p


def frozen_properties(gas, T, p, Y_frozen, dT=1.0, dp_frac=1e-4):
    """Frozen-composition partial derivatives via central differences."""
    dp = p * dp_frac

    gas.TPY = T, p, Y_frozen
    rho = gas.density

    gas.TPY = T + dT, p, Y_frozen
    rho_Tp, h_Tp = gas.density, gas.enthalpy_mass
    gas.TPY = T - dT, p, Y_frozen
    rho_Tm, h_Tm = gas.density, gas.enthalpy_mass
    rho_T = (rho_Tp - rho_Tm) / (2 * dT)
    cp_fr = (h_Tp - h_Tm) / (2 * dT)

    gas.TPY = T, p + dp, Y_frozen
    rho_pp, h_pp = gas.density, gas.enthalpy_mass
    gas.TPY = T, p - dp, Y_frozen
    rho_pm, h_pm = gas.density, gas.enthalpy_mass
    rho_p = (rho_pp - rho_pm) / (2 * dp)
    h_p   = (h_pp - h_pm) / (2 * dp)

    gas.TPY = T, p, Y_frozen
    return rho, rho_T, rho_p, cp_fr, h_p


def finite_rate_properties(gas, T, p, Y, dT=1.0, dp_frac=1e-4, dY=1e-6):
    """Frozen partial derivatives at the CURRENT composition, plus reaction rates.

    Returns everything needed for the finite-rate ODE:
      rho, rho_T, rho_p, cp, h_p   — same as frozen_properties
      rho_Y   — d(rho)/d(Y_H2) at constant T, p  (with Y_H = 1 - Y_H2)
      h_Y     — d(h)/d(Y_H2)   at constant T, p  (= h_H2 - h_H)
      wdot    — net molar production rates [kmol/m³/s]
      W       — molecular weights [kg/kmol]
    """
    dp = p * dp_frac

    # --- Base state ---
    gas.TPY = T, p, Y
    rho = gas.density

    # --- T perturbation (frozen composition) ---
    gas.TPY = T + dT, p, Y
    rho_Tp, h_Tp = gas.density, gas.enthalpy_mass
    gas.TPY = T - dT, p, Y
    rho_Tm, h_Tm = gas.density, gas.enthalpy_mass
    rho_T = (rho_Tp - rho_Tm) / (2 * dT)
    cp    = (h_Tp - h_Tm) / (2 * dT)

    # --- p perturbation ---
    gas.TPY = T, p + dp, Y
    rho_pp, h_pp = gas.density, gas.enthalpy_mass
    gas.TPY = T, p - dp, Y
    rho_pm, h_pm = gas.density, gas.enthalpy_mass
    rho_p = (rho_pp - rho_pm) / (2 * dp)
    h_p   = (h_pp - h_pm) / (2 * dp)

    # --- Y_H2 perturbation (Y_H = 1 - Y_H2, perturbed in opposite direction) ---
    Y_plus       = Y.copy()
    Y_plus[iH2] += dY;  Y_plus[iH] -= dY
    gas.TPY = T, p, Y_plus
    rho_Yp, h_Yp = gas.density, gas.enthalpy_mass

    Y_minus       = Y.copy()
    Y_minus[iH2] -= dY;  Y_minus[iH] += dY
    gas.TPY = T, p, Y_minus
    rho_Ym, h_Ym = gas.density, gas.enthalpy_mass

    rho_Y = (rho_Yp - rho_Ym) / (2 * dY)
    h_Y   = (h_Yp - h_Ym) / (2 * dY)

    # --- Restore base state and get reaction rates ---
    gas.TPY = T, p, Y
    wdot = gas.net_production_rates.copy()   # [kmol/m³/s]
    W    = gas.molecular_weights.copy()      # [kg/kmol]

    return rho, rho_T, rho_p, cp, h_p, rho_Y, h_Y, wdot, W


# ── 6. ODE right-hand sides ──────────────────────────────────────────────────

def rhs_equilibrium(x, state):
    T, p = float(state[0]), float(state[1])
    rho, rho_T, rho_p, cp, h_p = equilibrium_properties(gas, T, p)

    Ax = A(x);  u = mdot / (rho * Ax);  u2 = u**2
    qx = q(x);  dA = dAdx(x)

    D = (1.0 - u2 * rho_p) + u2 * rho_T * (h_p - 1.0 / rho) / cp

    dp_dx = (u2 * rho_T * qx / (mdot * cp)
             + rho * u2 / Ax * dA) / D

    dT_dx = (qx / mdot - (h_p - 1.0 / rho) * dp_dx) / cp
    return [dT_dx, dp_dx]


def rhs_frozen(x, state):
    T, p = float(state[0]), float(state[1])
    rho, rho_T, rho_p, cp, h_p = frozen_properties(gas, T, p, Y0_fr)

    Ax = A(x);  u = mdot / (rho * Ax);  u2 = u**2
    qx = q(x);  dA = dAdx(x)

    D = (1.0 - u2 * rho_p) + u2 * rho_T * (h_p - 1.0 / rho) / cp

    dp_dx = (u2 * rho_T * qx / (mdot * cp)
             + rho * u2 / Ax * dA) / D

    dT_dx = (qx / mdot - (h_p - 1.0 / rho) * dp_dx) / cp
    return [dT_dx, dp_dx]


def rhs_finite_rate(x, state):
    """Finite-rate kinetics: state = [T, p, Y_H2].

    Three conservation equations plus species:

    Species:     dY_H2/dx = wdot_H2 * W_H2 / (rho * u)

    Pressure:    dp/dx = { u^2 rho_T q/(mdot cp) + rho u^2/A dA/dx
                           + u^2 (rho_Y - rho_T h_Y / cp) S_H2 } / D

    Temperature: dT/dx = [ q/mdot - (h_p - 1/rho) dp/dx
                           - h_Y S_H2 ] / cp

    The extra terms compared to equilibrium/frozen arise from the
    composition changing at a finite rate, coupling back into the
    density and enthalpy through d(rho)/d(Y_H2) and d(h)/d(Y_H2).
    """
    T, p, Y_H2 = float(state[0]), float(state[1]), float(state[2])

    Y_H2 = np.clip(Y_H2, 0.0, 1.0)
    Y_H  = 1.0 - Y_H2

    Y = np.zeros(gas.n_species)
    Y[iH2] = Y_H2;  Y[iH] = Y_H

    rho, rho_T, rho_p, cp, h_p, rho_Y, h_Y, wdot, W = \
        finite_rate_properties(gas, T, p, Y)

    Ax = A(x);  u = mdot / (rho * Ax);  u2 = u**2
    qx = q(x);  dA = dAdx(x)

    # Species source: dY_H2/dx
    S_H2 = wdot[iH2] * W[iH2] / (rho * u)

    # Generalised denominator (frozen derivatives at current Y)
    D = (1.0 - u2 * rho_p) + u2 * rho_T * (h_p - 1.0 / rho) / cp

    # Pressure gradient — includes species source coupling
    dp_dx = (u2 * rho_T * qx / (mdot * cp)
             + rho * u2 / Ax * dA
             + u2 * (rho_Y - rho_T * h_Y / cp) * S_H2) / D

    # Temperature gradient — includes enthalpy of reaction
    dT_dx = (qx / mdot - (h_p - 1.0 / rho) * dp_dx - h_Y * S_H2) / cp

    return [dT_dx, dp_dx, S_H2]


# ── 7. Choking events ────────────────────────────────────────────────────────

def choke_event_eq(x, state):
    T, p = float(state[0]), float(state[1])
    rho, rho_T, rho_p, cp, h_p = equilibrium_properties(gas, T, p)
    u = mdot / (rho * A(x));  u2 = u**2
    return (1.0 - u2 * rho_p) + u2 * rho_T * (h_p - 1.0 / rho) / cp

choke_event_eq.terminal  = True
choke_event_eq.direction = -1


def choke_event_fr(x, state):
    T, p = float(state[0]), float(state[1])
    rho, rho_T, rho_p, cp, h_p = frozen_properties(gas, T, p, Y0_fr)
    u = mdot / (rho * A(x));  u2 = u**2
    return (1.0 - u2 * rho_p) + u2 * rho_T * (h_p - 1.0 / rho) / cp

choke_event_fr.terminal  = True
choke_event_fr.direction = -1


def choke_event_finite(x, state):
    T, p, Y_H2 = float(state[0]), float(state[1]), float(state[2])
    Y_H2 = np.clip(Y_H2, 0.0, 1.0)
    Y = np.zeros(gas.n_species)
    Y[iH2] = Y_H2;  Y[iH] = 1.0 - Y_H2
    rho, rho_T, rho_p, cp, h_p = frozen_properties(gas, T, p, Y)
    u = mdot / (rho * A(x));  u2 = u**2
    return (1.0 - u2 * rho_p) + u2 * rho_T * (h_p - 1.0 / rho) / cp

choke_event_finite.terminal  = True
choke_event_finite.direction = -1


# ── 8. Integrate all three cases ─────────────────────────────────────────────

x_space = np.linspace(0.0, L, 400)

print("\n>>> Solving EQUILIBRIUM (with dissociation) case ...")
sol_eq = solve_ivp(
    rhs_equilibrium, (0.0, L), [T0, p0],
    method='RK45', t_eval=x_space,
    rtol=1e-5, atol=1e-6,
    events=choke_event_eq,
)
if not sol_eq.success:
    print(f"  Solver warning: {sol_eq.message}")
if sol_eq.t_events[0].size:
    print(f"  *** Flow choked at x = {sol_eq.t_events[0][0]:.4f} m ***")

print(">>> Solving FROZEN (no dissociation) case ...")
sol_fr = solve_ivp(
    rhs_frozen, (0.0, L), [T0, p0],
    method='RK45', t_eval=x_space,
    rtol=1e-5, atol=1e-6,
    events=choke_event_fr,
)
if not sol_fr.success:
    print(f"  Solver warning: {sol_fr.message}")
if sol_fr.t_events[0].size:
    print(f"  *** Flow choked at x = {sol_fr.t_events[0][0]:.4f} m ***")

# Finite-rate: use BDF (stiff solver) — the reaction rates at 500 atm are
# enormous, making the species equation very stiff relative to T and p.
print(">>> Solving FINITE-RATE case (stiff solver) ...")
sol_fin = solve_ivp(
    rhs_finite_rate, (0.0, L), [T0, p0, 1.0],
    method='BDF',                # stiff solver
    t_eval=x_space,
    rtol=1e-6, atol=[1e-1, 1e0, 1e-10],   # scaled tolerances: T~1000s, p~1e7, Y~0-1
    events=choke_event_finite,
    max_step=L / 200,            # prevent enormous leaps
)
if not sol_fin.success:
    print(f"  Solver warning: {sol_fin.message}")
if sol_fin.t_events[0].size:
    print(f"  *** Flow choked at x = {sol_fin.t_events[0][0]:.4f} m ***")


# ── 9. Post-process all three cases ──────────────────────────────────────────

def postprocess_equilibrium(sol):
    x = sol.t;  T = sol.y[0];  p = sol.y[1]
    n = len(x)
    rho = np.zeros(n);  u = np.zeros(n);  D = np.zeros(n)
    M = np.zeros(n);  XH2 = np.zeros(n);  XH = np.zeros(n)

    for i in range(n):
        ri, rT, rP, cp, hp = equilibrium_properties(gas, T[i], p[i])
        rho[i] = ri
        u[i]   = mdot / (ri * A(x[i]))
        u2 = u[i]**2
        D[i] = (1.0 - u2 * rP) + u2 * rT * (hp - 1.0 / ri) / cp
        gam = gas.cp_mass / gas.cv_mass
        a = (gam * p[i] / ri)**0.5
        M[i] = u[i] / a
        XH2[i] = gas.X[iH2]
        XH[i]  = gas.X[iH]
    return rho, u, D, M, XH2, XH


def postprocess_frozen(sol, Y_frozen):
    x = sol.t;  T = sol.y[0];  p = sol.y[1]
    n = len(x)
    rho = np.zeros(n);  u = np.zeros(n);  D = np.zeros(n);  M = np.zeros(n)

    for i in range(n):
        ri, rT, rP, cp, hp = frozen_properties(gas, T[i], p[i], Y_frozen)
        rho[i] = ri
        u[i]   = mdot / (ri * A(x[i]))
        u2 = u[i]**2
        D[i] = (1.0 - u2 * rP) + u2 * rT * (hp - 1.0 / ri) / cp
        gam = gas.cp_mass / gas.cv_mass
        a = (gam * p[i] / ri)**0.5
        M[i] = u[i] / a
    return rho, u, D, M


def postprocess_finite_rate(sol):
    x = sol.t;  T = sol.y[0];  p = sol.y[1];  Y_H2_arr = sol.y[2]
    n = len(x)
    rho = np.zeros(n);  u = np.zeros(n);  D = np.zeros(n)
    M = np.zeros(n);  XH2 = np.zeros(n);  XH = np.zeros(n)

    W_H2 = gas.molecular_weights[iH2]
    W_H  = gas.molecular_weights[iH]

    for i in range(n):
        Y_H2_i = np.clip(Y_H2_arr[i], 0.0, 1.0)
        Y = np.zeros(gas.n_species)
        Y[iH2] = Y_H2_i;  Y[iH] = 1.0 - Y_H2_i

        gas.TPY = T[i], p[i], Y
        ri = gas.density
        rho[i] = ri
        u[i]   = mdot / (ri * A(x[i]))
        u2 = u[i]**2

        _, rT, rP, cp, hp = frozen_properties(gas, T[i], p[i], Y)
        D[i] = (1.0 - u2 * rP) + u2 * rT * (hp - 1.0 / ri) / cp

        gam = gas.cp_mass / gas.cv_mass
        a = (gam * p[i] / ri)**0.5
        M[i] = u[i] / a

        # Mass fractions → mole fractions
        n_H2 = Y_H2_i / W_H2
        n_H  = (1.0 - Y_H2_i) / W_H
        n_tot = n_H2 + n_H
        XH2[i] = n_H2 / n_tot
        XH[i]  = n_H / n_tot

    return rho, u, D, M, XH2, XH


print("\n>>> Post-processing equilibrium ...")
rho_eq, u_eq, D_eq, M_eq, XH2_eq, XH_eq = postprocess_equilibrium(sol_eq)

print(">>> Post-processing frozen ...")
rho_fr, u_fr, D_fr, M_fr = postprocess_frozen(sol_fr, Y0_fr)

print(">>> Post-processing finite-rate ...")
rho_fin, u_fin, D_fin, M_fin, XH2_fin, XH_fin = postprocess_finite_rate(sol_fin)

# Normalised axial coordinates
xL_eq  = sol_eq.t / L
xL_fr  = sol_fr.t / L
xL_fin = sol_fin.t / L


# ── 10. Print outlet comparison ──────────────────────────────────────────────

print("\n" + "=" * 70)
print("OUTLET CONDITIONS")
print("=" * 70)
print(f"{'':30s} {'Equilibrium':>14s} {'Frozen':>14s} {'Finite-rate':>14s}")
print(f"{'Density [kg/m³]':30s} {rho_eq[-1]:14.3f} {rho_fr[-1]:14.3f} {rho_fin[-1]:14.3f}")
print(f"{'Velocity [m/s]':30s} {u_eq[-1]:14.2f} {u_fr[-1]:14.2f} {u_fin[-1]:14.2f}")
print(f"{'Mach (frozen gamma)':30s} {M_eq[-1]:14.4f} {M_fr[-1]:14.4f} {M_fin[-1]:14.4f}")
print(f"{'Temperature [K]':30s} {sol_eq.y[0][-1]:14.1f} {sol_fr.y[0][-1]:14.1f} {sol_fin.y[0][-1]:14.1f}")
print(f"{'Pressure [atm]':30s} {sol_eq.y[1][-1]/101325:14.1f} {sol_fr.y[1][-1]/101325:14.1f} {sol_fin.y[1][-1]/101325:14.1f}")
print(f"{'X_H2':30s} {XH2_eq[-1]:14.4f} {'1.0000':>14s} {XH2_fin[-1]:14.4f}")
print(f"{'X_H':30s} {XH_eq[-1]:14.4f} {'0.0000':>14s} {XH_fin[-1]:14.4f}")

# Outlet gamma and cp
gas.TP = sol_eq.y[0][-1], sol_eq.y[1][-1]
gas.equilibrate('TP')
gam_out_eq = gas.cp_mass / gas.cv_mass;  cp_out_eq = gas.cp_mass

gas.TPY = sol_fr.y[0][-1], sol_fr.y[1][-1], Y0_fr
gam_out_fr = gas.cp_mass / gas.cv_mass;  cp_out_fr = gas.cp_mass

Y_fin_out = np.zeros(gas.n_species)
Y_fin_out[iH2] = np.clip(sol_fin.y[2][-1], 0, 1)
Y_fin_out[iH]  = 1.0 - Y_fin_out[iH2]
gas.TPY = sol_fin.y[0][-1], sol_fin.y[1][-1], Y_fin_out
gam_out_fin = gas.cp_mass / gas.cv_mass;  cp_out_fin = gas.cp_mass

print(f"{'Gamma':30s} {gam_out_eq:14.4f} {gam_out_fr:14.4f} {gam_out_fin:14.4f}")
print(f"{'cp [J/(kg·K)]':30s} {cp_out_eq:14.2f} {cp_out_fr:14.2f} {cp_out_fin:14.2f}")

x = sol_fin.t
T = sol_fin.y[0]
p = sol_fin.y[1]
Y_H2_arr = sol_fin.y[2]

W_H2 = gas.molecular_weights[iH2]

for i in range(len(x)):
    Y_H2_i = np.clip(Y_H2_arr[i], 0.0, 1.0)
    Y = np.zeros(gas.n_species)
    Y[iH2] = Y_H2_i
    Y[iH] = 1.0 - Y_H2_i

    gas.TPY = T[i], p[i], Y

    rho_i = gas.density
    u_i = mdot / (rho_i * A(x[i]))

    # Everything Cantera knows about the reaction at this point
    wdot = gas.net_production_rates  # kmol/m³/s per species
    kf = gas.forward_rate_constants  # forward k
    kb = gas.reverse_rate_constants  # reverse k
    qf = gas.forward_rates_of_progress  # r_f (kmol/m³/s per reaction)
    qb = gas.reverse_rates_of_progress  # r_b
    qnet = gas.net_rates_of_progress  # r_f - r_b

    S_H2 = wdot[iH2] * W_H2 / (rho_i * u_i)  # dY_H2/dx

    print(f"x/L={x[i] / L:.3f}  T={T[i]:.0f}K  "
          f"r_f={qf[0]:.3e}  r_b={qb[0]:.3e}  "
          f"net={qnet[0]:.3e}  "
          f"wdot_H2={wdot[iH2]:.3e}  "
          f"dY_H2/dx={S_H2:.3e}")

    gas.TPY = T[i], p[i], Y

    conc_H2 = gas.concentrations[iH2] / 1000  # mol/m³ → kmol/m³
    qf = gas.forward_rates_of_progress[0]  # kmol/m³/s

    tau_chem = conc_H2 / qf if qf > 0 else float('inf')

    dx = L / 400
    tau_flow = dx / u_i  # residence time in the duct

    print(f"x/L={x[i] / L:.3f}  tau_chem={tau_chem:.3e} s  "
          f"tau_flow={tau_flow:.3e} s  "
          f"Da={tau_flow / tau_chem:.1f}")


# ── 11. Plot comparison (separate figures) ────────────────────────────────────

os.makedirs('figures', exist_ok=True)

lw = 2.0
c_eq  = '#1f77b4'
c_fr  = '#d62728'
c_fin = '#2ca02c'

def make_fig(ylabel, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlabel('x / L')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return fig, ax

# 1 ── Density
fig, ax = make_fig('ρ  [kg/m³]', 'Density')
ax.plot(xL_eq,  rho_eq,  color=c_eq,  lw=lw, label='Equilibrium')
ax.plot(xL_fr,  rho_fr,  color=c_fr,  lw=lw, ls='--', label='Frozen')
ax.plot(xL_fin, rho_fin, color=c_fin, lw=lw, ls='-.', label='Finite-rate')
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax.legend();  fig.tight_layout()
fig.savefig('figures/density_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 2 ── Temperature
fig, ax = make_fig('T  [K]', 'Temperature')
ax.plot(xL_eq,  sol_eq.y[0],  color=c_eq,  lw=lw, label='Equilibrium')
ax.plot(xL_fr,  sol_fr.y[0],  color=c_fr,  lw=lw, ls='--', label='Frozen')
ax.plot(xL_fin, sol_fin.y[0], color=c_fin, lw=lw, ls='-.', label='Finite-rate')
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
ax.legend();  fig.tight_layout()
fig.savefig('figures/temperature_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 3 ── Pressure drop
fig, ax = make_fig('Δp from inlet [kPa]', 'Pressure drop')
ax.plot(xL_eq,  (sol_eq.y[1]  - p0) / 1000, color=c_eq,  lw=lw, label='Equilibrium')
ax.plot(xL_fr,  (sol_fr.y[1]  - p0) / 1000, color=c_fr,  lw=lw, ls='--', label='Frozen')
ax.plot(xL_fin, (sol_fin.y[1] - p0) / 1000, color=c_fin, lw=lw, ls='-.', label='Finite-rate')
ax.legend();  fig.tight_layout()
fig.savefig('figures/pressure_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 4 ── Species mole fractions
fig, ax = make_fig('Mole fraction', 'Species mole fractions')
ax.plot(xL_eq,  XH2_eq,  color=c_eq,  lw=lw, label='H₂ (equil.)')
ax.plot(xL_eq,  XH_eq,   color=c_eq,  lw=lw, ls=':', label='H  (equil.)')
ax.plot(xL_fin, XH2_fin, color=c_fin, lw=lw, ls='-.', label='H₂ (finite-rate)')
ax.plot(xL_fin, XH_fin,  color=c_fin, lw=lw, ls=':', label='H  (finite-rate)')
ax.axhline(1.0, color=c_fr, ls='--', lw=1.0, alpha=0.6, label='H₂ (frozen)')
ax.axhline(0.0, color=c_fr, ls=':',  lw=1.0, alpha=0.6, label='H  (frozen)')
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
ax.legend(fontsize=7);  fig.tight_layout()
fig.savefig('figures/species_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 5 ── Mach number
fig, ax = make_fig('Mach number', 'Mach number')
ax.plot(xL_eq,  M_eq,  color=c_eq,  lw=lw, label='Equilibrium')
ax.plot(xL_fr,  M_fr,  color=c_fr,  lw=lw, ls='--', label='Frozen')
ax.plot(xL_fin, M_fin, color=c_fin, lw=lw, ls='-.', label='Finite-rate')
ax.legend();  fig.tight_layout()
fig.savefig('figures/mach_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# 6 ── Generalised denominator D
fig, ax = make_fig('D (gen. denominator)', 'Generalised denominator D')
ax.plot(xL_eq,  D_eq,  color=c_eq,  lw=lw, label='Equilibrium')
ax.plot(xL_fr,  D_fr,  color=c_fr,  lw=lw, ls='--', label='Frozen')
ax.plot(xL_fin, D_fin, color=c_fin, lw=lw, ls='-.', label='Finite-rate')
ax.axhline(0, color='k', ls='--', alpha=0.4, label='Choking (D=0)')
ax.legend();  fig.tight_layout()
fig.savefig('figures/denominator_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAll figures saved to figures/ folder.")

# ── 11. Export density profile for OpenMC GCR model ────────────────────────────────────
#
_profile_path = 'settings/h2_density_profile.npz'

np.savez(
    _profile_path,
    x_m = sol_fin.t,
    rho_kgm3 = rho_fin,
    T_K = sol_fin.y[0],
    p_Pa = sol_fin.y[1]
)
print(f"Finite-rate H₂ profile saved → {_profile_path}")


# ── 12. Power-perturbed profiles ─────────────────────────────────────────────
# Re-run the FINITE-RATE case at a range of cavity powers and save one profile
# per level. Equilibrium and frozen cases are diagnostic only and not
# re-integrated here -- only the finite-rate profile is consumed by the OpenMC
# GCR model (via h2_density_profile_path).

POWER_DELTAS = [-0.10, -0.05, +0.05, +0.10]  # fractional change in Q_total


def _power_tag(delta):
    sign = 'p' if delta >= 0 else 'm'
    return f'power_{sign}{abs(delta) * 100:02.0f}pct'


def solve_and_save_finite_rate(scale, out_path):
    """Re-integrate the finite-rate ODE with Q_total multiplied by `scale`
    and save the resulting x / rho / T / p profile to `out_path`.

    Reuses the module-level inlet conditions, geometry and solver settings.
    Returns (sol, rho_array) for downstream plotting.
    """
    global Q_SCALE
    Q_SCALE_backup = Q_SCALE
    Q_SCALE = scale
    try:
        sol = solve_ivp(
            rhs_finite_rate, (0.0, L), [T0, p0, 1.0],
            method='BDF',
            t_eval=x_space,
            rtol=1e-6, atol=[1e-1, 1e0, 1e-10],
            events=choke_event_finite,
            max_step=L / 200,
        )
        if not sol.success:
            print(f"  Solver warning (scale={scale:+.3f}): {sol.message}")
        if sol.t_events[0].size:
            print(f"  *** Flow choked at x = {sol.t_events[0][0]:.4f} m "
                  f"(scale = {scale:.3f}) ***")

        rho, _u, _D, _M, _XH2, _XH = postprocess_finite_rate(sol)

        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        np.savez(
            out_path,
            x_m=sol.t,
            rho_kgm3=rho,
            T_K=sol.y[0],
            p_Pa=sol.y[1],
        )
        print(f"  Outlet: T = {sol.y[0][-1]:7.1f} K   "
              f"p = {sol.y[1][-1] / 101325:6.1f} atm   "
              f"rho = {rho[-1]:6.3f} kg/m³")
        print(f"  Saved → {out_path}")
        return sol, rho
    finally:
        Q_SCALE = Q_SCALE_backup


print("\n" + "=" * 70)
print("POWER PERTURBATION SWEEP (finite-rate only)")
print("=" * 70)

perturbation_results = {}
for delta in POWER_DELTAS:
    scale = 1.0 + delta
    tag = _power_tag(delta)
    out_path = f'settings/h2_density_profile_{tag}.npz'
    print(f"\n>>> Power {delta * 100:+.0f}%  (Q scale = {scale:.3f})")
    sol, rho = solve_and_save_finite_rate(scale, out_path)
    perturbation_results[delta] = (sol, rho)

# Comparison plot: baseline + all perturbations
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].set_xlabel('x / L');  axes[0].set_ylabel('T [K]')
axes[0].set_title('Temperature')
axes[1].set_xlabel('x / L');  axes[1].set_ylabel('ρ [kg/m³]')
axes[1].set_title('Density')
axes[2].set_xlabel('x / L');  axes[2].set_ylabel('p [atm]')
axes[2].set_title('Pressure')

# Baseline (solid black)
axes[0].plot(sol_fin.t / L, sol_fin.y[0],          'k-', lw=2.0, label='baseline')
axes[1].plot(sol_fin.t / L, rho_fin,               'k-', lw=2.0, label='baseline')
axes[2].plot(sol_fin.t / L, sol_fin.y[1] / 101325, 'k-', lw=2.0, label='baseline')

# Perturbations — blue for negative, red for positive, shade by magnitude
cmap = plt.cm.coolwarm
max_abs_delta = max(abs(d) for d in POWER_DELTAS)
for delta in sorted(perturbation_results.keys()):
    sol, rho = perturbation_results[delta]
    c = cmap(0.5 + 0.5 * delta / max_abs_delta)
    label = f'{delta * 100:+.0f} %'
    axes[0].plot(sol.t / L, sol.y[0],          color=c, lw=1.5, ls='--', label=label)
    axes[1].plot(sol.t / L, rho,               color=c, lw=1.5, ls='--', label=label)
    axes[2].plot(sol.t / L, sol.y[1] / 101325, color=c, lw=1.5, ls='--', label=label)

for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle('Finite-rate H₂ profile: cavity-power sensitivity')
fig.tight_layout()
fig.savefig('figures/power_perturbation_profiles.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print(f"Done. {len(POWER_DELTAS)} perturbed profiles written to settings/.")
print("=" * 70)