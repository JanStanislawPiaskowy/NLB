"""Four-factor decomposition of k for the GCR model.

Theory
------
The four-factor formula splits the infinite multiplication factor into

    k_inf = epsilon * p * f * eta

In continuous-energy Monte Carlo the cleanest, exactly self-consistent way
to obtain the factors is as ratios of tallied reaction rates, split at a
thermal cutoff energy E_c:

    A_th        absorption rate, incident E <  E_c, whole model
    A_fast      absorption rate, incident E >= E_c, whole model
    A_th_fuel   absorption rate, incident E <  E_c, fuel materials only
    F_th        nu-fission rate induced by neutrons with E <  E_c
    F_tot       nu-fission rate, all incident energies

    epsilon = F_tot / F_th                (fast-fission factor)
    p       = A_th / (A_th + A_fast)      (resonance-escape surrogate)
    f       = A_th_fuel / A_th            (thermal utilisation)
    eta     = F_th / A_th_fuel            (reproduction factor)

Multiplying the chain telescopes to  k_inf = F_tot / A_tot  (production per
absorption, the leakage-free multiplication), and the lumped non-leakage
probability follows from the eigenvalue:  P_NL = k_eff / k_inf.

Caveats for the gas-core reactor
--------------------------------
* kT(fuel) ~ 1.7 eV at 20,000 K and kT(H2) ~ 0.3-0.5 eV: neutrons are
  upscattered ACROSS the classical 0.625 eV cutoff, so the textbook
  reading of p blurs.  The tally therefore carries SEVERAL candidate
  cutoffs at once; always check how sensitive conclusions are to E_c.
* p as defined folds leakage out entirely (four-factor, not six-factor).
  Splitting P_NL into fast/thermal parts would additionally need energy-
  resolved 'current' tallies on the vacuum boundary surfaces.

Usage
-----
    from gcr.analysis.four_factors import (add_four_factor_tallies,
                                           compute_four_factors)
    core = GCR(config)
    core.build()
    add_four_factor_tallies(core)          # just registers -- order-independent
    core.run()
    results = compute_four_factors(core.statepoint_path)

Note how much simpler registration is than in the pre-refactor version:
this module used to need a `_gather_existing_tallies` crutch mirroring the
model's private attributes; with the registry it is one call that cannot
drop anybody else's tallies.
"""

from dataclasses import dataclass

import numpy as np
import openmc

from ..tallies import TallyBundle

# Candidate thermal cutoffs [eV].  0.625 eV is the classical LWR choice;
# the higher values probe sensitivity given the extreme GCR temperatures.
DEFAULT_CUTOFFS_EV = (0.625, 1.86, 5.0)

_TALLY_GLOBAL = 'four_factor_global'
_TALLY_FUEL = 'four_factor_fuel'
_E_MAX = 2.0e7  # [eV]


# ---------------------------------------------------------------------------
# Tally construction (call any time between build() and run())
# ---------------------------------------------------------------------------

def _fuel_materials(materials: dict) -> list:
    """All fuel materials, deduplicated by object identity.

    Same name convention as materials.apply_fuel_density_alpha, so the two
    can never disagree about what counts as fuel.
    """
    seen, mats = set(), []
    for m in materials.values():
        if id(m) in seen:
            continue
        seen.add(id(m))
        if m.name == 'fuel' or m.name.startswith(('fuel_inner', 'fuel_outer')):
            mats.append(m)
    if not mats:
        raise RuntimeError('No fuel materials found -- call GCR.build() first.')
    return mats


def add_four_factor_tallies(gcr, cutoffs_ev=DEFAULT_CUTOFFS_EV) -> None:
    """Register the two reaction-rate tallies needed for the four factors.

    One EnergyFilter carries ALL candidate cutoffs simultaneously, so a
    single run supports the whole sensitivity study: the post-processor
    simply groups the energy bins differently for each cutoff.
    """
    edges = [0.0] + sorted(cutoffs_ev) + [_E_MAX]
    e_filt = openmc.EnergyFilter(edges)

    # Whole-model absorption, split in energy
    t_global = openmc.Tally(name=_TALLY_GLOBAL)
    t_global.filters = [e_filt]
    t_global.scores = ['absorption']

    # Fuel-only absorption and nu-fission, split in energy
    t_fuel = openmc.Tally(name=_TALLY_FUEL)
    t_fuel.filters = [openmc.MaterialFilter(_fuel_materials(gcr.materials)), e_filt]
    t_fuel.scores = ['absorption', 'nu-fission']

    gcr.register_tally(TallyBundle(
        tallies=[t_global, t_fuel],
        meta={'cutoffs_ev': tuple(sorted(cutoffs_ev))},
    ))

    print(f'Four-factor tallies registered (cutoff candidates: '
          f'{", ".join(f"{c:g} eV" for c in sorted(cutoffs_ev))})')


# ---------------------------------------------------------------------------
# Post-processing (needs only the statepoint)
# ---------------------------------------------------------------------------

@dataclass
class FourFactors:
    cutoff_ev: float
    epsilon: float
    epsilon_sd: float
    p: float
    p_sd: float
    f: float
    f_sd: float
    eta: float
    eta_sd: float
    k_inf: float
    k_inf_sd: float
    k_eff: float
    k_eff_sd: float
    P_NL: float
    P_NL_sd: float


def _ratio(a, sa, b, sb):
    """a/b with first-order uncertainty propagation (correlations ignored)."""
    r = a / b
    return r, abs(r) * np.sqrt((sa / a) ** 2 + (sb / b) ** 2)


def _split_at(cutoff, edges, mean, sd):
    """Sum energy-binned (mean, sd) below and at-or-above `cutoff`.

    `edges` are the filter bin edges; bin i spans [edges[i], edges[i+1]).
    Standard deviations of independent bins add in quadrature.
    """
    lo = np.array([edges[i + 1] <= cutoff + 1e-9 for i in range(len(mean))])
    m_lo, m_hi = mean[lo].sum(), mean[~lo].sum()
    s_lo = np.sqrt((sd[lo] ** 2).sum())
    s_hi = np.sqrt((sd[~lo] ** 2).sum())
    return (m_lo, s_lo), (m_hi, s_hi)


def compute_four_factors(statepoint_path: str, verbose: bool = True):
    """Compute the four factors for every cutoff carried by the tallies.

    Returns a list of FourFactors, one per candidate cutoff.
    """
    print('Readin SP')
    sp = openmc.StatePoint(statepoint_path, autolink=False)
    print('Read SP. Reading tallies')

    tg = sp.get_tally(name=_TALLY_GLOBAL)
    tf = sp.get_tally(name=_TALLY_FUEL)

    print('Read tallies. Prepapring filters')

    e_filt = tg.find_filter(openmc.EnergyFilter)
    edges = np.asarray(e_filt.values)          # length n_bins + 1
    n_e = len(edges) - 1
    cutoffs = [e for e in edges[1:-1]]         # interior edges = candidates

    print('Prepared filters. reshaping data')
    # --- global absorption: filters = [energy] -> shape (n_e, 1, 1) ---------
    gm = tg.get_reshaped_data(value='mean')
    gs = tg.get_reshaped_data(value='std_dev')
    assert gm.shape[0] == n_e, f'unexpected global tally shape {gm.shape}'
    A_mean = gm[:, 0, 0]
    A_sd = gs[:, 0, 0]

    # --- fuel tally: filters = [material, energy], scores = [abs, nu-fis] ---
    #     reshaped -> (n_mat, n_e, 1, 2); sum over materials (quadrature for
    #     the std devs).  Score order follows tally.scores.
    fm = tf.get_reshaped_data(value='mean')
    fs = tf.get_reshaped_data(value='std_dev')
    assert fm.shape[1] == n_e and fm.shape[-1] == 2, \
        f'unexpected fuel tally shape {fm.shape}'
    i_abs = tf.scores.index('absorption')
    i_nuf = tf.scores.index('nu-fission')
    Af_mean = fm[..., i_abs].sum(axis=0)[:, 0]
    Af_sd = np.sqrt((fs[..., i_abs] ** 2).sum(axis=0))[:, 0]
    F_mean = fm[..., i_nuf].sum(axis=0)[:, 0]
    F_sd = np.sqrt((fs[..., i_nuf] ** 2).sum(axis=0))[:, 0]

    print('Getting keff')

    k_eff = float(sp.keff.nominal_value)
    k_eff_sd = float(sp.keff.std_dev)

    results = []
    print('Entering for loop')
    for cutoff in cutoffs:
        print('cutoff', cutoff)
        (A_th, sA_th), (A_fs, sA_fs) = _split_at(cutoff, edges, A_mean, A_sd)
        (Af_th, sAf_th), _ = _split_at(cutoff, edges, Af_mean, Af_sd)
        (F_th, sF_th), (F_fs, sF_fs) = _split_at(cutoff, edges, F_mean, F_sd)

        F_tot, sF_tot = F_th + F_fs, np.hypot(sF_th, sF_fs)
        A_tot, sA_tot = A_th + A_fs, np.hypot(sA_th, sA_fs)

        eps, s_eps = _ratio(F_tot, sF_tot, F_th, sF_th)
        p, s_p = _ratio(A_th, sA_th, A_tot, sA_tot)
        f, s_f = _ratio(Af_th, sAf_th, A_th, sA_th)
        eta, s_eta = _ratio(F_th, sF_th, Af_th, sAf_th)

        k_inf = eps * p * f * eta
        s_kinf = k_inf * np.sqrt((s_eps / eps) ** 2 + (s_p / p) ** 2 +
                                 (s_f / f) ** 2 + (s_eta / eta) ** 2)

        # Exact identity check: eps*p*f*eta must equal F_tot / A_tot
        k_inf_direct = F_tot / A_tot
        assert abs(k_inf - k_inf_direct) < 1e-10 * k_inf, \
            'telescoping identity violated -- tally bookkeeping bug'

        P_NL, s_PNL = _ratio(k_eff, k_eff_sd, k_inf, s_kinf)

        results.append(FourFactors(
            cutoff_ev=float(cutoff),
            epsilon=eps, epsilon_sd=s_eps,
            p=p, p_sd=s_p,
            f=f, f_sd=s_f,
            eta=eta, eta_sd=s_eta,
            k_inf=k_inf, k_inf_sd=s_kinf,
            k_eff=k_eff, k_eff_sd=k_eff_sd,
            P_NL=P_NL, P_NL_sd=s_PNL,
        ))

    if verbose:
        bar = '=' * 78
        print('\n' + bar)
        print('  Four-factor decomposition   k_inf = eps * p * f * eta,'
              '   P_NL = k_eff / k_inf')
        print(bar)
        print(f'  {"E_c [eV]":>9} {"epsilon":>15} {"p":>15} {"f":>15}'
              f' {"eta":>15} {"k_inf":>15} {"P_NL":>8}')
        for r in results:
            print(f'  {r.cutoff_ev:>9.3f}'
                  f' {r.epsilon:>8.4f}+-{r.epsilon_sd:.4f}'
                  f' {r.p:>8.4f}+-{r.p_sd:.4f}'
                  f' {r.f:>8.4f}+-{r.f_sd:.4f}'
                  f' {r.eta:>8.4f}+-{r.eta_sd:.4f}'
                  f' {r.k_inf:>8.4f}+-{r.k_inf_sd:.4f}'
                  f' {r.P_NL:>8.4f}')
        print(f'  k_eff = {k_eff:.5f} +- {k_eff_sd:.5f}')
        print(bar + '\n')

    return results
