"""
Post processing code that analyses the shannon entropy tally.

July 2026, J.S. Piaskowy
"""

import numpy as np
import openmc

def convergence_report(statepoint_path, tol=0.005, tail_fraction=0.25):

    with openmc.StatePoint(statepoint_path) as sp:
        H = np.asarray(sp.entropy, dtype=float)
        k_gen = np.asarray(sp.k_generation, dtype=float)
        n_in = sp.n_inactive
        k, k_sd = sp.keff.nominal_value, sp.keff.std_dev

    if H.size == 0:
        raise RuntimeError(
                'No entropy in this statepoint.'
                )

    tail = max(10, int(tail_faction * H.size)) # get only the last values
    H_ref = H[-tail:].mean()
    inside = np.abs(H - H_ref) < tol * abs(H_ref)

    n_from_i = np.cumsum(inside[::-1])[::-1]
    n_conv = int(np.argmax(n_from_i == np.arrange(H.size, 0, -1)))


    # accounting for non-independence
    a = k_gen[n_in:] - k_gen[n_in:].mean()
    rho = float(a[:-1] @ a[1:] / (a @ a))
    infl = float(np.sqrt((1 + rho) / (1 - rho)))

    return_dict = {'n_converged': n_conv,
                   'n_inactive_used': n_in,
                   'H_plateau': float(H_ref),
                   'dominance_ratio_set': rho,
                   'sigma_inflation': infl,
                   'sigma_k_reported_pcm': 1e5 * k_sd,
                   'sigma_k_corrected_pcm': 1e5 * k_sd * infl}

    return return_dict

if __name__ == '__main__':
    statepoint_path = ''

    values = convergence_report(statepoint_path)

    for val in values:
        print(val)
