import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openmc
from gcr.analysis.four_factors import compute_four_factors

statepoint_path = 'settings/statepoint.50.h5'

# sp = openmc.StatePoint(statepoint_path)

# kinetic_parameters = sp.get_kinetics_parameters()
# print(kinetic_parameters.generation_time)
# print(kinetic_parameters.beta_effective)
#
# print(sp.keff.nominal_value, sp.keff.std_dev)

# n_batches = sp.n_batches
# n_inactive = sp.n_inactive
# n_particles = sp.n_particles
#
# print(f'Simulation for {n_batches} batches ({n_inactive} inactive), each with {n_particles} particles')
#
#
# tallies_present = sp.tallies_present
# tallies = sp.tallies
# tally_derivatives = sp.tally_derivatives

# print('Tallies present:', tallies_present)
# print('Tallies:', tallies)
# print('Tally derivatives:', tally_derivatives)

print('here')
results = compute_four_factors(statepoint_path)

print('results')
#print(results)
for r in results:
    print(f'E_c = {r.cutoff_ev} eV:  p = {r.p:.4f} ± {r.p_sd:.4f},  k_inf = {r.k_inf:.4f}')



# sp.close()