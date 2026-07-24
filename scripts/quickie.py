import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openmc
from gcr.analysis.four_factors import compute_four_factors

statepoint_path = 'settings/statepoint.50.h5'

sp = openmc.StatePoint(statepoint_path)

tallies_present = sp.tallies_present
tallies = sp.tallies
tally_derivatives = sp.tally_derivatives

print('Tallies present:', tallies_present)
print('Tallies:', tallies)
print('Tally derivatives:', tally_derivatives)

t = sp.get_tally(name='unweighted_lifetime')
df = t.get_pandas_dataframe()
I = df.loc[df.score == 'inverse-velocity', 'mean'].item()
F = df.loc[df.score == 'nu-fission',       'mean'].item()
A = df.loc[df.score == 'absorption',       'mean'].item()
k = sp.keff.nominal_value

Lambda_unweighted = I / F           # generation time [s]
leakage           = F / k - A
ell_unweighted    = I / (F / k - A) # removal lifetime [s]

print('Lambda unweighted', Lambda_unweighted)
print('Leakage', leakage)
print('ell_unweighted', ell_unweighted)

kinetic_parameters = sp.get_kinetics_parameters()
Lambda = kinetic_parameters.generation_time
beta = kinetic_parameters.beta_effective
print(Lambda.nominal_value, Lambda.std_dev)
print(beta.nominal_value, beta.std_dev)

print(sp.keff.nominal_value, sp.keff.std_dev)

n_batches = sp.n_batches
n_inactive = sp.n_inactive
n_particles = sp.n_particles

print(f'Simulation for {n_batches} batches ({n_inactive} inactive), each with {n_particles} particles')
print('here')
results = compute_four_factors(statepoint_path)

print('results')
#print(results)
for r in results:
    print(f'E_c = {r.cutoff_ev} eV:  p = {r.p:.4f} ± {r.p_sd:.4f},  k_inf = {r.k_inf:.4f}')



sp.close()
