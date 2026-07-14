import openmc.data
import numpy as np
import matplotlib.pyplot as plt

# Load the HDF5 file (already broadened)
H5_PATH_JEFF = '../../JanPiaskowy/MSc_Course/Thesis/Codes/CrossSections/jeff40_hdf5/H1.h5'
H5_PATH_TENDL = '../../JanPiaskowy/MSc_Course/Thesis/Codes/CrossSections/tendl2025_hdf5/H1.h5'
u233_jeff = openmc.data.IncidentNeutron.from_hdf5(H5_PATH_JEFF)
u233_tendl = openmc.data.IncidentNeutron.from_hdf5(H5_PATH_TENDL)

print(sorted(u233_jeff.reactions.keys()))

T = u233_jeff.temperatures[-10]

E_MIN, E_MAX = 0.01, 2_000_000.0

E = u233_jeff.energy[T]
mask = (E >= E_MIN) & (E <= E_MAX)
Em = E[mask]

# u233_jeff_fission = u233_jeff[3].xs[T](Em)
# u233_tendl_fission = u233_tendl[3].xs[T](Em)
#
# plt.figure(figsize=(9, 6))
# plt.loglog(Em, u233_jeff_fission, lw=0.8, label='JEFF-4.0')
# plt.loglog(Em, u233_tendl_fission, lw=0.8, label='TENDL2025', alpha=0.8)
#
# plt.xlabel('Energy [ev]')
# plt.ylabel('Elastic scattering cross section [b]')
# plt.title(f'$^{{233}}$Be (n,n0) at {T}')
# plt.legend()
# plt.grid(True, which='both', ls=':', alpha=0.4)
# plt.tight_layout()

u233_jeff_capture = u233_jeff[102].xs[T](Em)
u233_tendl_capture = u233_tendl[102].xs[T](Em)

plt.figure(figsize=(9, 6))
plt.loglog(Em, u233_jeff_capture, lw=0.8, label='JEFF-4.0')
plt.loglog(Em, u233_tendl_capture, lw=0.8, label='TENDL2025', alpha=0.8)

plt.xlabel('Energy [ev]')
plt.ylabel('Capture cross section [b]')
plt.title(f'$^{{233}}$U (n,gamma) at {T}')
plt.legend()
plt.grid(True, which='both', ls=':', alpha=0.4)
plt.tight_layout()
plt.show()





plt.show()

