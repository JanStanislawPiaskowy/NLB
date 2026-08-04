import openmc, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

with openmc.StatePoint("reference_runs/critical_nophoton/statepoint.250.h5") as sp:
    h, n_in = sp.entropy, sp.n_inactive

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(h, lw=0.8)
ax.axvline(n_in, ls="--", c="k", label=f"active batches begin ({n_in})")
ax.set_xlabel("batch"); ax.set_ylabel("Shannon entropy")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("entropy.pdf")
