"""Legacy shim -- the old `from GCR import ...` call sites keep working.

The code now lives in the gcr/ package.  New code should import from there:

    from gcr import GCRConfig, GCR
    from gcr.analysis.mass_estimate import print_u233_mass_estimate

This file exists only so that sweep_run.py, sensitivity_analysis.py,
find_critical_density.py and friends do not have to change on day one.
Migrate them at leisure, then delete this shim.
"""

from gcr.config import GCRConfig, ft_to_cm, in_to_cm          # noqa: F401
from gcr.model import GCR                                     # noqa: F401
from gcr.analysis.mass_estimate import print_u233_mass_estimate  # noqa: F401
