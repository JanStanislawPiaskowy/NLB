"""GCR -- OpenMC model of the seven-cavity NLB gas-core nuclear rocket reactor.

Quick start:

    from gcr import GCRConfig, GCR

    config = GCRConfig(cross_sections_dir='libraries_xs/jeff40_hdf5')
    core = GCR(config)
    core.build()                 # -> openmc.Model
    core.add_power_tally()
    core.run()

GCRConfig is importable without OpenMC installed (handy for config editing
and the pure-maths tests); GCR itself is imported lazily on first access.
"""

from .config import GCRConfig

__all__ = ['GCRConfig', 'GCR']


def __getattr__(name):
    # Lazy import so that `import gcr` (and the pure-numpy hexmaths tests)
    # work on machines without OpenMC.
    if name == 'GCR':
        from .model import GCR
        return GCR
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
