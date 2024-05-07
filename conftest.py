try:
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}

try:
    from spherical_geometry import __version__ as version
except ImportError:
    version = 'unknown'

# Uncomment and customize the following lines to add/remove entries
# from the list of packages for which version numbers are displayed
# when running the tests.
PYTEST_HEADER_MODULES['astropy'] = 'astropy'
PYTEST_HEADER_MODULES.pop('Scipy', None)
PYTEST_HEADER_MODULES.pop('Matplotlib', None)
PYTEST_HEADER_MODULES.pop('Pandas', None)
PYTEST_HEADER_MODULES.pop('h5py', None)

TESTED_VERSIONS['spherical-geometry'] = version


# This has to be in the root dir or it will not display in CI.
def pytest_report_header(config):
    import os

    from spherical_geometry import DISABLE_C_UFUNCS, HAS_C_UFUNCS

    # This gets added after the pytest-astropy-header output.
    return (
        f'CI: {os.environ.get("CI", "undefined")}\n'
        f'DISABLE_SPHR_GEOM_C_UFUNCS: {os.environ.get("DISABLE_SPHR_GEOM_C_UFUNCS", "undefined")}\n'
        f'DISABLE_C_UFUNCS: {DISABLE_C_UFUNCS}\n'
        f'HAS_C_UFUNCS: {HAS_C_UFUNCS}\n'
    )
