# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.pytest_plugins import *

## Uncomment the following line to treat all DeprecationWarnings as
## exceptions
import astropy
from astropy import __version__
if astropy.__version__ >= '0.4':
    enable_deprecations_as_exceptions()
