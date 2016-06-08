from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings

from .commands.test import AstropyTest


# Leaving this module here for now, but really it needn't exist
# (and it's doubtful that any code depends on it anymore)
warnings.warn('The astropy_helpers.test_helpers module is deprecated as '
              'of version 1.1.0; the AstropyTest command can be found in '
              'astropy_helpers.commands.test.', DeprecationWarning)
