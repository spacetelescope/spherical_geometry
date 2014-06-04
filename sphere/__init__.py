# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Package for managing polygons on the unit sphere.
"""

from ._astropy_init import *
import sys

if sys.version_info[0] >= 3:
    # Python 3 compatibility
    __builtins__['xrange'] = range

