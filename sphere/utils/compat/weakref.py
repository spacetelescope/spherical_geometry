# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A replacement wrapper around the weakref module that adds WeakSet for
versions of python that are missing it.

Instead of importing weakref, other modules should use this as follows::

    from .utils.compat import weakref

"""
from __future__ import absolute_import

import weakref

# python2.7 and later provide a WeakSet class
if not hasattr(weakref, 'WeakSet'):
    from .weakrefset import WeakSet

from weakref import *
