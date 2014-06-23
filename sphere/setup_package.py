# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, unicode_literals, print_function

from distutils.core import Extension
import os


def requires_2to3():
    return False


def get_extensions():
    ROOT = os.path.relpath(os.path.dirname(__file__))

    return [Extension(
        str('sphere.math_util'),
        [str(os.path.join(ROOT, 'src/math_util.c'))])]
