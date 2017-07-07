# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, unicode_literals, print_function

from distutils.core import Extension
import os
import sys

from astropy_helpers import setup_helpers


def requires_2to3():
    return False


def get_extensions():
    ROOT = os.path.relpath(os.path.dirname(__file__))

    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['include_dirs'].append('numpy')

    sources = [str(os.path.join(ROOT, 'src', 'math_util.c'))]

    if (not setup_helpers.use_system_library('qd') or
        sys.platform == 'win32'):
        qd_library_path = os.path.join(ROOT, '..', 'cextern', 'qd-library')
        qd_library_c_path = os.path.join(qd_library_path, 'src')
        qd_library_include_path = os.path.join(qd_library_path, 'include')

        qd_sources = [
            'bits.cpp',
            'c_qd.cpp',
            'dd_real.cpp',
            'qd_real.cpp',
            'dd_const.cpp',
            'qd_const.cpp',
            'fpu.cpp',
            'util.cpp']

        sources.extend([
            str(os.path.join(qd_library_c_path, x))
            for x in qd_sources])
        cfg['include_dirs'].extend([
            qd_library_include_path,
            str(os.path.join(ROOT, 'src'))])
        if not sys.platform.startswith('win'):
            cfg['libraries'].append('m')
    else:
        cfg.update(setup_helpers.pkg_config([], ['qd', 'm'], 'qd-config'))

    return [Extension(
        str('spherical_geometry.math_util'), sources, **cfg)]


def get_external_libraries():
    return ['qd']
