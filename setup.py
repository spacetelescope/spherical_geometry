#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import shutil
import subprocess
import sys
import sysconfig
from glob import glob

import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

WINDOWS = sys.platform.startswith('win')
MACOS = sys.platform == 'darwin'
USE_SYSTEM_QD = bool(os.environ.get('USE_SYSTEM_QD'))
FREE_THREADED_PYTHON = bool(sysconfig.get_config_var('Py_GIL_DISABLED'))


# ---------------------------------------------------------------------------
# Compiler and linker flags
# ---------------------------------------------------------------------------

if WINDOWS:
    # MSVC: keep setuptools' defaults (/O2 /W3 /DNDEBUG).
    macros = [('_CRT_SECURE_NO_WARNINGS', None)]
    ldflags = []
    libqd_cflags = []
    math_util_cflags = []
else:
    macros = []
    ldflags = []
    common_cflags = ['-O3', '-DNDEBUG']

    if MACOS:
        common_cflags.append('-mmacosx-version-min=10.9')
        ldflags.append('-mmacosx-version-min=10.9')

    # Bundled libqd is third-party code: show warnings, but don't fail on them.
    libqd_cflags = common_cflags + ['-Wall']

    # Our own C code: strict.
    math_util_cflags = common_cflags + [
        '-Wall',
        '-Wextra',
        '-Werror',
        '-Wpedantic',
        '-Wno-unused-parameter',
        '-Wincompatible-pointer-types',
    ]


# ---------------------------------------------------------------------------
# The math_util extension
# ---------------------------------------------------------------------------

ext_info = {
    'include_dirs': [numpy.get_include(), 'src'],
    'library_dirs': [],
    'libraries': [],
    'define_macros': list(macros),
    'extra_compile_args': list(math_util_cflags),
    'extra_link_args': list(ldflags),
    # libqd is C++, so link with the C++ driver (math_util.c is still compiled as C).
    'language': 'c++',
}


# ---------------------------------------------------------------------------
# libqd: the bundled copy (default) or a system installation
# ---------------------------------------------------------------------------

libraries = []  # libraries for build_clib to build (not to link)

if not USE_SYSTEM_QD:
    # Build the bundled libqd as a static library with its own flags;
    # build_ext then links it into the extension automatically.
    # Not named "qd", so that a system libqd can't be linked by mistake.
    libraries.append(('qd_bundled', {
        'sources': sorted(glob('libqd/src/*.cpp')),
        'include_dirs': ['libqd/include'],
        'macros': macros,
        'cflags': libqd_cflags,
    }))
    ext_info['include_dirs'].append('libqd/include')

    if not WINDOWS:
        ext_info['libraries'].append('m')  # math library (part of the C runtime on Windows)

elif WINDOWS:
    # System libqd on Windows: QD_PATH must point to its install prefix.
    qd_path = os.environ.get('QD_PATH')
    if not qd_path:
        sys.exit('Set QD_PATH to the prefix where "qd" is installed '
                 '(it must contain lib\\ and include\\).')

    qd_path = os.path.abspath(qd_path)
    ext_info['include_dirs'].append(os.path.join(qd_path, 'include'))
    ext_info['library_dirs'].append(os.path.join(qd_path, 'lib'))
    ext_info['libraries'].append('qd')

else:
    # System libqd elsewhere: get the flags from qd-config.
    if not shutil.which('qd-config'):
        sys.exit('"qd-config" not found. Please install "qd" '
                 '(see: https://www.davidhbailey.com/dhbsoftware)')

    qd_cflags = subprocess.check_output(['qd-config', '--cflags'], text=True).split()
    qd_libs = subprocess.check_output(['qd-config', '--libs'], text=True).split()
    ext_info['extra_compile_args'] += qd_cflags
    ext_info['extra_link_args'] += qd_libs


# ---------------------------------------------------------------------------
# Stable ABI (abi3) for CPython >= 3.11; not supported on free-threaded builds
# ---------------------------------------------------------------------------

SETUPTOOLS_OPTIONS = {}

if not FREE_THREADED_PYTHON:
    ext_info['py_limited_api'] = True
    ext_info['define_macros'].append(('Py_LIMITED_API', '0x030B0000'))  # 3.11
    SETUPTOOLS_OPTIONS['bdist_wheel'] = {'py_limited_api': 'cp311'}


class BuildExtWithClib(build_ext):
    """Make sure the bundled static lib exists before linking the extension.

    ``build`` already runs ``build_clib`` before ``build_ext``, but a bare
    ``build_ext`` (e.g. ``--inplace`` / some editable-install paths) does not.
    """

    def run(self):
        if self.distribution.has_c_libraries():
            self.run_command('build_clib')
        super().run()


setup(
    libraries=libraries,
    ext_modules=[
        Extension(
            'spherical_geometry.math_util',
            [os.path.join('src', 'math_util.c')],
            **ext_info,
        )
    ],
    cmdclass={'build_ext': BuildExtWithClib},
    options=SETUPTOOLS_OPTIONS,
)
