#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys

from setuptools import setup
from setuptools import Extension
from setuptools import find_packages

try:
    import numpy
except ImportError:
    print('Missing requirement: numpy. Cannot continue.', file=sys.stderr)
    exit(1)

# Get some values from the setup.cfg
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser
conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', 'Astropy affiliated package')
AUTHOR = metadata.get('author', '')
AUTHOR_EMAIL = metadata.get('author_email', '')
LICENSE = metadata.get('license', 'unknown')
URL = metadata.get('url', 'https://github.com/spacetelescope')

# Include all .c files, recursively, including those generated by
# Cython, since we can not do this in MANIFEST.in with a "dynamic"
# directory name.
c_files = []
for root, dirs, files in os.walk(PACKAGENAME):
    for filename in files:
        if filename.endswith('.c'):
            c_files.append(
                os.path.join(
                    os.path.relpath(root, PACKAGENAME), filename))

ext_info = {
    'include_dirs': [numpy.get_include()],
    'libraries': ['m'],
    'define_macros': [],
}

sources = [
    os.path.join('src', 'math_util.c')
]

qd_library_path = os.path.join('cextern', 'qd-library')
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

ext_info['include_dirs'].extend([
    qd_library_include_path,
    'src'])

if sys.platform.startswith('win'):
    # no math library on Windows
    ext_info['libraries'] = []
    ext_info['define_macros'] += [
        ('_CRT_SECURE_NO_WARNING', None),
    ]


setup(
    name=PACKAGENAME,
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description=DESCRIPTION,
    install_requires=[
        'astropy',
        'numpy',
    ],
    extras_require={
        'test': [
            'pytest',
        ],
        'docs': [
            'sphinx',
            'sphinx-automodapi',
            'numpydoc',
        ],
    },
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    zip_safe=False,
    packages=find_packages(),
    package_data={
        '': ['README.rst', 'licenses/*'],
        PACKAGENAME: [
            os.path.join(PACKAGENAME, '*'),
        ]
    },
    ext_modules=[
        Extension('spherical_geometry.math_util', sources, **ext_info)
    ],
)
