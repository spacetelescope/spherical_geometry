#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys
import shutil
from glob import glob
from setuptools import setup
from setuptools import Extension
from setuptools import find_packages

use_system_qd = os.environ.get('USE_SYSTEM_QD', '')
have_windows = bool(sys.platform.startswith('win'))
have_darwin = bool(sys.platform == 'darwin')
have_linux = bool(sys.platform == 'linux')

qd_library_path = os.path.relpath(os.path.join('libqd'))
qd_library_c_path = os.path.join(qd_library_path, 'src')
qd_library_include_path = os.path.join(qd_library_path, 'include')
qd_sources = glob(os.path.join(qd_library_c_path, '*.cpp'))


def qd_config(arg):
    result = ''
    if not use_system_qd:
        if arg == 'libs':
            result = '' if have_windows else 'm'
        elif arg == 'cflags':
            result = qd_library_include_path
    else:
        if have_windows:
            qdpath = os.environ.get('QD_PATH', '')
            if not qdpath:
                print('WINDOWS USERS:\n\n'
                      'Define QD_PATH to the prefix where "qd" '
                      'is installed:\n\n'
                      'set QD_PATH="x:\\prefix\\of\\qd"\n\n'
                      'Expected directory structure of QD_PATH:\n'
                      '    QD_PATH\\lib\n'
                      '    QD_PATH\\include\n\n', file=sys.stderr)
                exit(1)

            qdpath = os.path.abspath(qdpath)
            if arg == 'libs':
                result = '/LIBPATH:' + os.path.join(qdpath, 'lib')
                result += ' qd.lib'
            elif arg == 'cflags':
                result = '-I' + os.path.join(qdpath, 'include')
            else:
                print('Unsupported option: {}'.format(arg), file=sys.stderr)
                exit(1)
        else:
            from subprocess import check_output
            if not shutil.which('qd-config'):
                print('"qd-config" not found. Please install "qd" '
                      '(see: https://www.davidhbailey.com/dhbsoftware)',
                      file=sys.stderr)
                exit(1)
            result = check_output(['qd-config'] + ['--' + arg]).decode().strip()

    return result.split()


try:
    import numpy
except ImportError:
    print('Missing requirement: numpy. Cannot continue.', file=sys.stderr)
    exit(1)

# Get some values from the setup.cfg
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

sources = [os.path.join('src', 'math_util.c')]

ext_info = {
    'include_dirs': [numpy.get_include(), 'src'],
    'libraries': [],
    'extra_link_args': [],
    'extra_compile_args': [],
    'define_macros': [],
}


if not use_system_qd:
    sources += qd_sources
    ext_info['libraries'] += qd_config('libs')
    ext_info['include_dirs'] += qd_config('cflags')
else:
    ext_info['extra_link_args'] += qd_config('libs')
    ext_info['extra_compile_args'] += qd_config('cflags')

if have_windows:
    ext_info['define_macros'] += [
        ('_CRT_SECURE_NO_WARNINGS', None),
    ]
elif have_darwin:
    ext_info['extra_link_args'] += [
        '-mmacosx-version-min=10.9'
    ]
    ext_info['extra_compile_args'] += [
        '-mmacosx-version-min=10.9'
    ]
    ext_info['language'] = 'c++'


setup(
    name=PACKAGENAME,
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description=DESCRIPTION,
    install_requires=[
        'astropy>=5.0.4',
        'numpy>=1.20',
    ],
    python_requiers='>=3.9',
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
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
