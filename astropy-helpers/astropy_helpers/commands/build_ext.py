import errno
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap

from distutils import log, ccompiler, sysconfig
from distutils.cmd import Command
from distutils.core import Extension
from distutils.ccompiler import get_default_compiler
from setuptools.command.build_ext import build_ext as SetuptoolsBuildExt

from ..utils import get_numpy_include_path, invalidate_caches, classproperty
from ..version_helpers import get_pkg_version_module


def should_build_with_cython(package, release=None):
    """Returns the previously used Cython version (or 'unknown' if not
    previously built) if Cython should be used to build extension modules from
    pyx files.  If the ``release`` parameter is not specified an attempt is
    made to determine the release flag from `astropy.version`.
    """

    try:
        version_module = __import__(package + '.cython_version',
                                    fromlist=['release', 'cython_version'])
    except ImportError:
        version_module = None

    if release is None and version_module is not None:
        try:
            release = version_module.release
        except AttributeError:
            pass

    try:
        cython_version = version_module.cython_version
    except AttributeError:
        cython_version = 'unknown'

    # Only build with Cython if, of course, Cython is installed, we're in a
    # development version (i.e. not release) or the Cython-generated source
    # files haven't been created yet (cython_version == 'unknown'). The latter
    # case can happen even when release is True if checking out a release tag
    # from the repository
    have_cython = False
    try:
        import Cython
        have_cython = True
    except ImportError:
        pass

    if have_cython and (not release or cython_version == 'unknown'):
        return cython_version
    else:
        return False


_compiler_versions = {}


def get_compiler_version(compiler):
    if compiler in _compiler_versions:
        return _compiler_versions[compiler]

    # Different flags to try to get the compiler version
    # TODO: It might be worth making this configurable to support
    # arbitrary odd compilers; though all bets may be off in such
    # cases anyway
    flags = ['--version', '--Version', '-version', '-Version',
             '-v', '-V']

    def try_get_version(flag):
        process = subprocess.Popen(
            shlex.split(compiler, posix=('win' not in sys.platform)) + [flag],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return 'unknown'

        output = stdout.strip().decode('latin-1')  # Safest bet
        if not output:
            # Some compilers return their version info on stderr
            output = stderr.strip().decode('latin-1')

        if not output:
            output = 'unknown'

        return output

    for flag in flags:
        version = try_get_version(flag)
        if version != 'unknown':
            break

    # Cache results to speed up future calls
    _compiler_versions[compiler] = version

    return version


# TODO: I think this can be reworked without having to create the class
# programmatically.
def generate_build_ext_command(packagename, release):
    """
    Creates a custom 'build_ext' command that allows for manipulating some of
    the C extension options at build time.  We use a function to build the
    class since the base class for build_ext may be different depending on
    certain build-time parameters (for example, we may use Cython's build_ext
    instead of the default version in distutils).

    Uses the default distutils.command.build_ext by default.
    """

    class build_ext(SetuptoolsBuildExt, object):
        package_name = packagename
        is_release = release
        _user_options = SetuptoolsBuildExt.user_options[:]
        _boolean_options = SetuptoolsBuildExt.boolean_options[:]
        _help_options = SetuptoolsBuildExt.help_options[:]

        force_rebuild = False

        _broken_compiler_mapping = [
            ('i686-apple-darwin[0-9]*-llvm-gcc-4.2', 'clang')
            ]

        # Warning: Spaghetti code ahead.
        # During setup.py, the setup_helpers module needs the ability to add
        # items to a command's user_options list.  At this stage we don't know
        # whether or not we can build with Cython, and so don't know for sure
        # what base class will be used for build_ext; nevertheless we want to
        # be able to provide a list to add options into.
        #
        # Later, once setup() has been called we should have all build
        # dependencies included via setup_requires available.  distutils needs
        # to be able to access the user_options as a *class* attribute before
        # the class has been initialized, but we do need to be able to
        # enumerate the options for the correct base class at that point

        @classproperty
        def user_options(cls):
            from distutils import core

            if core._setup_distribution is None:
                # We haven't gotten into setup() yet, and the Distribution has
                # not yet been initialized
                return cls._user_options

            return cls._final_class.user_options

        @classproperty
        def boolean_options(cls):
            # Similar to user_options above
            from distutils import core

            if core._setup_distribution is None:
                # We haven't gotten into setup() yet, and the Distribution has
                # not yet been initialized
                return cls._boolean_options

            return cls._final_class.boolean_options

        @classproperty
        def help_options(cls):
            # Similar to user_options above
            from distutils import core

            if core._setup_distribution is None:
                # We haven't gotten into setup() yet, and the Distribution has
                # not yet been initialized
                return cls._help_options

            return cls._final_class.help_options

        @classproperty(lazy=True)
        def _final_class(cls):
            """
            Late determination of what the build_ext base class should be,
            depending on whether or not Cython is available.
            """

            uses_cython = should_build_with_cython(cls.package_name,
                                                   cls.is_release)

            if uses_cython:
                # We need to decide late on whether or not to use Cython's
                # build_ext (since Cython may not be available earlier in the
                # setup.py if it was brought in via setup_requires)
                from Cython.Distutils import build_ext as base_cls
            else:
                base_cls = SetuptoolsBuildExt

            # Create and return an instance of a new class based on this class
            # using one of the above possible base classes
            def merge_options(attr):
                base = getattr(base_cls, attr)
                ours = getattr(cls, '_' + attr)

                all_base = set(opt[0] for opt in base)

                return base + [opt for opt in ours if opt[0] not in all_base]

            boolean_options = (base_cls.boolean_options +
                               [opt for opt in cls._boolean_options
                                if opt not in base_cls.boolean_options])

            members = dict(cls.__dict__)
            members.update({
                'user_options': merge_options('user_options'),
                'help_options': merge_options('help_options'),
                'boolean_options': boolean_options,
                'uses_cython': uses_cython,
            })

            # Update the base class for the original build_ext command
            build_ext.__bases__ = (base_cls, object)

            # Create a new class for the existing class, but now with the
            # appropriate base class depending on whether or not to use Cython.
            # Ensure that object is one of the bases to make a new-style class.
            return type(cls.__name__, (build_ext,), members)

        def __new__(cls, *args, **kwargs):
            # By the time the command is actually instantialized, the
            # Distribution instance for the build has been instantiated, which
            # means setup_requires has been processed--now we can determine
            # what base class we can use for the actual build, and return an
            # instance of a build_ext command that uses that base class (right
            # now the options being Cython.Distutils.build_ext, or the stock
            # setuptools build_ext)
            new_cls = super(build_ext, cls._final_class).__new__(
                    cls._final_class)

            # Since the new cls is not a subclass of the original cls, we must
            # manually call its __init__
            new_cls.__init__(*args, **kwargs)
            return new_cls

        def finalize_options(self):
            # Add a copy of the _compiler.so module as well, but only if there
            # are in fact C modules to compile (otherwise there's no reason to
            # include a record of the compiler used)
            # Note, self.extensions may not be set yet, but
            # self.distribution.ext_modules is where any extension modules
            # passed to setup() can be found
            self._adjust_compiler()

            extensions = self.distribution.ext_modules
            if extensions:
                src_path = os.path.relpath(
                    os.path.join(os.path.dirname(__file__), 'src'))
                shutil.copy2(os.path.join(src_path, 'compiler.c'),
                             os.path.join(self.package_name, '_compiler.c'))
                ext = Extension(self.package_name + '._compiler',
                                [os.path.join(self.package_name, '_compiler.c')])
                extensions.insert(0, ext)

            super(build_ext, self).finalize_options()

            # Generate
            if self.uses_cython:
                try:
                    from Cython import __version__ as cython_version
                except ImportError:
                    # This shouldn't happen if we made it this far
                    cython_version = None

                if (cython_version is not None and
                        cython_version != self.uses_cython):
                    self.force_rebuild = True
                    # Update the used cython version
                    self.uses_cython = cython_version

            # Regardless of the value of the '--force' option, force a rebuild
            # if the debug flag changed from the last build
            if self.force_rebuild:
                self.force = True

        def run(self):
            # For extensions that require 'numpy' in their include dirs,
            # replace 'numpy' with the actual paths
            np_include = get_numpy_include_path()
            for extension in self.extensions:
                if 'numpy' in extension.include_dirs:
                    idx = extension.include_dirs.index('numpy')
                    extension.include_dirs.insert(idx, np_include)
                    extension.include_dirs.remove('numpy')

                self._check_cython_sources(extension)

            super(build_ext, self).run()

            # Update cython_version.py if building with Cython
            try:
                cython_version = get_pkg_version_module(
                        packagename, fromlist=['cython_version'])[0]
            except (AttributeError, ImportError):
                cython_version = 'unknown'
            if self.uses_cython and self.uses_cython != cython_version:
                package_dir = os.path.relpath(packagename)
                cython_py = os.path.join(package_dir, 'cython_version.py')
                with open(cython_py, 'w') as f:
                    f.write('# Generated file; do not modify\n')
                    f.write('cython_version = {0!r}\n'.format(self.uses_cython))

                if os.path.isdir(self.build_lib):
                    # The build/lib directory may not exist if the build_py
                    # command was not previously run, which may sometimes be
                    # the case
                    self.copy_file(cython_py,
                                   os.path.join(self.build_lib, cython_py),
                                   preserve_mode=False)

                invalidate_caches()

        def _adjust_compiler(self):
            """
            This function detects broken compilers and switches to another.  If
            the environment variable CC is explicitly set, or a compiler is
            specified on the commandline, no override is performed -- the
            purpose here is to only override a default compiler.

            The specific compilers with problems are:

                * The default compiler in XCode-4.2, llvm-gcc-4.2,
                  segfaults when compiling wcslib.

            The set of broken compilers can be updated by changing the
            compiler_mapping variable.  It is a list of 2-tuples where the
            first in the pair is a regular expression matching the version of
            the broken compiler, and the second is the compiler to change to.
            """
            if 'CC' in os.environ:
                # Check that CC is not set to llvm-gcc-4.2
                c_compiler = os.environ['CC']

                try:
                    version = get_compiler_version(c_compiler)
                except OSError:
                    msg = textwrap.dedent(
                        """
                        The C compiler set by the CC environment variable:

                            {compiler:s}

                        cannot be found or executed.
                        """.format(compiler=c_compiler))
                    log.warn(msg)
                    sys.exit(1)

                for broken, fixed in self._broken_compiler_mapping:
                    if re.match(broken, version):
                        msg = textwrap.dedent(
                            """Compiler specified by CC environment variable
                            ({compiler:s}:{version:s}) will fail to compile
                            {pkg:s}.

                            Please set CC={fixed:s} and try again.
                            You can do this, for example, by running:

                                CC={fixed:s} python setup.py <command>

                            where <command> is the command you ran.
                            """.format(compiler=c_compiler, version=version,
                                       pkg=self.package_name, fixed=fixed))
                        log.warn(msg)
                        sys.exit(1)

                # If C compiler is set via CC, and isn't broken, we are good to go. We
                # should definitely not try accessing the compiler specified by
                # ``sysconfig.get_config_var('CC')`` lower down, because this may fail
                # if the compiler used to compile Python is missing (and maybe this is
                # why the user is setting CC). For example, the official Python 2.7.3
                # MacOS X binary was compiled with gcc-4.2, which is no longer available
                # in XCode 4.
                return

            if self.compiler is not None:
                # At this point, self.compiler will be set only if a compiler
                # was specified in the command-line or via setup.cfg, in which
                # case we don't do anything
                return

            compiler_type = ccompiler.get_default_compiler()

            if compiler_type == 'unix':
                # We have to get the compiler this way, as this is the one that is
                # used if os.environ['CC'] is not set. It is actually read in from
                # the Python Makefile. Note that this is not necessarily the same
                # compiler as returned by ccompiler.new_compiler()
                c_compiler = sysconfig.get_config_var('CC')

                try:
                    version = get_compiler_version(c_compiler)
                except OSError:
                    msg = textwrap.dedent(
                        """
                        The C compiler used to compile Python {compiler:s}, and
                        which is normally used to compile C extensions, is not
                        available. You can explicitly specify which compiler to
                        use by setting the CC environment variable, for example:

                            CC=gcc python setup.py <command>

                        or if you are using MacOS X, you can try:

                            CC=clang python setup.py <command>
                        """.format(compiler=c_compiler))
                    log.warn(msg)
                    sys.exit(1)

                for broken, fixed in self._broken_compiler_mapping:
                    if re.match(broken, version):
                        os.environ['CC'] = fixed
                        break

        def _check_cython_sources(self, extension):
            """
            Where relevant, make sure that the .c files associated with .pyx
            modules are present (if building without Cython installed).
            """

            # Determine the compiler we'll be using
            if self.compiler is None:
                compiler = get_default_compiler()
            else:
                compiler = self.compiler

            # Replace .pyx with C-equivalents, unless c files are missing
            for jdx, src in enumerate(extension.sources):
                base, ext = os.path.splitext(src)
                pyxfn = base + '.pyx'
                cfn = base + '.c'
                cppfn = base + '.cpp'

                if not os.path.isfile(pyxfn):
                    continue

                if self.uses_cython:
                    extension.sources[jdx] = pyxfn
                else:
                    if os.path.isfile(cfn):
                        extension.sources[jdx] = cfn
                    elif os.path.isfile(cppfn):
                        extension.sources[jdx] = cppfn
                    else:
                        msg = (
                            'Could not find C/C++ file {0}.(c/cpp) for Cython '
                            'file {1} when building extension {2}. Cython '
                            'must be installed to build from a git '
                            'checkout.'.format(base, pyxfn, extension.name))
                        raise IOError(errno.ENOENT, msg, cfn)

                # Current versions of Cython use deprecated Numpy API features
                # the use of which produces a few warnings when compiling.
                # These additional flags should squelch those warnings.
                # TODO: Feel free to remove this if/when a Cython update
                # removes use of the deprecated Numpy API
                if compiler == 'unix':
                    extension.extra_compile_args.extend([
                        '-Wp,-w', '-Wno-unused-function'])

    return build_ext
