"""
Provides a base class for a 'dummy' setup.py command that has no functionality
(probably due to a missing requirement).  This dummy command can raise an
exception when it is run, explaining to the user what dependencies must be met
to use this command.

The reason this is at all tricky is that we want the command to be able to
provide this message even when the user passes arguments to the command.  If we
don't know ahead of time what arguments the command can take, this is
difficult, because distutils does not allow unknown arguments to be passed to a
setup.py command.  This hacks around that restriction to provide a useful error
message even when a user passes arguments to the dummy implementation of a
command.

Use this like:

    try:
        from some_dependency import SetupCommand
    except ImportError:
        from ._dummy import _DummyCommand

        class SetupCommand(_DummyCommand):
            description = \
                'Implementation of SetupCommand from some_dependency; '
                'some_dependency must be installed to run this command'

            # This is the message that will be raised when a user tries to
            # run this command--define it as a class attribute.
            error_msg = \
                "The 'setup_command' command requires the some_dependency "
                "package to be installed and importable."
"""

import sys
from setuptools import Command
from distutils.errors import DistutilsArgError
from textwrap import dedent


class _DummyCommandMeta(type):
    """
    Causes an exception to be raised on accessing attributes of a command class
    so that if ``./setup.py command_name`` is run with additional command-line
    options we can provide a useful error message instead of the default that
    tells users the options are unrecognized.
    """

    def __init__(cls, name, bases, members):
        if bases == (Command, object):
            # This is the _DummyCommand base class, presumably
            return

        if not hasattr(cls, 'description'):
            raise TypeError(
                "_DummyCommand subclass must have a 'description' "
                "attribute.")

        if not hasattr(cls, 'error_msg'):
            raise TypeError(
                "_DummyCommand subclass must have an 'error_msg' "
                "attribute.")

    def __getattribute__(cls, attr):
        if attr in ('description', 'error_msg'):
            # Allow cls.description to work so that `./setup.py
            # --help-commands` still works
            return super(_DummyCommandMeta, cls).__getattribute__(attr)

        raise DistutilsArgError(cls.error_msg)


if sys.version_info[0] < 3:
    exec(dedent("""
        class _DummyCommand(Command, object):
            __metaclass__ = _DummyCommandMeta
    """))
else:
    exec(dedent("""
        class _DummyCommand(Command, object, metaclass=_DummyCommandMeta):
            pass
    """))
