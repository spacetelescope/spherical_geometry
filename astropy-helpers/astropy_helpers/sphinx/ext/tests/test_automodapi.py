# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys

import pytest

from . import *
from ....utils import iteritems

pytest.importorskip('sphinx')  # skips these tests if sphinx not present


class FakeConfig(object):
    """
    Mocks up a sphinx configuration setting construct for automodapi tests
    """
    def __init__(self, **kwargs):
        for k, v in iteritems(kwargs):
            setattr(self, k, v)


class FakeApp(object):
    """
    Mocks up a `sphinx.application.Application` object for automodapi tests
    """

    # Some default config values
    _defaults = {
        'automodapi_toctreedirnm': 'api',
        'automodapi_writereprocessed': False
    }

    def __init__(self, **configs):
        config = self._defaults.copy()
        config.update(configs)
        self.config = FakeConfig(**config)
        self.info = []
        self.warnings = []

    def info(self, msg, loc):
        self.info.append((msg, loc))

    def warn(self, msg, loc):
        self.warnings.append((msg, loc))


am_replacer_str = """
This comes before

.. automodapi:: astropy_helpers.sphinx.ext.tests.test_automodapi
{options}

This comes after
"""

am_replacer_basic_expected = """
This comes before

astropy_helpers.sphinx.ext.tests.test_automodapi Module
-------------------------------------------------------

.. automodule:: astropy_helpers.sphinx.ext.tests.test_automodapi

Functions
^^^^^^^^^

.. automodsumm:: astropy_helpers.sphinx.ext.tests.test_automodapi
    :functions-only:
    :toctree: api/

Classes
^^^^^^^

.. automodsumm:: astropy_helpers.sphinx.ext.tests.test_automodapi
    :classes-only:
    :toctree: api/

Class Inheritance Diagram
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automod-diagram:: astropy_helpers.sphinx.ext.tests.test_automodapi
    :private-bases:
    :parts: 1
    {empty}

This comes after
""".format(empty='')
# the .format is necessary for editors that remove empty-line whitespace


def test_am_replacer_basic():
    """
    Tests replacing an ".. automodapi::" with the automodapi no-option
    template
    """
    from ..automodapi import automodapi_replace

    fakeapp = FakeApp()
    result = automodapi_replace(am_replacer_str.format(options=''), fakeapp)

    assert result == am_replacer_basic_expected

am_replacer_noinh_expected = """
This comes before

astropy_helpers.sphinx.ext.tests.test_automodapi Module
-------------------------------------------------------

.. automodule:: astropy_helpers.sphinx.ext.tests.test_automodapi

Functions
^^^^^^^^^

.. automodsumm:: astropy_helpers.sphinx.ext.tests.test_automodapi
    :functions-only:
    :toctree: api/

Classes
^^^^^^^

.. automodsumm:: astropy_helpers.sphinx.ext.tests.test_automodapi
    :classes-only:
    :toctree: api/


This comes after
""".format(empty='')


def test_am_replacer_noinh():
    """
    Tests replacing an ".. automodapi::" with no-inheritance-diagram
    option
    """
    from ..automodapi import automodapi_replace

    fakeapp = FakeApp()
    ops = ['', ':no-inheritance-diagram:']
    ostr = '\n    '.join(ops)
    result = automodapi_replace(am_replacer_str.format(options=ostr), fakeapp)

    assert result == am_replacer_noinh_expected

am_replacer_titleandhdrs_expected = """
This comes before

astropy_helpers.sphinx.ext.tests.test_automodapi Module
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

.. automodule:: astropy_helpers.sphinx.ext.tests.test_automodapi

Functions
*********

.. automodsumm:: astropy_helpers.sphinx.ext.tests.test_automodapi
    :functions-only:
    :toctree: api/

Classes
*******

.. automodsumm:: astropy_helpers.sphinx.ext.tests.test_automodapi
    :classes-only:
    :toctree: api/

Class Inheritance Diagram
*************************

.. automod-diagram:: astropy_helpers.sphinx.ext.tests.test_automodapi
    :private-bases:
    :parts: 1
    {empty}


This comes after
""".format(empty='')


def test_am_replacer_titleandhdrs():
    """
    Tests replacing an ".. automodapi::" entry with title-setting and header
    character options.
    """
    from ..automodapi import automodapi_replace

    fakeapp = FakeApp()
    ops = ['', ':title: A new title', ':headings: &*']
    ostr = '\n    '.join(ops)
    result = automodapi_replace(am_replacer_str.format(options=ostr), fakeapp)

    assert result == am_replacer_titleandhdrs_expected


am_replacer_nomain_str = """
This comes before

.. automodapi:: astropy_helpers.sphinx.ext.automodapi
    :no-main-docstr:

This comes after
"""

am_replacer_nomain_expected = """
This comes before

astropy_helpers.sphinx.ext.automodapi Module
--------------------------------------------



Functions
^^^^^^^^^

.. automodsumm:: astropy_helpers.sphinx.ext.automodapi
    :functions-only:
    :toctree: api/


This comes after
""".format(empty='')


def test_am_replacer_nomain():
    """
    Tests replacing an ".. automodapi::" with "no-main-docstring" .
    """
    from ..automodapi import automodapi_replace

    fakeapp = FakeApp()
    result = automodapi_replace(am_replacer_nomain_str, fakeapp)

    assert result == am_replacer_nomain_expected


am_replacer_skip_str = """
This comes before

.. automodapi:: astropy_helpers.sphinx.ext.automodapi
    :skip: something1
    :skip: something2

This comes after
"""

am_replacer_skip_expected = """
This comes before

astropy_helpers.sphinx.ext.automodapi Module
--------------------------------------------

.. automodule:: astropy_helpers.sphinx.ext.automodapi

Functions
^^^^^^^^^

.. automodsumm:: astropy_helpers.sphinx.ext.automodapi
    :functions-only:
    :toctree: api/
    :skip: something1,something2


This comes after
""".format(empty='')


def test_am_replacer_skip():
    """
    Tests using the ":skip: option in an ".. automodapi::" .
    """
    from ..automodapi import automodapi_replace

    fakeapp = FakeApp()
    result = automodapi_replace(am_replacer_skip_str, fakeapp)

    assert result == am_replacer_skip_expected


am_replacer_invalidop_str = """
This comes before

.. automodapi:: astropy_helpers.sphinx.ext.automodapi
    :invalid-option:

This comes after
"""


def test_am_replacer_invalidop():
    """
    Tests that a sphinx warning is produced with an invalid option.
    """
    from ..automodapi import automodapi_replace

    fakeapp = FakeApp()
    automodapi_replace(am_replacer_invalidop_str, fakeapp)

    expected_warnings = [('Found additional options invalid-option in '
                          'automodapi.', None)]

    assert fakeapp.warnings == expected_warnings


am_replacer_cython_str = """
This comes before

.. automodapi:: apyhtest_eva.unit02
{options}

This comes after
"""

am_replacer_cython_expected = """
This comes before

apyhtest_eva.unit02 Module
--------------------------

.. automodule:: apyhtest_eva.unit02

Functions
^^^^^^^^^

.. automodsumm:: apyhtest_eva.unit02
    :functions-only:
    :toctree: api/

This comes after
""".format(empty='')


def test_am_replacer_cython(cython_testpackage):
    """
    Tests replacing an ".. automodapi::" for a Cython module.
    """

    from ..automodapi import automodapi_replace

    fakeapp = FakeApp()
    result = automodapi_replace(am_replacer_cython_str.format(options=''),
                                fakeapp)

    assert result == am_replacer_cython_expected
