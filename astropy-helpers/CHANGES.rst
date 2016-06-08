astropy-helpers Changelog
=========================

1.2 (unreleased)
----------------

- Added sphinx configuration value ``automodsumm_inherited_members``.
  If ``True`` this will include members that are inherited from a base
  class in the generated API docs. Defaults to ``False`` which matches
  the previous behavior. [#215]

- Fixed ``build_sphinx`` to recognize builds that succeeded but have output
  *after* the "build succeeded." statement. This only applies when
  ``--warnings-returncode`` is  given (which is primarily relevant for Travis
  documentation builds).  [#223]

- Fixed ``build_sphinx`` the sphinx extensions to not output a spurious warning
  for sphinx versions > 1.4. [#229]

- Add Python version dependent local sphinx inventories that contain
  otherwise missing references. [#216]

- ``astropy_helpers`` now require Sphinx 1.3 or later. [#226]

1.1.2 (2016-03-9)
-----------------

- The CSS for the sphinx documentation was altered to prevent some text overflow
  problems. [#217]


1.1.1 (2015-12-23)
------------------

- Fixed crash in build with ``AttributeError: cython_create_listing`` with
  older versions of setuptools. [#209]


1.1 (2015-12-10)
----------------

- The original ``AstropyTest`` class in ``astropy_helpers``, which implements
  the ``setup.py test`` command, is deprecated in favor of moving the
  implementation of that command closer to the actual Astropy test runner in
  ``astropy.tests``.  Now a dummy ``test`` command is provided solely for
  informing users that they need ``astropy`` installed to run the tests
  (however, the previous, now deprecated implementation is still provided and
  continues to work with older versions of Astropy). See the related issue for
  more details. [#184]

- Added a useful new utility function to ``astropy_helpers.utils`` called
  ``find_data_files``.  This is similar to the ``find_packages`` function in
  setuptools in that it can be used to search a package for data files
  (matching a pattern) that can be passed to the ``package_data`` argument for
  ``setup()``.  See the docstring to ``astropy_helpers.utils.find_data_files``
  for more details. [#42]

- The ``astropy_helpers`` module now sets the global ``_ASTROPY_SETUP_``
  flag upon import (from within a ``setup.py``) script, so it's not necessary
  to have this in the ``setup.py`` script explicitly.  If in doubt though,
  there's no harm in setting it twice.  Putting it in ``astropy_helpers``
  just ensures that any other imports that occur during build will have this
  flag set. [#191]

- It is now possible to use Cython as a ``setup_requires`` build requirement,
  and still build Cython extensions even if Cython wasn't available at the
  beginning of the build processes (that is, is automatically downloaded via
  setuptools' processing of ``setup_requires``). [#185]

- Moves the ``adjust_compiler`` check into the ``build_ext`` command itself,
  so it's only used when actually building extension modules.  This also
  deprecates the stand-alone ``adjust_compiler`` function. [#76]

- When running the ``build_sphinx`` / ``build_docs`` command with the ``-w``
  option, the output from Sphinx is streamed as it runs instead of silently
  buffering until the doc build is complete. [#197]

1.0.7 (unreleased)
------------------

- Fix missing import in ``astropy_helpers/utils.py``. [#196]

1.0.6 (2015-12-04)
------------------

- Fixed bug where running ``./setup.py build_sphinx`` could return successfully
  even when the build was not successful (and should have returned a non-zero
  error code). [#199]


1.0.5 (2015-10-02)
------------------

- Fixed a regression in the ``./setup.py test`` command that was introduced in
  v1.0.4.


1.0.4 (2015-10-02)
------------------

- Fixed issue with the sphinx documentation css where the line numbers for code
  blocks were not aligned with the code. [#179]

- Fixed crash that could occur when trying to build Cython extension modules
  when Cython isn't installed. Normally this still results in a failed build,
  but was supposed to provide a useful error message rather than crash
  outright (this was a regression introduced in v1.0.3). [#181]

- Fixed a crash that could occur on Python 3 when a working C compiler isn't
  found. [#182]

- Quieted warnings about deprecated Numpy API in Cython extensions, when
  building Cython extensions against Numpy >= 1.7. [#183]

- Improved support for py.test >= 2.7--running the ``./setup.py test`` command
  now copies all doc pages into the temporary test directory as well, so that
  all test files have a "common root directory". [#189]


1.0.3 (2015-07-22)
------------------

- Added workaround for sphinx-doc/sphinx#1843, a but in Sphinx which
  prevented descriptor classes with a custom metaclass from being documented
  correctly. [#158]

- Added an alias for the ``./setup.py build_sphinx`` command as
  ``./setup.py build_docs`` which, to a new contributor, should hopefully be
  less cryptic. [#161]

- The fonts in graphviz diagrams now match the font of the HTML content. [#169]

- When the documentation is built on readthedocs.org, MathJax will be
  used for math rendering.  When built elsewhere, the "pngmath"
  extension is still used for math rendering. [#170]

- Fix crash when importing astropy_helpers when running with ``python -OO``
  [#171]

- The ``build`` and ``build_ext`` stages now correctly recognize the presence
  of C++ files in Cython extensions (previously only vanilla C worked). [#173]


1.0.2 (2015-04-02)
------------------

- Various fixes enabling the astropy-helpers Sphinx build command and
  Sphinx extensions to work with Sphinx 1.3. [#148]

- More improvement to the ability to handle multiple versions of
  astropy-helpers being imported in the same Python interpreter session
  in the (somewhat rare) case of nested installs. [#147]

- To better support high resolution displays, use SVG for the astropy
  logo and linkout image, falling back to PNGs for browsers that
  support it. [#150, #151]

- Improve ``setup_helpers.get_compiler_version`` to work with more compilers,
  and to return more info.  This will help fix builds of Astropy on less
  common compilers, like Sun C. [#153]

1.0.1 (2015-03-04)
------------------

- Released in concert with v0.4.8 to address the same issues.

0.4.8 (2015-03-04)
------------------

- Improved the ``ah_bootstrap`` script's ability to override existing
  installations of astropy-helpers with new versions in the context of
  installing multiple packages simultaneously within the same Python
  interpreter (e.g. when one package has in its ``setup_requires`` another
  package that uses a different version of astropy-helpers. [#144]

- Added a workaround to an issue in matplotlib that can, in rare cases, lead
  to a crash when installing packages that import matplotlib at build time.
  [#144]

1.0 (2015-02-17)
----------------

- Added new pre-/post-command hook points for ``setup.py`` commands.  Now any
  package can define code to run before and/or after any ``setup.py`` command
  without having to manually subclass that command by adding
  ``pre_<command_name>_hook`` and ``post_<command_name>_hook`` callables to
  the package's ``setup_package.py`` module.  See the PR for more details.
  [#112]

- The following objects in the ``astropy_helpers.setup_helpers`` module have
  been relocated:

  - ``get_dummy_distribution``, ``get_distutils_*``, ``get_compiler_option``,
    ``add_command_option``, ``is_distutils_display_option`` ->
    ``astropy_helpers.distutils_helpers``

  - ``should_build_with_cython``, ``generate_build_ext_command`` ->
    ``astropy_helpers.commands.build_ext``

  - ``AstropyBuildPy`` -> ``astropy_helpers.commands.build_py``

  - ``AstropyBuildSphinx`` -> ``astropy_helpers.commands.build_sphinx``

  - ``AstropyInstall`` -> ``astropy_helpers.commands.install``

  - ``AstropyInstallLib`` -> ``astropy_helpers.commands.install_lib``

  - ``AstropyRegister`` -> ``astropy_helpers.commands.register``

  - ``get_pkg_version_module`` -> ``astropy_helpers.version_helpers``

  - ``write_if_different``, ``import_file``, ``get_numpy_include_path`` ->
    ``astropy_helpers.utils``

  All of these are "soft" deprecations in the sense that they are still
  importable from ``astropy_helpers.setup_helpers`` for now, and there is
  no (easy) way to produce deprecation warnings when importing these objects
  from ``setup_helpers`` rather than directly from the modules they are
  defined in.  But please consider updating any imports to these objects.
  [#110]

- Use of the ``astropy.sphinx.ext.astropyautosummary`` extension is deprecated
  for use with Sphinx < 1.2.  Instead it should suffice to remove this
  extension for the ``extensions`` list in your ``conf.py`` and add the stock
  ``sphinx.ext.autosummary`` instead. [#131]


0.4.7 (2015-02-17)
------------------

- Fixed incorrect/missing git hash being added to the generated ``version.py``
  when creating a release. [#141]


0.4.6 (2015-02-16)
------------------

- Fixed problems related to the automatically generated _compiler
  module not being created properly. [#139]


0.4.5 (2015-02-11)
------------------

- Fixed an issue where ah_bootstrap.py could blow up when astropy_helper's
  version number is 1.0.

- Added a workaround for documentation of properties in the rare case
  where the class's metaclass has a property of the same name. [#130]

- Fixed an issue on Python 3 where importing a package using astropy-helper's
  generated version.py module would crash when the current working directory
  is an empty git repository. [#114]

- Fixed an issue where the "revision count" appended to .dev versions by
  the generated version.py did not accurately reflect the revision count for
  the package it belongs to, and could be invalid if the current working
  directory is an unrelated git repository. [#107]

- Likewise, fixed a confusing warning message that could occur in the same
  circumstances as the above issue. [#121]


0.4.4 (2014-12-31)
------------------

- More improvements for building the documentation using Python 3.x. [#100]

- Additional minor fixes to Python 3 support. [#115]

- Updates to support new test features in Astropy [#92, #106]


0.4.3 (2014-10-22)
------------------

- The generated ``version.py`` file now preserves the git hash of installed
  copies of the package as well as when building a source distribution.  That
  is, the git hash of the changeset that was installed/released is preserved.
  [#87]

- In smart resolver add resolution for class links when they exist in the
  intersphinx inventory, but not the mapping of the current package
  (e.g. when an affiliated package uses an astropy core class of which
  "actual" and "documented" location differs) [#88]

- Fixed a bug that could occur when running ``setup.py`` for the first time
  in a repository that uses astropy-helpers as a submodule:
  ``AttributeError: 'NoneType' object has no attribute 'mkdtemp'`` [#89]

- Fixed a bug where optional arguments to the ``doctest-skip`` Sphinx
  directive were sometimes being left in the generated documentation output.
  [#90]

- Improved support for building the documentation using Python 3.x. [#96]

- Avoid error message if .git directory is not present. [#91]


0.4.2 (2014-08-09)
------------------

- Fixed some CSS issues in generated API docs. [#69]

- Fixed the warning message that could be displayed when generating a
  version number with some older versions of git. [#77]

- Fixed automodsumm to work with new versions of Sphinx (>= 1.2.2). [#80]


0.4.1 (2014-08-08)
------------------

- Fixed git revision count on systems with git versions older than v1.7.2.
  [#70]

- Fixed display of warning text when running a git command fails (previously
  the output of stderr was not being decoded properly). [#70]

- The ``--offline`` flag to ``setup.py`` understood by ``ah_bootstrap.py``
  now also prevents git from going online to fetch submodule updates. [#67]

- The Sphinx extension for converting issue numbers to links in the changelog
  now supports working on arbitrary pages via a new ``conf.py`` setting:
  ``changelog_links_docpattern``.  By default it affects the ``changelog``
  and ``whatsnew`` pages in one's Sphinx docs. [#61]

- Fixed crash that could result from users with missing/misconfigured
  locale settings. [#58]

- The font used for code examples in the docs is now the
  system-defined ``monospace`` font, rather than ``Minaco``, which is
  not available on all platforms. [#50]


0.4 (2014-07-15)
----------------

- Initial release of astropy-helpers.  See `APE4
  <https://github.com/astropy/astropy-APEs/blob/master/APE4.rst>`_ for
  details of the motivation and design of this package.

- The ``astropy_helpers`` package replaces the following modules in the
  ``astropy`` package:

  - ``astropy.setup_helpers`` -> ``astropy_helpers.setup_helpers``

  - ``astropy.version_helpers`` -> ``astropy_helpers.version_helpers``

  - ``astropy.sphinx`` - > ``astropy_helpers.sphinx``

  These modules should be considered deprecated in ``astropy``, and any new,
  non-critical changes to those modules will be made in ``astropy_helpers``
  instead.  Affiliated packages wishing to make use those modules (as in the
  Astropy package-template) should use the versions from ``astropy_helpers``
  instead, and include the ``ah_bootstrap.py`` script in their project, for
  bootstrapping the ``astropy_helpers`` package in their setup.py script.
