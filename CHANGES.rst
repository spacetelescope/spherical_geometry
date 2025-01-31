.. _release_notes:

=============
Release Notes
=============


1.3.3 (31-January-2025)
=======================

- Fixed a bug in the C implementation of ``dot_qd`` that may result in
  "invalid value encountered in length" runtime warning in Python 3.13
  when input vectors contain NaN values. [#288]


1.3.2 (12-June-2024)
====================

- Fixed a bug in the Python implementation of ``inner1d``. [#265]

- Removed the ``degree`` argument from the ``length`` and ``angle`` functions
  in the ``spherical_geometry.great_circle_arc`` module as it was never working
  correctly with the ``math_util`` module and add unnecessary
  complication. [#265]

- Build wheel with Numpy 2.0 release candidate ahead of Numpy 2.0 release [#279]


1.3.1 (5-November-2023)
=======================

- Updated the built-in ``qd`` library to version 2.3.24. [#258]


1.3.0 (1-November-2023)
=======================

- Add documentation to ``polygon.py`` for the ``SphericalPolygon``
  method ``multi_union`` has exponential time behavior and cannot
  be used for a large number of polynomials [#229]

- Added code to ``SphericalPolygon.multi_union()`` to pre-process
  polygon list and remove nearly identical polygons to avoid a bug in the
  ``graph`` code that could cause a crash when some of the input polygons are
  nearly identical. This workaround can be undone in the future once
  https://github.com/spacetelescope/spherical_geometry/issues/232
  is resolved. [#233]

- Removed unused ``spherical_geometry.utils`` module. [#239]

- Minimum supported version for Python is now 3.9. [#239]

- Minimum supported version for NumPy is now 1.20. [#239]

- Wheels for several common architectures are now available on PyPI. [#243]

- Replaced private ``numpy.core._umath_tests.inner1d()`` with an alternative
  implementation. [#253]

- Fixed a bug in the ``math_util.angles()`` function that would result in crash
  when one of the input vectors is a null vector. [#254]

- Enhanced the code in the ``SphericalPolygon.convex_hull()`` to ignore points
  that lead to null vectors in computations. [#254]

- Fixed a bug in quad-precision functions ``normalize`` and
  ``intersection`` functions affecting numerous other functions. [#256]


1.2.23 (10-October-2022)
========================

- Pin astropy min version to 5.0.4. [#218]

- Set Python min version to 3.8. [#219]

- Fix problem reported where angle of 180 degrees results in an
  error with the C version of the code. [#223]

- Use a different algorithm for square root in the ``qd`` library that
  seems to be less prone to accuracy loss. This helps the bug in the
  ``math_util.c`` due to which ``angle()`` returns a NaN for
  coplanar vectors where the angle between the surface points should be
  180 degrees and enhances the solution from #223. Also, use ``qd`` epsilon
  instead of arbitrary value for rounding error check. [#224]


1.2.22 (04-January-2022)
========================

- Fixed segmentation fault occuring in the ``math_util`` module when
  ``spherical_geometry`` is used with ``numpy`` version ``1.22`` due to
  incorrect ``ntypes`` value being used when defining generalized
  universal functions. [#216]


1.2.21 (30-December-2021)
=========================

Maintenance release.


1.2.20 (12-May-2021)
====================

- Added a new exception class ``MalformedPolygonError`` which replaces
  ``ValueError`` which replaced ``AssertionError`` in the
  ``Graph._sanity_check()`` method. Also replaced ``ValueError`` with
  ``RuntimeError`` in ``Graph.Edge.follow()`` method. Added error
  messages to some other exceptions. [#207]

- Updated CI/CD to use github actions, instead of Travis CI.

- To comply with ``bandit`` all ``assert`` statements were changed to standard
  ``if`` checks and raising the appropriate errors.


1.2.17 (15-August-2019)
=======================

- Fixed unexpected behavior if using pip on Windows


1.2.16 (13-August-2019)
=======================

- Updated qd library to ``2.3.22``

- Reimplement the ability to link to system's qd (``export USE_SYSTEM_QD=1``)


1.2.15 (31-July-2019)
=====================

- Fixes compilation issue under Windows


1.2.13 (30-July-2019)
=====================

- Undo-ing changes from release 1.2.11 (PRs #170 and #173) due to crashes
  in the union of multi-polygons until a better solution is found. [#176]


1.2.12 (27-June-2019)
=====================

- Package structure was updated.


1.2.11 (24-June-2019)
=====================

- Increase the dimension of the confusion region in which two nodes are
  considered to be equal. This reduces the likelihood of crashes when
  computing intersections of nearly identical polygons. [#170, #173]


1.2.10 (01-March-2019)
======================

- Fix incorrect query of astropy version information to deal with
  deprecation of ``_naxis1`` and ``_naxis2`` WCS attributes. [#165]


1.2.9 (22-February-2019)
========================

- Fixed a bug introduced in version ``1.2.8``.


1.2.8 (22-February-2019)
========================

- Add backwards compatibility with ``astropy 3.1.0``: ``<3.1.0`` uses
  ``wcs._naxis`` and ``>=3.1.0`` uses ``wcs.pixel_shape``.


1.2.7 (14-November-2018)
========================

- Restored ``_naxis``, ``pixel_shape`` not ready yet.


1.2.6 (13-November-2018)
========================

- Replaced ``_naxis`` with ``pixel_shape``.

- Updated ``README`` file.

- Removed debugging code from ``graph.py``.


1.2.5 (10-July-2018)
====================

- Added a method to create a polygon from the convex hull of a list
  of points.


1.2.4 (28-June-2018)
====================

- The public methods in ``SingleSphericalPolygon`` now match the methods in
  ``SphericalPolygon`` so that objects of either type can be used
  interchangably (for the most part.) ``SphericalPolygon`` now subclasses
  ``SingleSphericalPolygon``.


1.2.3 (20-June-2018)
====================

- Every method with ``lonlat`` in its name now has an alias with ``lonlat``
  replaced by ``radec``.

- The class ``_SingleSphericalPolygon`` has been renamed to
  ``SingleSphericalPolygon``. The former name has been retained as an alias.

- The from_lonlat (and from_radec) method is now available in
  ``SingleSphericalPolygon`` as well as ``SphericalPolygon``.

- The methods ``iter_polygons_flat`` have been renamed to to ``__iter__``. The
  former name has been retained as an alias.
