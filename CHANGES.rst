.. _release_notes:

=============
Release Notes
=============

.. 1.2.22 (unreleased)
   ===================


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
