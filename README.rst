========================================
The astro.sphere package
========================================

The `astro.sphere` library is a pure Python package for handling spherical
polygons that represent arbitrary regions of the sky.

Coordinates in world space are traditionally represented by right
ascension and declination (*ra* and *dec*), or longitude and latitude.
While these representations are convenient, they have discontinuities
at the poles, making operations on them trickier at arbitrary
locations on the sky sphere.  Therefore, all internal operations of
this library are done in 3D vector space, where coordinates are
represented as (*x*, *y*, *z*) vectors.  The `stsci.sphere.vector` module
contains functions to convert between (*ra*, *dec*) and (*x*, *y*,
*z*) representations.

While any (*x*, *y*, *z*) triple represents a vector and therefore a
location on the sky sphere, a distinction must be made between
normalized coordinates fall exactly on the unit sphere, and
unnormalized coordinates which do not.  A normalized coordinate is
defined as a vector whose length is 1.

The library allows the user to work in either degrees or radians.  All
methods that require or return an angular value have a `degrees`
keyword argument.  When `degrees` is `True`, these measurements are in
degrees, otherwise they are in radians.

Spherical polygons
------------------

Spherical polygons are arbitrary areas on the sky sphere enclosed by
great circle arcs.  They are represented by the
`~astro.sphere.polygon.SphericalPolygon` class.

Once one has a `SphericalPolygon` object, there are a number of
operations available:

  - `~SphericalPolygon.contains_point`: Determines if the given point is inside the polygon.

  - `~SphericalPolygon.intersects_poly`: Determines if one polygon intersects with another.

  - `~SphericalPolygon.area`: Determine the area of a polygon.

  - `~SphericalPolygon.union` and `~SphericalPolygon.multi_union`:
    Return a new polygon that is the union of two or more polygons.

  - `~SphericalPolygon.intersection` and
    `~SphericalPolygon.multi_intersection`: Return a new polygon that
    is the intersection of two or more polygons.

  - `~SphericalPolygon.overlap`: Determine how much a given polygon
    overlaps another.

  - `~SphericalPolygon.to_radec`: Convert (*x*, *y*, *z*) points in the
    polygon to (*ra*, *dec*) points.

  - `~SphericalPolygon.same_points_as`: Determines if one polygon has the
    same points as another. When only sorted unique points are considered
    (default behavior), polygons with same points might not be the same
    polygons because the order of the points matter.

  - `~SphericalPolygon.draw`: Plots the polygon using matplotlib’s
    Basemap toolkit.  This feature is rather bare and intended
    primarily for debugging purposes.
