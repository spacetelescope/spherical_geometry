# -*- coding: utf-8 -*-

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
The `spherical_geometry.polygon` module defines the `SphericalPolygon` class for
managing polygons on the unit sphere.
"""
from __future__ import division, print_function, unicode_literals, absolute_import

# STDLIB
from copy import copy, deepcopy

# THIRD-PARTY
import numpy as np

# LOCAL
from . import great_circle_arc
from . import vector

__all__ = ['SphericalPolygon']


class _SingleSphericalPolygon(object):
    r"""
    Polygons are represented by both a set of points (in Cartesian
    (*x*, *y*, *z*) normalized on the unit sphere), and an inside
    point.  The inside point is necessary, because both the inside and
    outside of the polygon are finite areas on the great sphere, and
    therefore we need a way of specifying which is which.
    """

    def __init__(self, points, inside=None):
        r"""
        Parameters
        ----------
        points : An Nx3 array of (*x*, *y*, *z*) triples in vector space
            These points define the boundary of the polygon.  It must
            be "closed", i.e., the last point is the same as the first.

            It may contain zero points, in which it defines the null
            polygon.  It may not contain one, two or three points.
            Four points are needed to define a triangle, since the
            polygon must be closed.

        inside : An (*x*, *y*, *z*) triple, optional
            This point must be inside the polygon.  If not provided, the
            mean of the points will be used.
        """
        if len(points) == 0:
            # handle special case of initializing with an empty list of
            # vertices (ticket #1079).
            self._inside = np.zeros(3)
            self._points = np.asanyarray(points)
            return
        elif len(points) < 3:
            raise ValueError("Polygon made of too few points")
        else:
            assert np.array_equal(points[0], points[-1]), 'Polygon is not closed'

        self._points = points = np.asanyarray(points)

        if inside is None:
            self._inside = self._find_new_inside(points)
        else:
            self._inside = np.asanyarray(inside)

        # TODO: Detect self-intersection and fix

    def __copy__(self):
        return deepcopy(self)

    copy = __copy__

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__,
                               self.points, self.inside)

    @property
    def points(self):
        """
        The points defining the polygon.  It is an Nx3 array of
        (*x*, *y*, *z*) vectors.  The polygon will be explicitly
        closed, i.e., the first and last points are the same.
        """
        return self._points

    @property
    def inside(self):
        """
        Get the inside point of the polygon.
        """
        return self._inside

    def iter_polygons_flat(self):
        """
        Iterate over all base polygons that make up this multi-polygon
        set.
        """
        yield self

    def to_radec(self):
        """
        Convert `_SingleSphericalPolygon` footprint to RA and DEC.

        Returns
        -------
        ra, dec : list of float
            List of *ra* and *dec* in degrees corresponding
            to `points`.
        """
        if len(self.points) == 0:
            return np.array([])
        return vector.vector_to_radec(self.points[:,0], self.points[:,1],
                                      self.points[:,2], degrees=True)

    @classmethod
    def from_radec(cls, ra, dec, center=None, degrees=True):
        r"""
        Create a new `SphericalPolygon` from a list of (*ra*, *dec*)
        points.

        Parameters
        ----------
        ra, dec : 1-D arrays of the same length
            The vertices of the polygon in right-ascension and
            declination.  It must be \"closed\", i.e., that is, the
            last point is the same as the first.

        center : (*ra*, *dec*) pair, optional
            A point inside of the polygon to define its inside.  If no
            *center* point is provided, the mean of the polygon's
            points in vector space will be used.  That approach may
            not work for concave polygons.

        degrees : bool, optional
            If `True`, (default) *ra* and *dec* are in decimal degrees,
            otherwise in radians.

        Returns
        -------
        polygon : `SphericalPolygon` object
        """
        # Convert to Cartesian
        x, y, z = vector.radec_to_vector(ra, dec, degrees=degrees)

        points = np.dstack((x, y, z))[0]

        if center is None:
            center = points.mean(axis=0)
            vector.normalize_vector(center, output=center)
        else:
            center = vector.radec_to_vector(*center, degrees=degrees)

        return cls(points, center)

    @classmethod
    def from_cone(cls, ra, dec, radius, degrees=True, steps=16.0):
        r"""
        Create a new `_SingleSphericalPolygon` from a cone (otherwise known
        as a "small circle") defined using (*ra*, *dec*, *radius*).

        The cone is not represented as an ideal circle on the sphere,
        but as a series of great circle arcs.  The resolution of this
        conversion can be controlled using the *steps* parameter.

        Parameters
        ----------
        ra, dec : float scalars
            This defines the center of the cone

        radius : float scalar
            The radius of the cone

        degrees : bool, optional
            If `True`, (default) *ra*, *dec* and *radius* are in
            decimal degrees, otherwise in radians.

        steps : int, optional
            The number of steps to use when converting the small
            circle to a polygon.

        Returns
        -------
        polygon : `_SingleSphericalPolygon` object
        """
        u, v, w = vector.radec_to_vector(ra, dec, degrees=degrees)
        if degrees:
            radius = np.deg2rad(radius)

        # Get an arbitrary perpendicular vector.  This be be obtained
        # by crossing (u, v, w) with any unit vector that is not itself.
        which_min = np.argmin([u*u, v*v, w*w])
        if which_min == 0:
            perp = np.cross([u, v, w], [1., 0., 0.])
        elif which_min == 1:
            perp = np.cross([u, v, w], [0., 1., 0.])
        else:
            perp = np.cross([u, v, w], [0., 0., 1.])
        perp = vector.normalize_vector(perp)
        
        # Rotate by radius around the perpendicular vector to get the
        # "pen"
        x, y, z = vector.rotate_around(
            u, v, w, perp[0], perp[1], perp[2], radius, degrees=False)

        # Then rotate the pen around the center point all 360 degrees
        C = np.linspace(0, np.pi * 2.0, steps)
        # Ensure that the first and last elements are exactly the
        # same.  2π should equal 0, but with rounding error that isn't
        # always the case.
        C[-1] = 0
        C = C[::-1]
        X, Y, Z = vector.rotate_around(x, y, z, u, v, w, C, degrees=False)

        return cls(np.dstack((X, Y, Z))[0], (u, v, w))

    @classmethod
    def from_wcs(cls, fitspath, steps=1, crval=None):
        r"""
        Create a new `_SingleSphericalPolygon` from the footprint of a FITS
        WCS specification.

        This method requires having `astropy <http://astropy.org>`__
        installed.

        Parameters
        ----------
        fitspath : path to a FITS file, `astropy.io.fits.Header`, or `astropy.wcs.WCS`
            Refers to a FITS header containing a WCS specification.

        steps : int, optional
            The number of steps along each edge to convert into
            polygon edges.

        Returns
        -------
        polygon : `_SingleSphericalPolygon` object
        """
        from astropy import wcs as pywcs
        from astropy.io import fits

        if isinstance(fitspath, fits.Header):
            header = fitspath
            wcs = pywcs.WCS(header)
        elif isinstance(fitspath, pywcs.WCS):
            wcs = fitspath
        else:
            wcs = pywcs.WCS(fitspath)
        if crval is not None:
            wcs.wcs.crval = crval
        xa, ya = [wcs._naxis1, wcs._naxis2]

        length = steps * 4 + 1
        X = np.empty(length)
        Y = np.empty(length)

        # Now define each of the 4 edges of the quadrilateral
        X[0      :steps  ] = np.linspace(1, xa, steps, False)
        Y[0      :steps  ] = 1
        X[steps  :steps*2] = xa
        Y[steps  :steps*2] = np.linspace(1, ya, steps, False)
        X[steps*2:steps*3] = np.linspace(xa, 1, steps, False)
        Y[steps*2:steps*3] = ya
        X[steps*3:steps*4] = 1
        Y[steps*3:steps*4] = np.linspace(ya, 1, steps, False)
        X[-1]              = 1
        Y[-1]              = 1

        # Use wcslib to convert to (ra, dec)
        ra, dec = wcs.all_pix2world(X, Y, 1)

        # Convert to Cartesian
        x, y, z = vector.radec_to_vector(ra, dec)

        # Calculate an inside point
        ra, dec = wcs.all_pix2world(xa / 2.0, ya / 2.0, 1)
        xc, yc, zc = vector.radec_to_vector(ra, dec)

        return cls(np.dstack((x, y, z))[0], (xc, yc, zc))

    @classmethod
    def _contains_point(cls, point, P, r):
        point = np.asanyarray(point)

        intersects = great_circle_arc.intersects(P[:-1], P[1:], r, point)
        crossings = np.sum(intersects)

        return (crossings % 2) == 0

    def contains_point(self, point):
        r"""
        Determines if this `_SingleSphericalPolygon` contains a given point.

        Parameters
        ----------
        point : an (*x*, *y*, *z*) triple
            The point to test.

        Returns
        -------
        contains : bool
            Returns `True` if the polygon contains the given *point*.
        """
        return self._contains_point(point, self._points, self._inside)

    def intersects_poly(self, other):
        r"""
        Determines if this `_SingleSphericalPolygon` intersects another
        `_SingleSphericalPolygon`.

        This method is much faster than actually computing the
        intersection region between two polygons.

        Parameters
        ----------
        other : `_SingleSphericalPolygon`

        Returns
        -------
        intersects : bool
            Returns `True` if this polygon intersects the *other*
            polygon.

        Notes
        -----

        The algorithm proceeds as follows:

            1. Determine if any single point of one polygon is contained
               within the other.

            2. Deal with the case where only the edges overlap as in::

               :       o---------o
               :  o----+---------+----o
               :  |    |         |    |
               :  o----+---------+----o
               :       o---------o

               In this case, an edge from one polygon must cross an
               edge from the other polygon.
        """
        assert isinstance(other, _SingleSphericalPolygon)

        # The easy case is in which a point of one polygon is
        # contained in the other polygon.
        for point in other._points:
            if self.contains_point(point):
                return True
        for point in self._points:
            if other.contains_point(point):
                return True

        # The hard case is when only the edges overlap, as in:
        #
        #         o---------o
        #    o----+---------+----o
        #    |    |         |    |
        #    o----+---------+----o
        #         o---------o
        #
        for i in range(len(self._points) - 1):
            A = self._points[i]
            B = self._points[i+1]
            if np.any(great_circle_arc.intersects(
                A, B, other._points[:-1], other._points[1:])):
                return True
        return False

    def intersects_arc(self, a, b):
        """
        Determines if this `_SingleSphericalPolygon` intersects or contains
        the given arc.
        """
        P = self._points

        if self.contains_arc(a, b):
            return True

        intersects = great_circle_arc.intersects(P[:-1], P[1:], a, b)
        return np.any(intersects)

    def contains_arc(self, a, b):
        """
        Returns `True` if the polygon fully encloses the arc given by a
        and b.
        """
        return self.contains_point(a) and self.contains_point(b)

    def area(self):
        r"""
        Returns the area of the polygon on the unit sphere, in
        steradians.

        The area is computed using a generalization of Girard's Theorem.

        if :math:`\theta` is the sum of the internal angles of the
        polygon, and *n* is the number of vertices, the area is:

        .. math::

            S = \theta - (n - 2) \pi
        """
        if len(self._points) < 3:
            return np.array(0.0)

        points = self._points
        angles = np.hstack([
            great_circle_arc.angle(
                points[:-2], points[1:-1], points[2:], degrees=False),
            great_circle_arc.angle(
                points[-2], points[0], points[1], degrees=False)])
        sum = np.sum(angles) - (len(angles) - 2) * np.pi

        return sum

    def union(self, other):
        """
        Return a new `SphericalPolygon` that is the union of *self*
        and *other*.

        Parameters
        ----------
        other : `_SingleSphericalPolygon`

        Returns
        -------
        polygon : `SphericalPolygon` object

        See also
        --------
        SphericalPolygon.multi_union

        Notes
        -----
        For implementation details, see the
        :mod:`~spherical_geometry.graph` module.
        """
        from . import graph
        if len(self._points) < 3:
            return SphericalPolygon([other.copy()])
        elif len(other._points) < 3:
            return SphericalPolygon([self.copy()])

        g = graph.Graph([self, other])

        return g.union()

    @classmethod
    def _find_new_inside(cls, points):
        """
        Finds an acceptable inside point inside of *points* that is
        also inside of *polygons*.  Used by the intersection
        algorithm, and is really only useful in that context because
        it requires existing polygons with known inside points.
        """
        if len(points) < 4:
            return points[0]

        # Special case for a triangle
        if len(points) == 4:
            return np.sum(points[:3]) / 3.0

        candidates = []
        for i in range(len(points) - 1):
            A = points[i]
            B = points[i+1]
            C = points[(i+2) % (len(points) - 1)]
            angle = great_circle_arc.angle(A, B, C, degrees=False)
            if angle <= np.pi * 2.0:
                inside = great_circle_arc.midpoint(A, C)
                poly = _SingleSphericalPolygon(points, inside)
                candidates.append((poly.area(), list(inside)))

        candidates.sort()
        if candidates[0][0] > np.pi * 2:
            # Fallback to the mean
            return np.sum(points[:-1], axis=0) / (len(points) - 1)
        else:
            return np.array(candidates[0][1])

    def intersection(self, other):
        """
        Return a new `SphericalPolygon` that is the intersection of
        *self* and *other*.

        If the intersection is empty, a `SphericalPolygon` with zero
        subpolyons will be returned.

        Parameters
        ----------
        other : `_SingleSphericalPolygon`

        Returns
        -------
        polygon : `SphericalPolygon` object

        See also
        --------
        SphericalPolygon.multi_union

        Notes
        -----
        For implementation details, see the
        :mod:`~spherical_geometry.graph` module.
        """
        from . import graph
        if len(self._points) < 3 or len(other._points) < 3:
            return SphericalPolygon([])

        g = graph.Graph([self, other])

        return g.intersection()

    def overlap(self, other):
        r"""
        Returns the fraction of *self* that is overlapped by *other*.

        Let *self* be *a* and *other* be *b*, then the overlap is
        defined as:

        .. math::

            \frac{S_a}{S_{a \cap b}}

        Parameters
        ----------
        other : `_SingleSphericalPolygon`

        Returns
        -------
        frac : float
            The fraction of *self* that is overlapped by *other*.
        """
        s1 = self.area()
        intersection = self.intersection(other)
        s2 = intersection.area()
        return s2 / s1

    def draw(self, m, **plot_args):
        """
        Draws the polygon in a matplotlib.Basemap axes.

        Parameters
        ----------
        m : Basemap axes object

        **plot_args : Any plot arguments to pass to basemap
        """
        if not len(self._points):
            return
        if not len(plot_args):
            plot_args = {'color': 'blue'}
        points = self._points
        if 'alpha' in plot_args:
            del plot_args['alpha']

        alpha = 1.0
        for A, B in zip(points[0:-1], points[1:]):
            length = great_circle_arc.length(A, B, degrees=True)
            if not np.isfinite(length):
                length = 2
            interpolated = great_circle_arc.interpolate(A, B, length * 4)
            ra, dec = vector.vector_to_radec(
                interpolated[:, 0], interpolated[:, 1], interpolated[:, 2],
                degrees=True)
            for r0, d0, r1, d1 in zip(ra[0:-1], dec[0:-1], ra[1:], dec[1:]):
                m.drawgreatcircle(r0, d0, r1, d1, alpha=alpha, **plot_args)
            alpha -= 1.0 / len(points)

        ra, dec = vector.vector_to_radec(
            *self._inside, degrees=True)
        x, y = m(ra, dec)
        m.scatter(x, y, 1, **plot_args)


class SphericalPolygon(object):
    r"""
    Polygons are represented by both a set of points (in Cartesian
    (*x*, *y*, *z*) normalized on the unit sphere), and an inside
    point.  The inside point is necessary, because both the inside and
    outside of the polygon are finite areas on the great sphere, and
    therefore we need a way of specifying which is which.

    This class contains a list of disjoint closed polygons.
    """

    def __init__(self, init, inside=None):
        r"""
        Parameters
        ----------
        init : object
            May be either:
               - A list of disjoint `SphericalPolygon` objects.

               - An Nx3 array of (*x*, *y*, *z*) triples in Cartesian
                 space.  These points define the boundary of the
                 polygon.  It must be "closed", i.e., the last point
                 is the same as the first.

                 It may contain zero points, in which it defines the
                 null polygon.  It may not contain one, two or three
                 points.  Four points are needed to define a triangle,
                 since the polygon must be closed.

        inside : An (*x*, *y*, *z*) triple, optional
            If *init* is an array of points, this point must be inside
            the polygon.  If not provided, the mean of the points will
            be used.
        """
        for polygon in init:
            if not isinstance(polygon, (SphericalPolygon, _SingleSphericalPolygon)):
                break
        else:
            self._polygons = tuple(init)
            return

        self._polygons = (_SingleSphericalPolygon(init, inside),)

    def __copy__(self):
        return deepcopy(self)

    copy = __copy__

    def iter_polygons_flat(self):
        """
        Iterate over all base polygons that make up this multi-polygon
        set.
        """
        for polygon in self._polygons:
            for subpolygon in polygon.iter_polygons_flat():
                yield subpolygon

    @property
    def points(self):
        """
        The points defining the polygons.  It is an iterator over
        disjoint closed polygons, where each element is an Nx3 array
        of (*x*, *y*, *z*) vectors.  Each polygon is explicitly
        closed, i.e., the first and last points are the same.
        """
        for polygon in self.iter_polygons_flat():
            yield polygon.points

    @property
    def inside(self):
        """
        Iterate over the inside point of each of the polygons.
        """
        for polygon in self.iter_polygons_flat():
            yield polygon.inside

    @property
    def polygons(self):
        """
        Get a sequence of all of the subpolygons.  Each subpolygon may
        itself have subpolygons.  To get a flattened sequence of all
        base polygons, use `iter_polygons_flat`.
        """
        return self._polygons

    def to_radec(self):
        """
        Convert the `SphericalPolygon` footprint to RA and DEC
        coordinates.

        Returns
        -------
        polyons : iterator
            Each element in the iterator is a tuple of the form (*ra*,
            *dec*), where each is an array of points.
        """
        for polygon in self.iter_polygons_flat():
            yield polygon.to_radec()

    @classmethod
    def from_radec(cls, ra, dec, center=None, degrees=True):
        r"""
        Create a new `SphericalPolygon` from a list of (*ra*, *dec*)
        points.

        Parameters
        ----------
        ra, dec : 1-D arrays of the same length
            The vertices of the polygon in right-ascension and
            declination.  It must be \"closed\", i.e., that is, the
            last point is the same as the first.

        center : (*ra*, *dec*) pair, optional
            A point inside of the polygon to define its inside.  If no
            *center* point is provided, the mean of the polygon's
            points in vector space will be used.  That approach may
            not work for concave polygons.

        degrees : bool, optional
            If `True`, (default) *ra* and *dec* are in decimal degrees,
            otherwise in radians.

        Returns
        -------
        polygon : `SphericalPolygon` object
        """
        return cls([
            _SingleSphericalPolygon.from_radec(
                ra, dec, center=center, degrees=degrees)])

    @classmethod
    def from_cone(cls, ra, dec, radius, degrees=True, steps=16.0):
        r"""
        Create a new `SphericalPolygon` from a cone (otherwise known
        as a "small circle") defined using (*ra*, *dec*, *radius*).

        The cone is not represented as an ideal circle on the sphere,
        but as a series of great circle arcs.  The resolution of this
        conversion can be controlled using the *steps* parameter.

        Parameters
        ----------
        ra, dec : float scalars
            This defines the center of the cone

        radius : float scalar
            The radius of the cone

        degrees : bool, optional
            If `True`, (default) *ra*, *dec* and *radius* are in
            decimal degrees, otherwise in radians.

        steps : int, optional
            The number of steps to use when converting the small
            circle to a polygon.

        Returns
        -------
        polygon : `SphericalPolygon` object
        """
        return cls([
            _SingleSphericalPolygon.from_cone(
                ra, dec, radius, degrees=degrees, steps=steps)])

    @classmethod
    def from_wcs(cls, fitspath, steps=1, crval=None):
        r"""
        Create a new `SphericalPolygon` from the footprint of a FITS
        WCS specification.

        This method requires having `astropy <http://astropy.org>`__
        installed.

        Parameters
        ----------
        fitspath : path to a FITS file, `astropy.io.fits.Header`, or `astropy.wcs.WCS`
            Refers to a FITS header containing a WCS specification.

        steps : int, optional
            The number of steps along each edge to convert into
            polygon edges.

        Returns
        -------
        polygon : `SphericalPolygon` object
        """
        return cls([
            _SingleSphericalPolygon.from_wcs(
                fitspath, steps=steps, crval=crval)])

    def contains_point(self, point):
        r"""
        Determines if this `SphericalPolygon` contains a given point.

        Parameters
        ----------
        point : an (*x*, *y*, *z*) triple
            The point to test.

        Returns
        -------
        contains : bool
            Returns `True` if the polygon contains the given *point*.
        """
        for polygon in self.iter_polygons_flat():
            if polygon.contains_point(point):
                return True
        return False

    def intersects_poly(self, other):
        r"""
        Determines if this `SphericalPolygon` intersects another
        `SphericalPolygon`.

        This method is much faster than actually computing the
        intersection region between two polygons.

        Parameters
        ----------
        other : `SphericalPolygon`

        Returns
        -------
        intersects : bool
            Returns `True` if this polygon intersects the *other*
            polygon.
        """
        assert isinstance(other, SphericalPolygon)

        for polya in self.iter_polygons_flat():
            for polyb in other.iter_polygons_flat():
                if polya.intersects_poly(polyb):
                    return True
        return False

    def intersects_arc(self, a, b):
        """
        Determines if this `SphericalPolygon` intersects or contains
        the given arc.
        """
        for subpolygon in self.iter_polygons_flat():
            if subpolygon.intersects_arc(a, b):
                return True
        return False

    def contains_arc(self, a, b):
        """
        Returns `True` if the polygon fully encloses the arc given by a
        and b.
        """
        for subpolygon in self.iter_polygons_flat():
            if subpolygon.contains_arc(a, b):
                return True
        return False

    def area(self):
        r"""
        Returns the area of the polygon on the unit sphere in
        steradians.

        The area is computed using a generalization of Girard's Theorem.

        if :math:`\theta` is the sum of the internal angles of the
        polygon, and *n* is the number of vertices, the area is:

        .. math::

            S = \theta - (n - 2) \pi
        """
        area = 0.0
        for subpoly in self.iter_polygons_flat():
            area += subpoly.area()
        return area

    def union(self, other):
        """
        Return a new `SphericalPolygon` that is the union of *self*
        and *other*.

        Parameters
        ----------
        other : `SphericalPolygon`

        Returns
        -------
        polygon : `SphericalPolygon` object

        See also
        --------
        multi_union

        Notes
        -----
        For implementation details, see the :mod:`~spherical_geometry.graph`
        module.
        """
        from . import graph

        if self.area() == 0.0:
            return other.copy()
        elif other.area() == 0.0:
            return self.copy()

        g = graph.Graph(
            list(self.iter_polygons_flat()) +
            list(other.iter_polygons_flat()))

        return g.union()

    @classmethod
    def multi_union(cls, polygons):
        """
        Return a new `SphericalPolygon` that is the union of all of the
        polygons in *polygons*.

        Parameters
        ----------
        polygons : sequence of `SphericalPolygon`

        Returns
        -------
        polygon : `SphericalPolygon` object

        See also
        --------
        union
        """
        assert len(polygons)
        for polygon in polygons:
            assert isinstance(polygon, SphericalPolygon)

        from . import graph

        all_polygons = []
        for polygon in polygons:
            all_polygons.extend(polygon.iter_polygons_flat())

        g = graph.Graph(all_polygons)
        return g.union()

    def intersection(self, other):
        """
        Return a new `SphericalPolygon` that is the intersection of
        *self* and *other*.

        If the intersection is empty, a `SphericalPolygon` with zero
        subpolygons will be returned.

        Parameters
        ----------
        other : `SphericalPolygon`

        Returns
        -------
        polygon : `SphericalPolygon` object

        Notes
        -----
        For implementation details, see the
        :mod:`~spherical_geometry.graph` module.
        """
        from . import graph

        if self.area() == 0.0:
            return SphericalPolygon([])
        elif other.area() == 0.0:
            return SphericalPolygon([])

        g = graph.Graph(
            list(self.iter_polygons_flat()) +
            list(other.iter_polygons_flat()))

        return g.intersection()

    @classmethod
    def multi_intersection(cls, polygons, method='parallel'):
        """
        Return a new `SphericalPolygon` that is the intersection of
        all of the polygons in *polygons*.

        Parameters
        ----------
        polygons : sequence of `SphericalPolygon`

        method : 'parallel' or 'serial', optional
            Specifies the method that is used to perform the
            intersections:

               - 'parallel' (default): A graph is built using all of
                 the polygons, and the intersection operation is computed on
                 the entire thing globally.

               - 'serial': The polygon is built in steps by adding one
                 polygon at a time and computing the intersection at
                 each step.

            This option is provided because one may be faster than the
            other depending on context, but it primarily exposed for
            testing reasons.  Both modes should theoretically provide
            equivalent results.

        Returns
        -------
        polygon : `SphericalPolygon` object
        """
        assert len(polygons)
        for polygon in polygons:
            assert isinstance(polygon, SphericalPolygon)

        from . import graph

        all_polygons = []
        for polygon in polygons:
            if polygon.area() == 0.0:
                return SphericalPolygon([])
            all_polygons.extend(polygon.iter_polygons_flat())

        if method.lower() == 'parallel':
            g = graph.Graph(all_polygons)
            return g.intersection()
        elif method.lower() == 'serial':
            results = copy(all_polygons[0])
            for polygon in all_polygons[1:]:
                results = results.intersection(polygon)
                # If we have a null intersection already, we don't
                # need to go any further.
                if len(results.polygons) == 0:
                    return results
            return results
        else:
            raise ValueError("method must be 'parallel' or 'serial'")

    def overlap(self, other):
        r"""
        Returns the fraction of *self* that is overlapped by *other*.

        Let *self* be *a* and *other* be *b*, then the overlap is
        defined as:

        .. math::

            \frac{S_a}{S_{a \cap b}}

        Parameters
        ----------
        other : `SphericalPolygon`

        Returns
        -------
        frac : float
            The fraction of *self* that is overlapped by *other*.
        """
        s1 = self.area()
        intersection = self.intersection(other)
        s2 = intersection.area()
        return s2 / s1

    def draw(self, m, **plot_args):
        """
        Draws the polygon in a matplotlib.Basemap axes.

        Parameters
        ----------
        m : Basemap axes object

        **plot_args : Any plot arguments to pass to basemap
        """
        for polygon in self._polygons:
            polygon.draw(m, **plot_args)
