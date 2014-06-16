# -*- coding: utf-8 -*-

# Copyright (C) 2011 Association of Universities for Research in
# Astronomy (AURA)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     1. Redistributions of source code must retain the above
#       copyright notice, this list of conditions and the following
#       disclaimer.
#
#     2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials
#       provided with the distribution.
#
#     3. The name of AURA and its representatives may not be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""
The `polygon` module defines the `SphericalPolygon` class for managing
polygons on the unit sphere.
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


class SphericalPolygon(object):
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

        self._points = np.asanyarray(points)

        if inside is None:
            self._inside = np.mean(points[:-1], axis=0)
        else:
            self._inside = np.asanyarray(inside)

        # TODO: Detect self-intersection and fix

    def __copy__(self):
        return deepcopy(self)

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__,
                               self.points, self.inside)

    def copy(self):
        return self.__class__(self._points.copy(), self._inside.copy())

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

    def to_radec(self):
        """
        Convert `SphericalPolygon` footprint to RA and DEC.

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

        return cls(np.dstack((x, y, z))[0], center)

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
        u, v, w = vector.radec_to_vector(ra, dec, degrees=degrees)
        if degrees:
            radius = np.deg2rad(radius)

        # Get an arbitrary perpendicular vector.  This be be obtained
        # by crossing (u, v, w) with any unit vector that is not itself.
        which_min = np.argmin([u, v, w])
        if which_min == 0:
            perp = np.cross([u, v, w], [1., 0., 0.])
        elif which_min == 1:
            perp = np.cross([u, v, w], [0., 1., 0.])
        else:
            perp = np.cross([u, v, w], [0., 0., 1.])

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
        Create a new `SphericalPolygon` from the footprint of a FITS
        WCS specification.

        This method requires having `astropy` installed.

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

    def _unique_points(self):
        """
        Return a copy of `points` with duplicates removed.
        Order is preserved.

        .. note:: Output cannot be used to build a new
            polygon.
        """
        val = []
        for p in self.points:
            v = tuple(p)
            if v not in val:
                val.append(v)
        return np.array(val)

    def _sorted_points(self, preserve_order=True, unique=False):
        """
        Return a copy of `points` sorted such that smallest
        (*x*, *y*, *z*) is on top.

        .. note:: Output cannot be used to build a new
            polygon.

        Parameters
        ----------
        preserve_order : bool
            Preserve original order? If `True`, polygon is
            rotated around min point. If `False`, all points
            are sorted in ascending order.

        unique : bool
            Exclude duplicates.
        """
        if len(self.points) == 0:
            return []

        if unique:
            pts = self._unique_points()
        else:
            pts = self.points

        idx = np.lexsort((pts[:,0], pts[:,1], pts[:,2]))

        if preserve_order:
            i_min = idx[0]
            val = np.vstack([pts[i_min:], pts[:i_min]])
        else:
            val = pts[idx]

        return val

    def same_points_as(self, other, do_sort=True, thres=0.01):
        """
        Determines if this `SphericalPolygon` points are the same
        as the other. Number of points and areas are also compared.

        When `do_sort` is `True`, even when *self* and *other*
        have same points, they might not be equivalent because
        the order of the points defines the polygon.

        Parameters
        ----------
        other : `SphericalPolygon`

        do_sort : bool
            Compare sorted unique points.

        thres : float
            Fraction of area to use in equality decision.

        Returns
        -------
        is_eq : bool
            `True` or `False`.
        """
        self_n = len(self.points)

        if self_n != len(other.points):
            return False

        if self_n == 0:
            return True

        self_a = self.area()
        is_same_limit = thres * self_a

        if np.abs(self_a - other.area()) > is_same_limit:
            return False

        if do_sort:
            self_pts  = self._sorted_points(preserve_order=False, unique=True)
            other_pts = other._sorted_points(preserve_order=False, unique=True)
        else:
            self_pts  = self.points
            other_pts = other.points

        is_eq = True

        for self_p, other_p in zip(self_pts, other_pts):
            x_sum = 0.0

            for a,b in zip(self_p, other_p):
                x_sum += (a - b) ** 2

            if np.sqrt(x_sum) > is_same_limit:
                is_eq = False
                break

        return is_eq

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
        P = self._points
        r = self._inside
        point = np.asanyarray(point)

        intersects = great_circle_arc.intersects(P[:-1], P[1:], r, point)
        crossings = np.sum(intersects)

        return (crossings % 2) == 0

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
        assert isinstance(other, SphericalPolygon)

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
        Determines if this `SphericalPolygon` intersects or contains
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
        Returns the area of the polygon on the unit sphere.

        The algorithm is not able to compute the area of polygons
        that are larger than half of the sphere.  Therefore, the
        area will always be less than 2π.

        The area is computed by transforming the polygon to two
        dimensions using the `Lambert azimuthal equal-area projection
        <http://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection>`_

        .. math::

            X = \sqrt{\frac{2}{1-z}}x

        .. math::

            Y = \sqrt{\frac{2}{1-z}}y

        The individual great arc circle segments are interpolated
        before doing the transformation so that the curves are not
        straightened in the process.

        It then uses a standard 2D algorithm to compute the area.

        .. math::

            A = \left| \sum^n_{i=0} X_i Y_{i+1} - X_{i+1}Y_i \right|
        """
        if len(self._points) < 3:
            #return np.float64(0.0)
            return np.array(0.0)

        points = self._points.copy()

        # Rotate polygon so that center of polygon is at north pole
        centroid = np.mean(points[:-1], axis=0)
        centroid = vector.normalize_vector(centroid)
        points = self._points - (centroid + np.array([0, 0, 1]))
        vector.normalize_vector(points, output=points)

        XYs = []
        for A, B in zip(points[:-1], points[1:]):
            length = great_circle_arc.length(A, B, degrees=True)
            interp = great_circle_arc.interpolate(A, B, length * 4)
            vector.normalize_vector(interp, output=interp)
            XY = vector.equal_area_proj(interp)
            XYs.append(XY)

        XY = np.vstack(XYs)
        X = XY[..., 0]
        Y = XY[..., 1]

        return np.abs(np.sum(X[:-1] * Y[1:] - X[1:] * Y[:-1]) * 0.5 * np.pi)

    def union(self, other):
        """
        Return a new `SphericalPolygon` that is the union of *self*
        and *other*.

        If the polygons are disjoint, they result will be connected
        using cut lines.  For example::

            : o---------o
            : |         |
            : o---------o=====o----------o
            :                 |          |
            :                 o----------o

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
        For implementation details, see the :mod:`~sphere.graph`
        module.
        """
        from . import graph
        if len(self._points) < 3:
            return other.copy()
        elif len(other._points) < 3:
            return self.copy()

        g = graph.Graph([self, other])

        polygon = g.union()

        return self.__class__(polygon, self._inside)

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

        g = graph.Graph(polygons)
        polygon = g.union()
        return cls(polygon, polygons[0]._inside)

    @staticmethod
    def _find_new_inside(points, polygons):
        """
        Finds an acceptable inside point inside of *points* that is
        also inside of *polygons*.  Used by the intersection
        algorithm, and is really only useful in that context because
        it requires existing polygons with known inside points.
        """
        if len(points) < 4:
            return np.array([0, 0, 0])

        # Special case for a triangle
        if len(points) == 4:
            return np.sum(points[:3]) / 3.0

        for i in range(len(points) - 1):
            A = points[i]
            # Skip the adjacent point, since it is by definition on
            # the edge of the polygon, not potentially running through
            # the middle.
            for j in range(i + 2, len(points) - 1):
                B = points[j]
                C = great_circle_arc.midpoint(A, B)
                in_all = True
                for polygon in polygons:
                    if not polygon.contains_point(C):
                        in_all = False
                        break
                if in_all:
                    return C

        raise RuntimeError("Suitable inside point could not be found")

    def intersection(self, other):
        """
        Return a new `SphericalPolygon` that is the intersection of
        *self* and *other*.

        If the intersection is empty, a `SphericalPolygon` with zero
        points will be returned.

        If the result is disjoint, the pieces will be connected using
        cut lines.  For example::

            : o---------o
            : |         |
            : o---------o=====o----------o
            :                 |          |
            :                 o----------o

        Parameters
        ----------
        other : `SphericalPolygon`

        Returns
        -------
        polygon : `SphericalPolygon` object

        Notes
        -----
        For implementation details, see the :mod:`~sphere.graph`
        module.
        """
        # if not self.intersects_poly(other):
        #     return self.__class__([], [0, 0, 0])

        from . import graph
        if len(self._points) < 3 or len(other._points) < 3:
            return self.__class__([], [0, 0, 0])

        g = graph.Graph([self, other])

        polygon = g.intersection()

        inside = self._find_new_inside(polygon, [self, other])

        return self.__class__(polygon, inside)

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

        # for i in range(len(polygons)):
        #     polyA = polygons[i]
        #     for j in range(i + 1, len(polygons)):
        #         polyB = polygons[j]
        #         if not polyA.intersects_poly(polyB):
        #             return cls([], [0, 0, 0])

        from . import graph

        if method.lower() == 'parallel':
            g = graph.Graph(polygons)
            polygon = g.intersection()
            inside = cls._find_new_inside(polygon, polygons)
            return cls(polygon, inside)
        elif method.lower() == 'serial':
            result = copy(polygons[0])
            for polygon in polygons[1:]:
                result = result.intersection(polygon)
                # If we have a null intersection already, we don't
                # need to go any further.
                if len(result._points) < 3:
                    return result
            return result
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
        if not len(self._points):
            return
        if not len(plot_args):
            plot_args = {'color': 'blue'}
        points = self._points

        for A, B in zip(points[0:-1], points[1:]):
            length = great_circle_arc.length(A, B, degrees=True)
            if not np.isfinite(length):
                length = 2
            interpolated = great_circle_arc.interpolate(A, B, length * 4)
            ra, dec = vector.vector_to_radec(
                interpolated[:, 0], interpolated[:, 1], interpolated[:, 2],
                degrees=True)
            for r0, d0, r1, d1 in zip(ra[0:-1], dec[0:-1], ra[1:], dec[1:]):
                m.drawgreatcircle(r0, d0, r1, d1, **plot_args)

        ra, dec = vector.vector_to_radec(
            *self._inside, degrees=True)
        x, y = m(ra, dec)
        m.scatter(x, y, 1, **plot_args)
