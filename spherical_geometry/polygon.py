# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The `spherical_geometry.polygon` module defines the `SphericalPolygon` class for
managing polygons on the unit sphere.
"""

# STDLIB
from copy import deepcopy

# THIRD-PARTY
import numpy as np

# LOCAL
from spherical_geometry import great_circle_arc, vector

__all__ = ['SingleSphericalPolygon', 'SphericalPolygon',
           'MalformedPolygonError']


class MalformedPolygonError(Exception):
    pass


class SingleSphericalPolygon(object):
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
            These points define the boundary of the polygon.

            It may contain zero points, in which it defines the null
            polygon.  It may not contain one, two or three points.
            Four points are needed to define a triangle, since the
            polygon must be closed.

        inside : An (*x*, *y*, *z*) triple, optional
            This point must be inside the polygon.  If not provided, an
            interior point will be calculated.  If the polygon is degenerate,
            the inside point will be set to `None`.
        """
        if points is None or len(points) == 0:
            # handle special case of initializing with an empty list of
            # vertices (ticket #1079).
            self._degenerate = True
            self._inside = None
            self._points = np.asanyarray([])
            return
        else:
            self._degenerate = False

        if not np.array_equal(points[0], points[-1]):
            points = list(points[:])
            points.append(points[0])

        self._points = points = np.asanyarray(points)

        if len(points) < 4:
            self._inside = None
            raise ValueError("Polygon made of too few points")

        orient, new_inside = self._get_orient(compute_inside=True)
        if orient is None:
            self._degenerate = True
            self._inside = None
            return

        if inside is None:
            self._inside = np.asanyarray(new_inside)

            if not orient:
                self._points = points[::-1]
        else:
            self._inside = np.asanyarray(inside)

            if self.contains_point(new_inside) != orient:
                self._points = points[::-1]

    def __copy__(self):
        return deepcopy(self)

    copy = __copy__

    def __len__(self):
        return 1

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__,
                               self.points, self.inside)

    def __iter__(self):
        """
        Iterate over all base polygons that make up this multi-polygon
        set.
        """
        yield self

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

    @staticmethod
    def _is_on_great_circle(vertices, tol=1e-11):
        """
        vertices: (N, 3) array of unit vectors
        tol: threshold for smallest eigenvalue ratio
        """
        cov = vertices.T @ vertices  # 3x3 symmetric covariance-like matrix

        # Eigenvalues sorted descending
        w = np.linalg.eigvalsh(cov)  # TODO: improve accuracy by using qd library
        w1, _, w3 = w  # ascending order (w_i >= 0 since cov is positive semidefinite)

        # The eigenvalues of cov give us a measure of how much the points
        # deviate from lying in a plane through the origin. If the
        # smallest eigenvalue is tiny relative to the largest, the points
        # lie almost in a plane through the origin, which means they lie
        # almost on a great circle.

        # If the whole configuration is tiny or collapsed, treat as degenerate.
        if w3 < tol:
            return True

        # If the smallest eigenvalue is tiny relative to the largest,
        # the points lie almost in a plane through the origin.
        return w1 < tol * w3

    # TODO: Not sure if we want to make this a public method since in the
    #       future we might not allow instantiation of degenerate polygons
    #       at all. For more info, see:
    # https://github.com/spacetelescope/spherical_geometry/pull/316#discussion_r2834501132
    #
    # def is_degenerate(self):
    #     """
    #     Return `True` if the polygon is degenerate (i.e., has no points,
    #     or is otherwise invalid).
    #     """
    #     # If we want to compute this at runtime, we can check whether
    #     # the points are all on a great circle (too expensive to do in general,
    #     # so we compute it at initialization and store the result).
    #     return self._degenerate

    def _get_orient(self, compute_inside=False):
        npoints = len(self._points)
        if npoints < 4:
            return None, None

        points = np.vstack((self._points, self._points[1]))
        A = points[:-2]
        B = points[1:-1]
        C = points[2:]

        orient = great_circle_arc.triple_product(A - B, C - B, B)
        clockwise = np.sum(orient) > 0.0

        if (np.max(np.abs(orient)) < 1e-11 and
                SingleSphericalPolygon._is_on_great_circle(
                    self._points, tol=1e-16)):
            # The points are all on a great circle, so the polygon is
            # degenerate and we can't determine if it's clockwise or
            # counter-clockwise. Return None to indicate this.
            clockwise = None

        if compute_inside:
            if clockwise is None:
                inside = None
            elif len(self._points) > 4:
                orient2 = orient if clockwise else -1.0 * orient
                midpoint = great_circle_arc.midpoint(A, C)
                candidate = max(zip(orient2, midpoint), key=lambda x: x[0])
                inside = candidate[1]
            else:
                # Fall back on computing the mean point
                inside = self._points.mean(axis=0)
                vector.normalize_vector(inside, output=inside)
        else:
            inside = None

        return clockwise, inside

    def _find_new_inside(self):
        """
        Finds an acceptable inside point inside of *points* that is
        also inside of *polygons*.
        """
        _, inside = self._get_orient(compute_inside=True)
        return inside

    def _find_new_outside(self):
        """
        Finds an acceptable point outside of the polygon
        """
        tagged_points = []
        points = self._points[:-1]

        # Compute the minimum distance between all polygon points
        # and each antipode to a polygon point
        for point in points:
            point = -1.0 * point
            dot = great_circle_arc.inner1d(point, points)
            tag = np.amax(dot)
            tagged_points.append((tag, point))

        # find the antipode with the maximum distance
        # to any polygon point. It is our outside point.
        (tag, point) = min(tagged_points, key=lambda p: p[0])
        return point

    def is_clockwise(self):
        """
        Return `True` if the points in this polygon are in clockwise order.
        Return `None` if the points are all on a great circle.
        """
        clockwise, _ = self._get_orient(compute_inside=False)
        return clockwise

    def to_lonlat(self):
        """
        Convert `SingleSphericalPolygon` footprint to longitude and latitude.

        Returns
        -------
        lon, lat : list of float
            List of *lon* and *lat* in degrees corresponding
            to `points`.
        """
        if len(self.points) == 0:
            return np.array([])
        return vector.vector_to_lonlat(self.points[:, 0], self.points[:, 1],
                                       self.points[:, 2], degrees=True)

    # Alias for to_lonlat
    to_radec = to_lonlat

    @classmethod
    def from_lonlat(cls, lon, lat, center=None, degrees=True):
        r"""
        Create a new `SingleSphericalPolygon` from a list of
        (*longitude*, *latitude*) points.

        Parameters
        ----------
        lon, lat : 1-D arrays of the same length
            The vertices of the polygon in longitude and
            latitude.

        center : (*lon*, *lat*) pair, optional
            A point inside of the polygon to define its inside.

        degrees : bool, optional
            If `True`, (default) *lon* and *lat* are in decimal degrees,
            otherwise in radians.

        Returns
        -------
        polygon : `SingleSphericalPolygon` object
        """
        # Convert to Cartesian
        x, y, z = vector.lonlat_to_vector(lon, lat, degrees=degrees)

        points = np.dstack((x, y, z))[0]

        if center is not None:
            center = vector.lonlat_to_vector(*center, degrees=degrees)

        return cls(points, center)

    # from_radec is an alias for from_lon_lat
    from_radec = from_lonlat

    @classmethod
    def from_cone(cls, lon, lat, radius, degrees=True, steps=16):
        r"""
        Create a new `SingleSphericalPolygon` from a cone (otherwise known
        as a "small circle") defined using (*lon*, *lat*, *radius*).

        The cone is not represented as an ideal circle on the sphere,
        but as a series of great circle arcs.  The resolution of this
        conversion can be controlled using the *steps* parameter.

        Parameters
        ----------
        lon, lat : float scalars
            This defines the center of the cone

        radius : float scalar
            The radius of the cone

        degrees : bool, optional
            If `True`, (default) *lon*, *lat* and *radius* are in
            decimal degrees, otherwise in radians.

        steps : int, optional
            The number of steps to use when converting the small
            circle to a polygon.

        Returns
        -------
        polygon : `SingleSphericalPolygon` object
        """
        u, v, w = vector.lonlat_to_vector(lon, lat, degrees=degrees)
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
    def from_wcs(cls, wcs, steps: int = 1) -> "SingleSphericalPolygon":
        r"""Create a `SingleSphericalPolygon` from image footprint - the
        intersection of the image shape given by ``wcs.array_shape`` and
        the ``wcs.bounding_box`` if available) - converted to a world
        coordinate system.

        If the number of edges per side is set to 1, the polygon will be
        rectangular. Otherwise, the polygon will capture WCS distortion along
        the edges of the footprint.

        This method requires `astropy <http://astropy.org>`__ installed.

        Parameters
        ----------
        wcs: astropy.wcs.WCS | astropy.io.fits.Header | str :
            any WCS object that implements the common WCS API
        steps: int, optional :
            The number of edges to create along each side of the polygon.
            (Default value = 1)

        Returns
        -------
        polygon : `SingleSphericalPolygon` object
        """
        import astropy

        if isinstance(wcs, astropy.io.fits.Header | str):
            wcs = astropy.wcs.WCS(wcs)

        if (bbox_attr := getattr(wcs, "bounding_box", None)) is not None:
            if isinstance(
                bbox_attr,
                astropy.modeling.bounding_box.ModelBoundingBox
            ):
                bbox = bbox_attr.bounding_box(order="F")
            else:
                bbox = tuple(bbox_attr)

            xmin, xmax = bbox[0]
            ymin, ymax = bbox[1]
            width = xmax - xmin
            height = ymax - ymin

            if (shape := getattr(wcs, "array_shape", None)) is not None:
                # clip (intersect) the bounding box with the array shape
                # if it is available, to avoid having edge vertices outside
                # of the image footprint.
                xmin = max(xmin, -0.5)
                ymin = max(ymin, -0.5)
                xmax = min(xmax, shape[1] - 0.5)
                ymax = min(ymax, shape[0] - 0.5)

        elif (shape := getattr(wcs, "array_shape", None)) is not None:
            height, width = shape
            xmin = ymin = -0.5
            xmax = width - 0.5
            ymax = height - 0.5

        else:
            raise ValueError(
                "Unable to infer footprint from WCS: the WCS object must have "
                "either a 'bounding_box' or an 'array_shape' property set."
            )

        # constrain number of vertices to the maximum number
        # of pixels on an edge:
        steps = max(1, min(int(steps), int(width), int(height)))

        # build a list of pixel indices that represent
        # equally-spaced edge vertices:
        x0 = np.repeat(xmin, steps)
        y0 = np.repeat(ymin, steps)
        x1 = np.repeat(xmax, steps)
        y1 = np.repeat(ymax, steps)
        x = np.linspace(xmin, xmax, num=steps, endpoint=False)
        y = np.linspace(ymin, ymax, num=steps, endpoint=False)

        # define each of the 4 edges of the quadrilateral
        vertices = np.concatenate(
            [
                # south edge
                np.stack([x, y0], axis=1),
                # east edge
                np.stack([x1, y], axis=1),
                # north edge
                np.stack([x1 - x + x0, y1], axis=1),
                # west edge
                np.stack([x0, y1 - y + y0], axis=1),
            ],
            axis=0,
        )

        # ensure bounding box is None because, due to rounding errors,
        # edge vertices may be outside of the bounding box, which would
        # result in ``NaN`` values when converting to sky coordinates
        if hasattr(wcs, "bounding_box"):
            wcs.bounding_box = None

        # convert the pixel indices into sky coordinates using the WCS
        try:
            vertex_skycoords = wcs.pixel_to_world(*vertices.T)
            xc = xmin + width / 2
            yc = ymin + height / 2
            center_skycoord = wcs.pixel_to_world(xc, yc)
            center = center_skycoord.ra.degree, center_skycoord.dec.degree
        finally:
            # restore the original bounding box if it was set, since we do not
            # want to have side effects on the WCS object passed in by the user
            if bbox_attr is not None:
                wcs.bounding_box = bbox_attr

        # pass the sky coordinates to a new polygon object as degrees
        return cls.from_lonlat(
            vertex_skycoords.ra.degree,
            vertex_skycoords.dec.degree,
            center=center
        )

    @classmethod
    def convex_hull(cls, points):
        """
        Create a new `SingleSphericalPolygon` from the convex hull of a
        list of points using the Graham Scan algorithm

        Parameters
        ----------
        points: A list of points on the unit sphere

        Returns
        -------
        polygon : `SingleSphericalPolygon` object
        """
        points = np.asarray(points)

        # Find an extremal point, it must be on boundary

        j = np.argmin(np.arctan2(points[:,1], points[:,0]))
        extreme = points[j,:]
        points = np.vstack((points[0:j,:], points[j+1:,:]))

        # Sort points around extreme point by angle from true north

        north = [0., 0., 1.]
        ang = great_circle_arc.angle(north, extreme, points)
        pt = [points[i,:] for i in range(points.shape[0])]

        duo = list(zip(pt, ang))
        duo = sorted(duo, key=lambda d: d[1])
        points = np.asarray([d[0] for d in duo if np.isfinite(d[1])])

        # Set the first point on the hull to the extreme point

        pbottom = extreme
        hull = [pbottom]

        # If a point is to the left of previous points on the
        # hull, add it to the hull. If to the right, the top
        # point is an inside point and is popped off the hull.
        # See any description of the Graham Scan algorithm
        # for a more detailed explanation.

        i = 0
        inside = None
        while i < points.shape[0]:
            ptop = hull[-1]
            if ptop is pbottom:
                hull.append(points[i,:])
                i += 1
            else:
                pprevious = hull[-2]
                if great_circle_arc.triple_product(pprevious, ptop,
                                                   points[i,:]) > 0.0:
                    hull.append(points[i,:])
                    i += 1
                else:
                    inside = hull.pop()

        # Create a polygon from points on the hull

        return cls(hull, inside)

    def invert_polygon(self):
        """
        Compute the inverse (complement) of a single polygon.
        Returns `None` if the polygon is degenerate.
        """
        poly = self.copy()
        if self._degenerate:
            return None
        poly._points = poly._points[::-1]
        poly._inside = np.asanyarray(self._find_new_outside())
        return poly

    def _contains_point(self, point, P, r):
        if self._degenerate or point is None:
            return None
        point = np.asanyarray(point)
        if np.array_equal(r, point):
            return True

        intersects = great_circle_arc.intersects(P[:-1], P[1:], r, point)
        crossings = np.sum(intersects)
        return (crossings % 2) == 0

    def contains_point(self, point):
        r"""
        Determines if this `SingleSphericalPolygon` contains a given point.

        Parameters
        ----------
        point : an (*x*, *y*, *z*) triple
            The point to test.

        Returns
        -------
        contains : bool, None
            Returns `True` if the polygon contains the given *point*.
            Returns `None` if the polygon is degenerate.
        """
        return self._contains_point(point, self._points, self._inside)

    def contains_lonlat(self, lon, lat, degrees=True):
        r"""
        Determines if this `SingleSphericalPolygon` contains a given
        longitude and latitude.

        Parameters
        ----------
        lon, lat: Longitude and latitude. Must be scalars.

        degrees : bool, optional

       If `True`, (default) *lon* and *lat* are in decimal degrees,
       otherwise in radians.

        Returns
        -------
        contains : bool, None
            Returns `True` if the polygon contains the given *point*.
            Returns `None` if the polygon is degenerate.
        """
        point = vector.lonlat_to_vector(lon, lat, degrees=degrees)
        return self._contains_point(point, self._points, self._inside)

    # Alias for contains_lonlat
    contains_radec = contains_lonlat

    def intersects_poly(self, other):
        r"""
        Determines if this `SingleSphericalPolygon` intersects another
        `SingleSphericalPolygon`.

        This method is much faster than actually computing the
        intersection region between two polygons.

        Parameters
        ----------
        other : `SingleSphericalPolygon`

        Returns
        -------
        intersects : bool, None
            Returns `True` if this polygon intersects the *other*
            polygon. Returns `None` if either polygon is degenerate.

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
        if not isinstance(other, SingleSphericalPolygon):
            raise TypeError

        # The easy case is in which a point of one polygon is
        # contained in the other polygon.
        if self._degenerate or other._degenerate:
            return None
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
        Determines if this `SingleSphericalPolygon` intersects or contains
        the given arc. Returns `None` if the polygon is degenerate.
        """
        if self._degenerate:
            return None

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
        if self._degenerate:
            return None
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

        The area can be negative if the points on the polygon are ordered
        counter-clockwise. Take the absolute value if that is not desired.
        Area of degenerate polygons is defined to be zero.
        """
        if self._degenerate:
            return 0.0

        if len(self._points) < 4:
            return 0.0

        points = np.vstack((self._points, self._points[1]))
        angles = great_circle_arc.angle(points[:-2], points[1:-1], points[2:])

        return np.sum(angles) - (len(angles) - 2) * np.pi

    def union(self, other):
        """
        Return a new `SphericalPolygon` that is the union of *self*
        and *other*.

        Parameters
        ----------
        other : `SingleSphericalPolygon`

        Returns
        -------
        polygon : `SphericalPolygon`

        See also
        --------
        SphericalPolygon.multi_union

        Notes
        -----
        For implementation details, see the
        :mod:`~spherical_geometry.graph` module.
        """
        from . import graph
        if self._degenerate:
            return SphericalPolygon([other.copy()])
        elif other._degenerate:
            return SphericalPolygon([self.copy()])

        g = graph.Graph([self, other])

        return g.union()

    def intersection(self, other):
        """
        Return a new `SphericalPolygon` that is the intersection of
        *self* and *other*.

        If the intersection is empty, a `SphericalPolygon` with zero
        subpolyons will be returned.

        Parameters
        ----------
        other : `SingleSphericalPolygon`

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
        if self._degenerate or other._degenerate:
            return SphericalPolygon([])

        g = graph.Graph([self, other])

        return g.intersection()

    def overlap(self, other):
        r"""
        Returns the fraction of *self* that is overlapped by *other*.

        Let *self* be *a* and *other* be *b*, then the overlap is
        defined as:

        .. math::

            \frac{S_{a \cap b}}{S_a}

        Parameters
        ----------
        other : `SingleSphericalPolygon`

        Returns
        -------
        frac : float
            The fraction of *self* that is overlapped by *other*.
        """
        if self._degenerate or other._degenerate:
            return 0.0
        s1 = self.area()
        intersection = self.intersection(other)
        s2 = intersection.area()
        return abs(s2 / s1)

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
            length = np.rad2deg(great_circle_arc.length(A, B))
            if not np.isfinite(length):
                length = 2
            interpolated = great_circle_arc.interpolate(A, B, length * 4)
            lon, lat = vector.vector_to_lonlat(
                interpolated[:, 0], interpolated[:, 1], interpolated[:, 2],
                degrees=True)
            for lon0, lat0, lon1, lat1 in zip(lon[0:-1], lat[0:-1], lon[1:], lat[1:]):
                m.drawgreatcircle(lon0, lat0, lon1, lat1, alpha=alpha, **plot_args)
            alpha -= 1.0 / len(points)

        lon, lat = vector.vector_to_lonlat(
            *self._inside, degrees=True)
        x, y = m(lon, lat)
        m.scatter(x, y, 1, **plot_args)


class SphericalPolygon(SingleSphericalPolygon):
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
                 polygon.

                 It may contain zero points, in which it defines the
                 null polygon.  It may not contain one or two points.

        inside : An (*x*, *y*, *z*) triple, optional
            If *init* is an array of points, this point must be inside
            the polygon.  If it is not provided, one will be created.

        """
        from . import graph

        self._degenerate = False

        for polygon in init:
            if not isinstance(polygon, (SphericalPolygon, SingleSphericalPolygon)):
                break
            if polygon._degenerate:
                self._degenerate = True
        else:
            self._polygons = tuple(init)
            return

        p = SingleSphericalPolygon(init, inside)
        if p._degenerate:
            self._degenerate = True
            self._polygons = tuple()
            return
        else:
            self._degenerate = False

        self._polygons = (p,)

        polygons = []
        for polygon in self:
            if polygon._degenerate:
                continue
            g = graph.Graph((polygon,))
            polygons.extend(g.disjoint_polygons())
        if not polygons:
            self._degenerate = True
            self._polygons = tuple()
        else:
            self._polygons = polygons

    def __copy__(self):
        return deepcopy(self)

    copy = __copy__

    def __len__(self):
        return len(self._polygons)

    def __repr__(self):
        buffer = []
        for polygon in self._polygons:
            buffer.append(repr(polygon))
        return '[' + ',\n'.join(buffer) + ']'

    def __iter__(self):
        """
        Iterate over all base polygons that make up this multi-polygon
        set.
        """
        for polygon in self._polygons:
            for subpolygon in polygon:
                yield subpolygon

    @property
    def degenerate(self):
        """
        Return `True` if the polygon is degenerate (i.e., has no points,
        or is otherwise invalid).
        """
        return self._degenerate

    @property
    def points(self):
        """
        The points defining the polygons.  It is an iterator over
        disjoint closed polygons, where each element is an Nx3 array
        of (*x*, *y*, *z*) vectors.  Each polygon is explicitly
        closed, i.e., the first and last points are the same.
        """
        for polygon in self:
            yield polygon.points

    @property
    def inside(self):
        """
        Iterate over the inside point of each of the polygons.
        """
        for polygon in self:
            yield polygon.inside

    @property
    def polygons(self):
        """
        Get a sequence of all of the subpolygons.  Each subpolygon may
        itself have subpolygons.
        """
        return self._polygons

    def is_clockwise(self):
        """
        Return `True` if all subpolygons are clockwise, `False` if all are
        counter-clockwise, and `None` if some are clockwise and some are
        counter-clockwise or are degenerate.
        """
        cw = 0
        cww = 0
        deg = 0

        for polygon in self._polygons:
            is_cw = polygon.is_clockwise()
            if is_cw is None:
                deg += 1
            elif is_cw:
                cw += 1
            else:
                cww += 1

        if cw > 0 and cww == 0 and deg == 0:
            return True
        if cww > 0 and cw == 0 and deg == 0:
            return False
        return None

    @staticmethod
    def self_intersect(points):
        """
        Return true if the path defined by a list of points
        intersects itself
        """
        from . import graph
        polygon = SingleSphericalPolygon(points)
        g = graph.Graph((polygon,))
        return g._find_all_intersections()

    def to_lonlat(self):
        """
        Convert the `SphericalPolygon` footprint to longitude and latitude
        coordinates.

        Returns
        -------
        polyons : iterator
            Each element in the iterator is a tuple of the form (*lon*,
            *lat*), where each is an array of points.
        """
        for polygon in self:
            yield polygon.to_lonlat()

    # to_ra_dec is an alias for to_lonlat
    to_radec = to_lonlat

    @classmethod
    def from_lonlat(cls, lon, lat, center=None, degrees=True):
        # TODO Move into SingleSphericalPolygon
        r"""
        Create a new `SphericalPolygon` from a list of
        (*longitude*, *latitude*) points.

        Parameters
        ----------
        lon, lat : 1-D arrays of the same length
            The vertices of the polygon in longitude and
            latitude.

        center : (*lon*, *lat*) pair, optional
            A point inside of the polygon to define its inside.

        degrees : bool, optional
            If `True`, (default) *lon* and *lat* are in decimal degrees,
            otherwise in radians.

        Returns
        -------
        polygon : `SphericalPolygon` object
        """
        polygon = SingleSphericalPolygon.from_lonlat(lon, lat,
                                                     center, degrees)
        return cls((polygon,))

    # from_radec is an alias for from_lon_lat
    from_radec = from_lonlat

    @classmethod
    def from_cone(cls, lon, lat, radius, degrees=True, steps=16):
        r"""
        Create a new `SphericalPolygon` from a cone (otherwise known
        as a "small circle") defined using (*lon*, *lat*, *radius*).

        The cone is not represented as an ideal circle on the sphere,
        but as a series of great circle arcs.  The resolution of this
        conversion can be controlled using the *steps* parameter.

        Parameters
        ----------
        lon, lat : float scalars
            This defines the center of the cone

        radius : float scalar
            The radius of the cone

        degrees : bool, optional
            If `True`, (default) *lon*, *lat* and *radius* are in
            decimal degrees, otherwise in radians.

        steps : int, optional
            The number of steps to use when converting the small
            circle to a polygon.

        Returns
        -------
        polygon : `SphericalPolygon` object
        """
        polygon = SingleSphericalPolygon.from_cone(lon, lat, radius,
                                                   degrees=degrees, steps=steps)
        return cls((polygon,))

    @classmethod
    def from_wcs(cls, wcs, steps: int = 1) -> "SphericalPolygon":
        """Create a `SphericalPolygon` from the footprint of a world coordinate system.

        If the number of edges per side is set to 1, the polygon will be rectangular.
        Otherwise, the polygon will capture WCS distortion along the edges of the footprint.

        This method requires `astropy <http://astropy.org>`__ installed.

        Parameters
        ----------
        wcs: astropy.wcs.WCS | astropy.io.fits.Header | str :
            any WCS object that implements the common WCS API
        steps: int, optional :
            The number of edges to create along each side of the polygon. (Default value = 1)

        Returns
        -------
        polygon : `SphericalPolygon` object
        """
        return cls((SingleSphericalPolygon.from_wcs(wcs, steps),))

    @classmethod
    def convex_hull(cls, points):
        r"""
        Create a new `SphericalPolygon` from the convex hull of a
        list of points.

        Parameters
        ----------
        points: A list of points on the unit sphere

        Returns
        -------
        polygon : `SphericalPolygon` object
        """
        polygon = SingleSphericalPolygon.convex_hull(points)
        return cls((polygon,))

    def invert_polygon(self):
        """
        Construct a polygon which is the inverse (complement) of the
        original polygon
        """
        if len(self._polygons) != 1:
            raise RuntimeError("Can only invert a single polygon")

        klass = self.__class__
        inverted_polygon = self._polygons[0].invert_polygon()
        if inverted_polygon is None:
            return None
        return klass((inverted_polygon, ))

    def contains_point(self, point):
        r"""
        Determines if this `SphericalPolygon` contains a given point.

        Parameters
        ----------
        point : an (*x*, *y*, *z*) triple
            The point to test.

        Returns
        -------
        contains : bool, None
            Returns `True` if the polygon contains the given *point*.
            Returns `None` if the polygon is degenerate.
        """
        if self._degenerate:
            return None
        for polygon in self:
            if polygon.contains_point(point):
                return True
        return False

    def contains_lonlat(self, lon, lat, degrees=True):
        r"""
        Determines if this `SphericalPolygon` contains a given
        longitude and latitude.

        Parameters
        ----------
        lon, lat: Longitude and latitude. Must be scalars.

        degrees : bool, optional

       If `True`, (default) *lon* and *lat* are in decimal degrees,
       otherwise in radians.

        Returns
        -------
        contains : bool, None
            Returns `True` if the polygon contains the given *point*.
            Returns `None` if the polygon is degenerate.
        """
        if self._degenerate:
            return None
        for polygon in self:
            if polygon.contains_lonlat(lon, lat, degrees=degrees):
                return True
        return False

    # Alias for contains_lonlat
    contains_radec = contains_lonlat

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
        intersects : bool, None
            Returns `True` if this polygon intersects the *other*
            polygon. Returns `None` if either polygon is degenerate.
        """
        if not isinstance(other, SphericalPolygon):
            raise TypeError

        if self._degenerate or other._degenerate:
            return None

        for polya in self:
            for polyb in other:
                if polya.intersects_poly(polyb):
                    return True
        return False

    def intersects_arc(self, a, b):
        """
        Determines if this `SphericalPolygon` intersects or contains
        the given arc. Returns `None` if the polygon is degenerate.
        """
        if self._degenerate:
            return None
        for subpolygon in self:
            if subpolygon.intersects_arc(a, b):
                return True
        return False

    def contains_arc(self, a, b):
        """
        Returns `True` if the polygon fully encloses the arc given by a
        and b. Returns `None` if the polygon is degenerate.
        """
        if self._degenerate:
            return None
        for subpolygon in self:
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
        for subpoly in self:
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

        g = graph.Graph(list(self) + list(other))
        return g.union()

    @classmethod
    def multi_union(cls, polygons):
        """
        Return a new `SphericalPolygon` that is the union of all of the
        polygons in *polygons*. Currently this implementation exhibits
        exponential time behavior and becomes practically unusable when
        dealing with on the order of 40 or more polygons.

        Also, current implementation struggles when some of the input polygons
        are nearly identical. As a workaround, this method pre-filters
        input polygons and excludes those nearly the same as some other input
        polygon. Two poligons treated as the same polygon if their vertices
        (``x``, ``y``, and ``z`` cordinates on a unit sphere) differ by less
        than ``5e-9``. This is equivalent to polygon vertices being separated
        by less than 0.0015 arcsec on the sky or by less than ``2 mm`` on Earth
        (at average Earth radius).

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
        if not len(polygons):
            raise ValueError

        valid_polygons = []
        for polygon in polygons:
            if not isinstance(polygon, SphericalPolygon):
                raise TypeError("Expected a sequence of SphericalPolygon")
            if not polygon._degenerate:
                valid_polygons.append(polygon)

        if not valid_polygons:
            raise ValueError("No valid polygons provided")

        # Next block is a workaround to a bug in the graph code that leads to a
        # crash when computing multi_union of polygons
        # when some of input polygons are very close to each other.
        # Remove the next block once the bug is fixed.
        # See https://github.com/spacetelescope/spherical_geometry/issues/232
        #
        # This woraround will result in two poligons treated as the same
        # polygon if their vertices (x, y, z on a unit sphere) differ by 5e-9.
        # This is equivalent to polygon vertices being separated by less than
        # 0.0015 arcsec on the sky or by less than 2mm on Earth (at average
        # Earth radius).
        accepted_polygon_points = [np.sort(list(valid_polygons[0].points)[0], axis=0)]
        filtered_polygons = [valid_polygons[0]]
        for p in valid_polygons[1:]:
            pts = np.sort(list(p.points)[0], axis=0)
            for pts2 in accepted_polygon_points:
                if (pts.size == pts2.size and
                        np.allclose(pts, pts2, rtol=0, atol=5e-9)):
                    break
            else:
                filtered_polygons.append(p)
                accepted_polygon_points.append(pts)
                continue

        from . import graph

        all_polygons = []
        for polygon in filtered_polygons:
            if polygon._degenerate:
                continue
            for p in polygon:
                if p._degenerate:
                    continue
                all_polygons.append(p)

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

        if self.area() == 0.0:
            return SphericalPolygon([])
        elif other.area() == 0.0:
            return SphericalPolygon([])

        all_polygons = []
        for polya in self:
            for polyb in other:
                if polya.intersects_poly(polyb):
                    subpolygons = polya.intersection(polyb)
                    if subpolygons is None:
                        continue
                    for p in subpolygons:
                        if p._degenerate:
                            continue
                        all_polygons.append(p)

        return SphericalPolygon(all_polygons)

    @classmethod
    def multi_intersection(cls, polygons):
        """
        Return a new `SphericalPolygon` that is the intersection of
        all of the polygons in *polygons*.

        Parameters
        ----------
        polygons : sequence of `SphericalPolygon`

        Returns
        -------
        polygon : `SphericalPolygon` object
        """
        if not len(polygons):
            raise ValueError

        for polygon in polygons:
            if not isinstance(polygon, SphericalPolygon):
                raise TypeError

        results = None
        for polygon in polygons:
            if results is None and not polygon._degenerate:
                results = polygon
            elif len(results.polygons) == 0:
                return results
            else:
                results = results.intersection(polygon)

        return results

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
