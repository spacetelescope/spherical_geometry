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
import s2geometry as s2

# LOCAL
from spherical_geometry import great_circle_arc, vector

__all__ = ["SingleSphericalPolygon", "SphericalPolygon", "MalformedPolygonError"]


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
        if len(points) > 0:
            boundary = s2.S2Loop()
            boundary.Init(
                [
                    point
                    if isinstance(point, s2.S2Point)
                    else point.ToPoint()
                    if isinstance(point, s2.S2LatLng)
                    else s2.S2LatLng.FromDegrees(*point).ToPoint()
                    if len(point) == 2
                    else s2.S2Point_FromRaw(*point)
                    for point in points
                ]
            )
        else:
            boundary = None
        self._s2polygon = s2.S2Polygon(boundary)

        if boundary.num_vertices() == 0:
            # handle special case of initializing with an empty list of
            # vertices (ticket #1079).
            self._degenerate = True
            self._inside = None
        else:
            self._degenerate = False

            if inside is None:
                self._inside = inside
            else:
                self._inside = np.asanyarray(inside)

    def __copy__(self):
        clone = self.__class__(points=None, inside=self._inside)
        clone._s2polygon = self._s2polygon.Clone()
        return clone

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{self.__class__.__name__}({self.points})"

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
        if self._s2polygon.num_loops() > 0:
            boundary = self._s2polygon.loop(0)
            return np.stack(
                [
                    (point.x(), point.y(), point.z())
                    for point in (
                        boundary.vertex(index) for index in range(boundary.num_vertices())
                    )
                ]
            )
        else:
            return np.empty((0, 3))

    def to_lonlat(self):
        """
        Convert `SingleSphericalPolygon` footprint to longitude and latitude.

        Returns
        -------
        lon, lat : list of float
            List of *lon* and *lat* in degrees corresponding
            to `points`.
        """
        if self._s2polygon.num_loops() > 0:
            boundary = self._s2polygon.loop(0)
            latlngs = [
                s2.S2LatLng(boundary.vertex(index)) for index in range(boundary.num_vertices())
            ]
            return [latlng.lng() for latlng in latlngs], [latlng.lat() for latlng in latlngs]
        else:
            return [], []

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
        return cls(
            [
                s2.S2LatLng.FromDegrees(lat[index], lon[index]).ToPoint()
                for index in range(len(lon))
            ],
            inside=center,
        )

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

        # TODO use s2geometry

        u, v, w = vector.lonlat_to_vector(lon, lat, degrees=degrees)
        if degrees:
            radius = np.deg2rad(radius)

        # Get an arbitrary perpendicular vector.  This be be obtained
        # by crossing (u, v, w) with any unit vector that is not itself.
        which_min = np.argmin([u * u, v * v, w * w])
        if which_min == 0:
            perp = np.cross([u, v, w], [1.0, 0.0, 0.0])
        elif which_min == 1:
            perp = np.cross([u, v, w], [0.0, 1.0, 0.0])
        else:
            perp = np.cross([u, v, w], [0.0, 0.0, 1.0])
        perp = vector.normalize_vector(perp)

        # Rotate by radius around the perpendicular vector to get the
        # "pen"
        x, y, z = vector.rotate_around(u, v, w, perp[0], perp[1], perp[2], radius, degrees=False)

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
            if isinstance(bbox_attr, astropy.modeling.bounding_box.ModelBoundingBox):
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
            vertex_skycoords.ra.degree, vertex_skycoords.dec.degree, center=center
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

        # TODO use s2geometry

        points = np.asarray(points)

        # Find an extremal point, it must be on boundary

        j = np.argmin(np.arctan2(points[:, 1], points[:, 0]))
        extreme = points[j, :]
        points = np.vstack((points[0:j, :], points[j + 1 :, :]))

        # Sort points around extreme point by angle from true north

        north = [0.0, 0.0, 1.0]
        ang = great_circle_arc.angle(north, extreme, points)
        pt = [points[i, :] for i in range(points.shape[0])]

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
                hull.append(points[i, :])
                i += 1
            else:
                pprevious = hull[-2]
                if great_circle_arc.triple_product(pprevious, ptop, points[i, :]) > 0.0:
                    hull.append(points[i, :])
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
        poly._s2polygon.Invert()
        return poly

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
        return self._s2polygon.Contains(s2.S2Point_FromRaw(*point))

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
        return self._s2polygon.Contains(
            s2.S2LatLng.FromDegrees(lat, lon) if degrees else s2.S2LatLng.FromRadians(lat, lon)
        )

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

        """
        return self._s2polygon.Intersects(other._s2polygon)

    def intersects_arc(self, a, b):
        """
        Determines if this `SingleSphericalPolygon` intersects or contains
        the given arc. Returns `None` if the polygon is degenerate.
        """
        point_a = s2.S2Point_FromRaw(*a)
        point_b = s2.S2Point_FromRaw(*b)
        arc = s2.S2Polyline()
        arc.InitFromS2Points([point_a, point_b])
        return self._s2polygon.IntersectWithPolyline(arc)

    def contains_arc(self, a, b):
        """
        Returns `True` if the polygon fully encloses the arc given by a
        and b.
        """
        point_a = s2.S2Point_FromRaw(*a)
        point_b = s2.S2Point_FromRaw(*b)
        arc = s2.S2Polyline()
        arc.InitFromS2Points([point_a, point_b])
        return self._s2polygon.Contains(arc)

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
        return self._s2polygon.GetArea()

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
        """
        poly = self.__class__()
        poly._s2polygon.InitToUnion(self._s2polygon, other._s2polygon)
        return poly

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
        """
        poly = self.__class__()
        poly._s2polygon.InitToIntersection(self._s2polygon, other._s2polygon)
        return poly

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
        return s2.S2Polygon.GetOverlapFractions(self._s2polygon, other._s2polygon)[0]

    def draw(self, m, **plot_args):
        """
        Draws the polygon in a matplotlib.Basemap axes.

        Parameters
        ----------
        m : Basemap axes object

        **plot_args : Any plot arguments to pass to basemap
        """
        if not len(self.points):
            return
        if not len(plot_args):
            plot_args = {"color": "blue"}
        points = self.points
        if "alpha" in plot_args:
            del plot_args["alpha"]

        alpha = 1.0
        for A, B in zip(points[0:-1], points[1:]):
            length = np.rad2deg(great_circle_arc.length(A, B))
            if not np.isfinite(length):
                length = 2
            interpolated = great_circle_arc.interpolate(A, B, length * 4)
            lon, lat = vector.vector_to_lonlat(
                interpolated[:, 0], interpolated[:, 1], interpolated[:, 2], degrees=True
            )
            for lon0, lat0, lon1, lat1 in zip(lon[0:-1], lat[0:-1], lon[1:], lat[1:]):
                m.drawgreatcircle(lon0, lat0, lon1, lat1, alpha=alpha, **plot_args)
            alpha -= 1.0 / len(points)

        lon, lat = vector.vector_to_lonlat(*self._inside, degrees=True)
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
            polygons.append(polygon)
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
        return "[" + ",\n".join(buffer) + "]"

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
        raise NotImplementedError()  # TODO
        polyline = s2.S2Polyline.InitFromS2Points(
            [
                point
                if isinstance(point, s2.S2Point)
                else point.ToPoint()
                if isinstance(point, s2.S2LatLng)
                else s2.S2LatLng.FromDegrees(*point).ToPoint()
                if len(point) == 2
                else s2.S2Point_FromRaw(*point)
                for point in points
            ]
        )
        return s2.S2Polyline.Intersects()

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

        # TODO Move into SingleSphericalPolygon

        polygon = SingleSphericalPolygon.from_lonlat(lon, lat, center, degrees)
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
        polygon = SingleSphericalPolygon.from_cone(lon, lat, radius, degrees=degrees, steps=steps)
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
        return klass((inverted_polygon,))

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
        """
        return self.multi_union(self.polygons + other.polygons)

    @classmethod
    def multi_union(cls, polygons):
        """
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
        base = s2.S2Polygon()
        for multipolygon in polygons:
            for polygon in multipolygon:
                base = s2.S2Polygon.InitToUnion(base, polygon._s2polygon)
        return cls((base,))

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
        """

        return self.multi_intersection(self.polygons + other.polygons)

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
        base = s2.S2Polygon()
        for multipolygon in polygons:
            for polygon in multipolygon:
                base = s2.S2Polygon.InitToIntersection(base, polygon._s2polygon)
        return cls((base,))

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
