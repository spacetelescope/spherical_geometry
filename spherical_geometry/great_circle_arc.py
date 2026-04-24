# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The `spherical_geometry.great_circle_arc` module contains functions for computing
the length, intersection, angle and midpoint of great circle arcs.

Great circles are circles on the unit sphere whose center is
coincident with the center of the sphere.  Great circle arcs are the
section of those circles between two points on the unit sphere.
"""

# THIRD-PARTY
import numpy as np
import s2geometry as s2

# LOCAL
from spherical_geometry.vector import two_d

# C versions of the code have been written to speed up operations
# the python versions are a fallback if the C cannot be used
try:
    from spherical_geometry import math_util

    HAS_C_UFUNCS = True
except ImportError:
    HAS_C_UFUNCS = False

__all__ = [
    "angle",
    "interpolate",
    "intersection",
    "intersects",
    "intersects_point",
    "length",
    "midpoint",
]


def _inner1d_np(x, y):
    return np.multiply(x, y).sum(axis=-1)


if HAS_C_UFUNCS:
    inner1d = math_util.inner1d
else:
    inner1d = _inner1d_np


if HAS_C_UFUNCS:
    _fast_cross = math_util.cross
else:

    def _fast_cross(a, b):
        """
        This is a reimplementation of `numpy.cross` that only does 3D x
        3D, and is therefore faster since it doesn't need any
        conditionals.
        """
        if HAS_C_UFUNCS:
            return math_util.cross(a, b)

        cp = np.empty(np.broadcast(a, b).shape)
        aT = a.T
        bT = b.T
        cpT = cp.T

        cpT[0] = aT[1] * bT[2] - aT[2] * bT[1]
        cpT[1] = aT[2] * bT[0] - aT[0] * bT[2]
        cpT[2] = aT[0] * bT[1] - aT[1] * bT[0]

        return cp


if HAS_C_UFUNCS:

    def _cross_and_normalize(A, B):
        with np.errstate(invalid="ignore"):
            return math_util.cross_and_norm(A, B)
else:

    def _cross_and_normalize(A, B):
        T = _fast_cross(A, B)
        # Normalization
        l = np.sqrt(np.sum(T**2, axis=-1))
        l = two_d(l)
        # Might get some divide-by-zeros
        with np.errstate(invalid="ignore"):
            TN = T / l
        # ... but set to zero, or we miss real NaNs elsewhere
        TN = np.nan_to_num(TN)
        return TN


if HAS_C_UFUNCS:
    triple_product = math_util.triple_product
else:

    def triple_product(A, B, C):
        return inner1d(C, _fast_cross(A, B))


def length(A, B):
    r"""
    Returns the angular distance between two points (in vector space)
    on the unit sphere.

    Parameters
    ----------
    A, B : (*x*, *y*, *z*) triples or Nx3 arrays of triples
       The endpoints of the great circle arc, in vector space.

    Returns
    -------
    length : scalar or array of scalars
        The angular length of the great circle arc in radians.
    """

    return s2.S1Angle(s2.S2Point_FromRaw(*A), s2.S2Point_FromRaw(*B)).radians()


def angle(A, B, C):
    """
    Returns the angle at *B* between *AB* and *BC*.

    Parameters
    ----------
    A, B, C : (*x*, *y*, *z*) triples or Nx3 arrays of triples
        Points on sphere.

    Returns
    -------
    angle : float or array of floats
        The angle at *B* between *AB* and *BC*, in range 0 to 2π.

    References
    ----------

    .. [1] Miller, Robert D.  Computing the area of a spherical
       polygon.  Graphics Gems IV.  1994.  Academic Press.
    """
    if HAS_C_UFUNCS:
        angle = math_util.angle(A, B, C)
    else:
        A = np.asanyarray(A)
        B = np.asanyarray(B)
        C = np.asanyarray(C)

        A, B, C = np.broadcast_arrays(A, B, C)

        ABX = _cross_and_normalize(A, B)
        BCX = _cross_and_normalize(C, B)
        X = _cross_and_normalize(ABX, BCX)
        m = np.logical_or(np.linalg.norm(ABX, axis=-1) == 0.0, np.linalg.norm(BCX, axis=-1) == 0.0)

        diff = inner1d(B, X)
        inner = inner1d(ABX, BCX)
        with np.errstate(invalid="ignore"):
            inner = np.clip(inner, -1.0, 1.0)  # needed due to accuracy loss
            angle = np.arccos(inner)

        angle = np.where(diff < 0.0, (2.0 * np.pi) - angle, angle)

        angle[m] = np.nan

    return angle


def midpoint(A, B):
    """
    Returns the midpoint on the great circle arc between *A* and *B*.

    Parameters
    ----------
    A, B : (*x*, *y*, *z*) triples
        The endpoints of the great circle arc.  It is assumed that
        these points are already normalized.

    Returns
    -------
    midpoint : (*x*, *y*, *z*) triple or Nx3 arrays of triples
        The midpoint between *A* and *B*, normalized on the unit
        sphere.
    """
    arc = s2.S2Polyline.InitFromS2Points([s2.S2Point_FromRaw(*A), s2.S2Point_FromRaw(*B)])
    point = arc.Interpolate(0.5)
    return point.x(), point.y(), point.z()


def interpolate(A, B, steps=50):
    r"""
    Interpolate along the great circle arc.

    Parameters
    ----------
    A, B : (*x*, *y*, *z*) triples or Nx3 arrays of triples
        The endpoints of the great circle arc.  It is assumed thats
        these points are already normalized.

    steps : int
        The number of interpolation steps

    Returns
    -------
    array : (*x*, *y*, *z*) triples
        The points interpolated along the great circle arc
    """

    arc = s2.S2Polyline.InitFromS2Points([s2.S2Point_FromRaw(*A), s2.S2Point_FromRaw(*B)])
    points = [arc.Interpolate(index / steps + 2) for index in range(steps + 2)]
    return [(point.x(), point.y(), point.z()) for point in points]
