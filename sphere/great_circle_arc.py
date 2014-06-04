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
The `sphere.great_circle_arc` module contains functions for computing
the length, intersection, angle and midpoint of great circle arcs.

Great circles are circles on the unit sphere whose center is
coincident with the center of the sphere.  Great circle arcs are the
section of those circles between two points on the unit sphere.
"""

from __future__ import with_statement, division, absolute_import, unicode_literals

# THIRD-PARTY
import numpy as np


try:
    from . import math_util
    HAS_C_UFUNCS = True
except ImportError:
    HAS_C_UFUNCS = False

if HAS_C_UFUNCS:
    inner1d = math_util.inner1d
else:
    from numpy.core.umath_tests import inner1d


__all__ = ['angle', 'intersection', 'intersects', 'length', 'midpoint',
           'interpolate']


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

    cpT[0] = aT[1]*bT[2] - aT[2]*bT[1]
    cpT[1] = aT[2]*bT[0] - aT[0]*bT[2]
    cpT[2] = aT[0]*bT[1] - aT[1]*bT[0]

    return cp

def _cross_and_normalize(A, B):
    if HAS_C_UFUNCS:
        return math_util.cross_and_norm(A, B)

    T = _fast_cross(A, B)
    # Normalization
    l = np.sqrt(np.sum(T ** 2, axis=-1))
    l = np.expand_dims(l, 2)
    # Might get some divide-by-zeros, but we don't care
    with np.errstate(invalid='ignore'):
        TN = T / l
    return TN


def intersection(A, B, C, D):
    r"""
    Returns the point of intersection between two great circle arcs.
    The arcs are defined between the points *AB* and *CD*.  Either *A*
    and *B* or *C* and *D* may be arrays of points, but not both.

    Parameters
    ----------
    A, B : (*x*, *y*, *z*) triples or Nx3 arrays of triples
        Endpoints of the first great circle arc.

    C, D : (*x*, *y*, *z*) triples or Nx3 arrays of triples
        Endpoints of the second great circle arc.

    Returns
    -------
    T : (*x*, *y*, *z*) triples or Nx3 arrays of triples
        If the given arcs intersect, the intersection is returned.  If
        the arcs do not intersect, the triple is set to all NaNs.

    Notes
    -----
    The basic intersection is computed using linear algebra as follows
    [1]_:

    .. math::

        T = \lVert(A × B) × (C × D)\rVert

    To determine the correct sign (i.e. hemisphere) of the
    intersection, the following four values are computed:

    .. math::

        s_1 = ((A × B) × A) · T

        s_2 = (B × (A × B)) · T

        s_3 = ((C × D) × C) · T

        s_4 = (D × (C × D)) · T

    For :math:`s_n`, if all positive :math:`T` is returned as-is.  If
    all negative, :math:`T` is multiplied by :math:`-1`.  Otherwise
    the intersection does not exist and is undefined.

    References
    ----------

    .. [1] Method explained in an `e-mail
        <http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271>`_
        by Roger Stafford.

    http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271
    """
    if HAS_C_UFUNCS:
        return math_util.intersection(A, B, C, D)
    
    A = np.asanyarray(A)
    B = np.asanyarray(B)
    C = np.asanyarray(C)
    D = np.asanyarray(D)

    A, B = np.broadcast_arrays(A, B)
    C, D = np.broadcast_arrays(C, D)

    ABX = _fast_cross(A, B)
    CDX = _fast_cross(C, D)
    T = _cross_and_normalize(ABX, CDX)
    T_ndim = len(T.shape)

    if T_ndim > 1:
        s = np.zeros(T.shape[0])
    else:
        s = np.zeros(1)
    s += np.sign(inner1d(_fast_cross(ABX, A), T))
    s += np.sign(inner1d(_fast_cross(B, ABX), T))
    s += np.sign(inner1d(_fast_cross(CDX, C), T))
    s += np.sign(inner1d(_fast_cross(D, CDX), T))
    if T_ndim > 1:
        s = np.expand_dims(s, 2)

    cross = np.where(s == -4, -T, np.where(s == 4, T, np.nan))

    # If they share a common point, it's not an intersection.  This
    # gets around some rounding-error/numerical problems with the
    # above.
    equals = (np.all(A == C, axis = -1) |
              np.all(A == D, axis = -1) |
              np.all(B == C, axis = -1) |
              np.all(B == D, axis = -1))

    equals = np.expand_dims(equals, 2)

    return np.where(equals, np.nan, cross)


def length(A, B, degrees=True):
    r"""
    Returns the angular distance between two points (in vector space)
    on the unit sphere.

    Parameters
    ----------
    A, B : (*x*, *y*, *z*) triples or Nx3 arrays of triples
       The endpoints of the great circle arc, in vector space.

    degrees : bool, optional
        If `True` (default) the result is returned in decimal degrees,
        otherwise radians.

    Returns
    -------
    length : scalar or array of scalars
        The angular length of the great circle arc.

    Notes
    -----
    The length is computed using the following:

    .. math::

       \Delta = \arccos(A \dot B)
    """
    A = np.asanyarray(A)
    B = np.asanyarray(B)

    dot = inner1d(A, B)
    dot = np.clip(dot, -1.0, 1.0)
    with np.errstate(invalid='ignore'):
        result = np.arccos(dot)

    if degrees:
        return np.rad2deg(result)
    else:
        return result


def intersects(A, B, C, D):
    """
    Returns `True` if the great circle arcs between *AB* and *CD*
    intersect.  Either *A* and *B* or *C* and *D* may be arrays of
    points, but not both.

    Parameters
    ----------
    A, B : (*x*, *y*, *z*) triples or Nx3 arrays of triples
        Endpoints of the first great circle arc.

    C, D : (*x*, *y*, *z*) triples or Nx3 arrays of triples
        Endpoints of the second great circle arc.

    Returns
    -------
    intersects : bool or array of bool
        If the given arcs intersect, the intersection is returned as
        `True`.
    """
    with np.errstate(invalid='ignore'):
        intersections = intersection(A, B, C, D)
    return np.isfinite(intersections[..., 0])


def angle(A, B, C, degrees=True):
    """
    Returns the angle at *B* between *AB* and *BC*.

    This always returns the shortest angle < π.

    Parameters
    ----------
    A, B, C : (*x*, *y*, *z*) triples or Nx3 arrays of triples
        Points on sphere.

    degrees : bool, optional
        If `True` (default) the result is returned in decimal degrees,
        otherwise radians.

    Returns
    -------
    angle : float or array of floats
        The angle at *B* between *AB* and *BC*.

    References
    ----------

    .. [1] Miller, Robert D.  Computing the area of a spherical
       polygon.  Graphics Gems IV.  1994.  Academic Press.
    """
    A = np.asanyarray(A)
    B = np.asanyarray(B)
    C = np.asanyarray(C)

    A, B, C = np.broadcast_arrays(A, B, C)

    ABX = _fast_cross(A, B)
    ABX = _cross_and_normalize(B, ABX)
    BCX = _fast_cross(C, B)
    BCX = _cross_and_normalize(B, BCX)
    with np.errstate(invalid='ignore'):
        angle = np.arccos(inner1d(ABX, BCX))

    if degrees:
        angle = np.rad2deg(angle)
    return angle


def midpoint(A, B):
    """
    Returns the midpoint on the great circle arc between *A* and *B*.

    Parameters
    ----------
    A, B : (*x*, *y*, *z*) triples or Nx3 arrays of triples
        The endpoints of the great circle arc.  It is assumed that
        these points are already normalized.

    Returns
    -------
    midpoint : (*x*, *y*, *z*) triple or Nx3 arrays of triples
        The midpoint between *A* and *B*, normalized on the unit
        sphere.
    """
    P = (A + B) / 2.0
    # Now normalize...
    l = np.sqrt(np.sum(P * P, axis=-1))
    l = np.expand_dims(l, 2)
    return P / l


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

    Notes
    -----

    This uses Slerp interpolation where *Ω* is the angle subtended by
    the arc, and *t* is the parameter 0 <= *t* <= 1.

    .. math::

        \frac{\sin((1 - t)\Omega)}{\sin \Omega}A + \frac{\sin(t \Omega)}{\sin \Omega}B
    """
    steps = max(steps, 2)
    t = np.linspace(0.0, 1.0, steps, endpoint=True).reshape((steps, 1))

    omega = length(A, B, degrees=False)
    if omega == 0.0:
        offsets = t
    else:
        sin_omega = np.sin(omega)
        offsets = np.sin(t * omega) / sin_omega

    return offsets[::-1] * A + offsets * B
