# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
The `spherical_geometry.great_circle_arc` module contains functions for computing
the length, intersection, angle and midpoint of great circle arcs.

Great circles are circles on the unit sphere whose center is
coincident with the center of the sphere.  Great circle arcs are the
section of those circles between two points on the unit sphere.
"""
# STDLIB
from fractions import Fraction

# THIRD-PARTY
import numpy as np

# LOCAL
from spherical_geometry.vector import two_d

# C versions of the code have been written to speed up operations
# the python versions are a fallback if the C cannot be used
try:
    from spherical_geometry import math_util
    HAS_C_UFUNCS = True
except ImportError:
    HAS_C_UFUNCS = False


__all__ = ['angle', 'interpolate', 'intersection', 'intersects',
           'intersects_point', 'length', 'midpoint']


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

        cpT[0] = aT[1]*bT[2] - aT[2]*bT[1]
        cpT[1] = aT[2]*bT[0] - aT[0]*bT[2]
        cpT[2] = aT[0]*bT[1] - aT[1]*bT[0]

        return cp


if HAS_C_UFUNCS:
    def _cross_and_normalize(A, B, eps=1e-31):
        with np.errstate(invalid='ignore'):
            # TODO: Figure out how to pass eps to C-ufunc
            return math_util.cross_and_norm(A, B)
else:
    def _cross_and_normalize(a, b, eps=1e-15):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        # Cross product, shape (..., 3)
        x = _fast_cross(a, b)

        # Norm, shape (...)
        n = np.linalg.norm(x, axis=-1)

        # Scalar case
        if np.isscalar(n):
            if n < eps:
                return np.full_like(x, np.nan, dtype=float)
            return x / n

        # Array case
        mask_small = n <= eps  # shape (...)
        denom = n.copy()
        denom[mask_small] = np.nan

        # Broadcast denom to (..., 3)
        denom = denom[..., None]

        with np.errstate(invalid="ignore", divide="ignore"):
            out = x / denom
        return out


if HAS_C_UFUNCS:
    triple_product = math_util.triple_product
else:
    def triple_product(A, B, C):
        return inner1d(C, _fast_cross(A, B))


def intersection_old(A, B, C, D):
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

        s_1 = ((A × B) × A) \cdot T

        s_2 = (B × (A × B)) \cdot T

        s_3 = ((C × D) × C) \cdot T

        s_4 = (D × (C × D)) \cdot T

    For :math:`s_n`, if all positive :math:`T` is returned as-is.  If
    all negative, :math:`T` is multiplied by :math:`-1`.  Otherwise
    the intersection does not exist and is undefined.

    References
    ----------

    .. [1] Method explained in an `e-mail
        <http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271>`_
        by Roger Stafford.

    http://www.mathworks.com/matlabcentral/newsreader/view_thread/276271

    Also see: http://www.boeing-727.com/Data/fly%20odds/distance.html
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
        s = two_d(s)

    cross = np.where(s == -4, -T, np.where(s == 4, T, np.nan))

    # Q: Why do we check for strict equality below? Should we check for
    # closeness instead? What about rounding errors?

    # If they share a common point, it's not an intersection.  This
    # gets around some rounding-error/numerical problems with the
    # above.
    equals = (np.all(A == C, axis=-1) |
              np.all(A == D, axis=-1) |
              np.all(B == C, axis=-1) |
              np.all(B == D, axis=-1))

    equals = two_d(equals)

    return np.where(equals, np.nan, cross)


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

    Notes
    -----
    The length is computed using the following:

    .. math::

       \Delta = \arccos(A \cdot B)
    """
    if HAS_C_UFUNCS:
        result = math_util.length(A, B)
    else:
        # Original code used arccos of the dot product (arccos(a·b)), but this
        # can be inaccurate for very small angles due to floating point
        # precision. The following is more accurate both for small and large
        # angles, but is more expensive to compute:
        # length = arctan2(|a×b|, a·b)
        A = np.asanyarray(A)
        B = np.asanyarray(B)

        if np.any(np.all(A == 0, axis=-1)) or np.any(np.all(B == 0, axis=-1)):
            raise ValueError("Null vector")

        try:
            with np.errstate(invalid='raise'):
                dot = inner1d(A, B)
        except FloatingPointError:
            raise ValueError("Out of domain for acos")

        A, B = np.broadcast_arrays(A, B)
        cross = np.linalg.norm(_fast_cross(A, B), axis=-1)

        result = np.arctan2(cross, dot)

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
    if HAS_C_UFUNCS:
        return math_util.intersects(A, B, C, D)

    with np.errstate(invalid='ignore'):
        intersections = intersection(A, B, C, D)

    return np.isfinite(intersections[..., 0])


def intersects_point(A, B, C):
    """
    Returns True if point C is along the great circle arc *AB*.

    Parameters
    ----------
    A, B : (*x*, *y*, *z*) triples or Nx3 arrays of triples
        Endpoints of the great circle arc.

    C : (*x*, *y*, *z*) triples or array of triples of points

    Returns
    -------
    intersects : bool or array of bool
        If the point is on the line, returns `True`.
    """
    # Q: How do we know how to get from A to B? Clockwise or counterclockwise?
    # That is, do we always consider only minor arcs between two nodes?
    #
    # Depending on the direction, we might get different results. For example,
    # if A is Atlanta and B is Los Angeles, do we travel from A to B moving
    # westward or eastward?
    #
    # Also, if A and B are antipodes, then any point C would be along the great
    # circle arc between A and B, but we don't want to return True for all
    # points. So we need to determine the direction from A to B, and then check
    # if C is along that direction.

    if HAS_C_UFUNCS:
        return math_util.intersects_point(A, B, C)

    total_length = length(A, B)
    left_length = length(A, C)
    right_length = length(C, B)

    length_diff = np.abs((left_length + right_length) - total_length)

    return length_diff < 3e-11


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
        m = np.logical_or(
                np.linalg.norm(ABX, axis=-1) == 0.0,
                np.linalg.norm(BCX, axis=-1) == 0.0
            )

        diff = inner1d(B, X)
        inner = inner1d(ABX, BCX)
        with np.errstate(invalid='ignore'):
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
    l = two_d(l)
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
    steps = int(max(steps, 2))
    t = np.linspace(0.0, 1.0, steps, endpoint=True).reshape((steps, 1))

    omega = length(A, B)
    if omega == 0.0:
        offsets = t
    else:
        sin_omega = np.sin(omega)
        offsets = np.sin(t * omega) / sin_omega

    return offsets[::-1] * A + offsets * B


# TODO: Consider moving all this code (_det3_broadcast, robust_det_sign,
# intersection, and possibly _exact_det3) to math_util.c.
# _exact_det3 would require extra dependencies or, alternatively, we could
# eliminate _exact_det3 altogether by usinbg quad precision and use small epsilon
# to determine the sign of the determinant. Further

def _det3_broadcast(a, b, c):
    """
    Determinant of 3x3 with rows a, b, c.
    a, b, c: (..., 3) arrays, already broadcast to same shape.
    Returns: (...) array of determinants.
    """
    return (
        a[..., 0] * (b[..., 1] * c[..., 2] - b[..., 2] * c[..., 1]) -
        a[..., 1] * (b[..., 0] * c[..., 2] - b[..., 2] * c[..., 0]) +
        a[..., 2] * (b[..., 0] * c[..., 1] - b[..., 1] * c[..., 0])
    )


def _exact_det3(a, b, c):
    """
    Exact 3x3 determinant using Fractions for single 3-vectors.
    a, b, c: shape (3,) arrays, assumed finite and non-NaN.
    """
    fa = [Fraction(x) for x in a]
    fb = [Fraction(x) for x in b]
    fc = [Fraction(x) for x in c]

    return (
        fa[0] * (fb[1] * fc[2] - fb[2] * fc[1]) -
        fa[1] * (fb[0] * fc[2] - fb[2] * fc[0]) +
        fa[2] * (fb[0] * fc[1] - fb[1] * fc[0])
    )


def robust_det_sign(a, b, c, eps=1e-15):
    """
    Shewchuk-style adaptive sign of det([a; b; c]) with broadcasting.

    a, b, c: (..., 3) arrays (any broadcastable combination).
    Returns: int array of shape (...) with values in {-1, 0, +1}.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    # Broadcast all three to a common shape (..., 3)
    a, b, c = np.broadcast_arrays(a, b, c)

    det = _det3_broadcast(a, b, c)
    sign = np.zeros_like(det, dtype=int)

    # fast path: |det| > eps → sign from float
    mask_fast_pos = det > eps
    mask_fast_neg = det < -eps
    sign[mask_fast_pos] = 1
    sign[mask_fast_neg] = -1

    # slow path: |det| <= eps → exact Fraction per element
    mask_slow = ~(mask_fast_pos | mask_fast_neg)

    if np.any(mask_slow):
        it = np.nditer(mask_slow, flags=['multi_index'])
        while not it.finished:
            if it[0]:
                idx = it.multi_index
                a_i = a[idx]  # shape (3,)
                b_i = b[idx]
                c_i = c[idx]

                # If any NaN/inf is present, we cannot do exact arithmetic;
                # treat as indeterminate sign → 0.
                if (np.isnan(a_i).any() or np.isnan(b_i).any() or np.isnan(c_i).any() or
                    np.isinf(a_i).any() or np.isinf(b_i).any() or np.isinf(c_i).any()):
                    sign[idx] = 0
                else:
                    det_exact = _exact_det3(a_i, b_i, c_i)
                    if det_exact > 0:
                        sign[idx] = 1
                    elif det_exact < 0:
                        sign[idx] = -1
                    else:
                        sign[idx] = 0
            it.iternext()

    return sign


def intersection(A, B, C, D, eps=1e-13):
    r"""
    Returns the point of intersection between two great circle arcs.
    The arcs are defined between the points *AB* and *CD*.  Either *A*
    and *B* or *C* and *D* may be arrays of points, but not both.
    """

    A = np.asanyarray(A, dtype=float)
    B = np.asanyarray(B, dtype=float)
    C = np.asanyarray(C, dtype=float)
    D = np.asanyarray(D, dtype=float)

    A, B = np.broadcast_arrays(A, B)
    C, D = np.broadcast_arrays(C, D)

    ABX = _fast_cross(A, B)
    CDX = _fast_cross(C, D)
    T = _cross_and_normalize(ABX, CDX, eps=eps)
    T_ndim = T.ndim

    if T_ndim > 1:
        s = np.zeros(T.shape[0], dtype=int)
    else:
        s = np.zeros(1, dtype=int)

    # ((A x B) x A) . T  == det([A x B; A; T])
    s += robust_det_sign(ABX, A, T, eps=eps)
    # (B x (A x B)) · T  == det([B; A x B; T])
    s += robust_det_sign(B, ABX, T, eps=eps)
    # ((C x D) x C) · T  == det([C x D; C; T])
    s += robust_det_sign(CDX, C, T, eps=eps)
    # (D x (C x D)) · T  == det([D; C x D; T])
    s += robust_det_sign(D, CDX, T, eps=eps)

    if T_ndim > 1:
        s_col = two_d(s)
    else:
        s_col = s

    cross = np.where(s_col == -4, -T,
                     np.where(s_col == 4, T, np.nan))

    equals = (np.all(A == C, axis=-1) |
              np.all(A == D, axis=-1) |
              np.all(B == C, axis=-1) |
              np.all(B == D, axis=-1))

    equals = two_d(equals)

    return np.where(equals, np.nan, cross)
