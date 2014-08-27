# -*- coding: utf-8 -*-

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
The `spherical_geometry.vector` module contains the basic operations for handling
vectors and converting them to and from other representations.
"""

from __future__ import unicode_literals

# THIRD-PARTY
import numpy as np

try:
    from . import math_util
    HAS_C_UFUNCS = True
except ImportError:
    HAS_C_UFUNCS = False


__all__ = ['radec_to_vector', 'vector_to_radec', 'normalize_vector',
           'rotate_around']


def radec_to_vector(ra, dec, degrees=True):
    r"""
    Converts a location on the unit sphere from right-ascension and
    declination to an *x*, *y*, *z* vector.

    Parameters
    ----------
    ra, dec : scalars or 1-D arrays

    degrees : bool, optional

       If `True`, (default) *ra* and *dec* are in decimal degrees,
       otherwise in radians.

    Returns
    -------
    x, y, z : tuple of scalars or 1-D arrays of the same length

    Notes
    -----
    Where right-ascension is *α* and declination is *δ*:

    .. math::
        x = \cos\alpha \cos\delta

        y = \sin\alpha \cos\delta

        z = \sin\delta
    """
    ra = np.asanyarray(ra)
    dec = np.asanyarray(dec)

    if degrees:
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
    else:
        ra_rad = ra
        dec_rad = dec

    cos_dec = np.cos(dec_rad)

    return (
        np.cos(ra_rad) * cos_dec,
        np.sin(ra_rad) * cos_dec,
        np.sin(dec_rad))


def vector_to_radec(x, y, z, degrees=True):
    r"""
    Converts a vector to right-ascension and declination.

    Parameters
    ----------
    x, y, z : scalars or 1-D arrays
        The input vectors

    degrees : bool, optional
        If `True` (default) the result is returned in decimal degrees,
        otherwise radians.

    Returns
    -------
    ra, dec : tuple of scalars or arrays of the same length

    Notes
    -----
    Where right-ascension is *α* and declination is
    *δ*:

    .. math::
        \alpha = \arctan2(y, x)

        \delta = \arctan2(z, \sqrt{x^2 + y^2})
    """
    x = np.asanyarray(x, dtype=np.float64)
    y = np.asanyarray(y, dtype=np.float64)
    z = np.asanyarray(z, dtype=np.float64)

    result = (
        np.arctan2(y, x),
        np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))

    if degrees:
        return np.rad2deg(result[0]), np.rad2deg(result[1])
    else:
        return result


def normalize_vector(xyz, output=None):
    r"""
    Normalizes a vector so it falls on the unit sphere.

    Parameters
    ----------
    xyz : Nx3 array of vectors
        The input vectors

    output : Nx3 array of vectors, optional
        The array to store the results in.  If `None`, a new array
        will be created and returned.

    Returns
    -------
    output : Nx3 array of vectors
    """
    xyz = np.asanyarray(xyz, dtype=np.float64)

    if output is None:
        output = np.empty(xyz.shape, dtype=np.float64)

    if HAS_C_UFUNCS:
        math_util.normalize(xyz, output)
        return output

    l = np.sqrt(np.sum(xyz * xyz, axis=-1))

    output = xyz / np.expand_dims(l, 2)

    return output


def rotate_around(x, y, z, u, v, w, theta, degrees=True):
    r"""
    Rotates the vector (*x*, *y*, *z*) around the arbitrary axis defined by
    vector (*u*, *v*, *w*) by *theta*.

    It is assumed that both (*x*, *y*, *z*) and (*u*, *v*, *w*) are
    already normalized.

    Parameters
    ----------
    x, y, z : doubles
        The normalized vector to rotate

    u, v, w : doubles
        The normalized vector to rotate around

    theta : double, or array of doubles
        The amount to rotate

    degrees : bool, optional
        When `True`, *theta* is given in degrees, otherwise radians.

    Returns
    -------
    X, Y, Z : doubles
        The rotated vector
    """
    if degrees:
        theta = np.deg2rad(theta)

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    icostheta = 1.0 - costheta

    det = (-u*x - v*y - w*z)
    X = (-u*det)*icostheta + x*costheta + (-w*y + v*z)*sintheta
    Y = (-v*det)*icostheta + y*costheta + ( w*x - u*z)*sintheta
    Z = (-w*det)*icostheta + z*costheta + (-v*x + u*y)*sintheta

    return X, Y, Z


def equal_area_proj(points):
    """
    Transform the coordinates to a 2-dimensional plane using the
    Lambert azimuthal equal-area projection.

    Parameters
    ----------
    points : Nx3 array of vectors
        The input vectors

    Returns
    -------
    output : Nx2 array of points

    Notes
    -----

    .. math::

        X = \sqrt{\frac{2}{1-z}}x

    .. math::

        Y = \sqrt{\frac{2}{1-z}}y
    """
    scale = np.sqrt(2.0 / (1.0 - points[..., 2]))
    return np.expand_dims(scale, 2) * points[:, 0:2]
