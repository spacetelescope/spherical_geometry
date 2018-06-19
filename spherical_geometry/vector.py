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


__all__ = ['two_d', 'lonlat_to_vector', 'vector_to_lonlat',
           'normalize_vector', 'radec_to_vector', 'vector_to_radec',
           'rotate_around']

def two_d(vec):
    """
    Reshape a one dimensional vector so it has a second dimension
    """
    shape = list(vec.shape)
    shape.append(1)
    shape = tuple(shape)
    return np.reshape(vec, shape)

def lonlat_to_vector(lon, lat, degrees=True):
    r"""
    Converts a location on the unit sphere from longitude and
    latitude to an *x*, *y*, *z* vector.

    Parameters
    ----------
    lon, lat : scalars or 1-D arrays

    degrees : bool, optional

       If `True`, (default) *lon* and *lat* are in decimal degrees,
       otherwise in radians.

    Returns
    -------
    x, y, z : tuple of scalars or 1-D arrays of the same length

    Notes
    -----
    Where longitude is *l* and latitude is *b*:

    .. math::
        x = \cos l \cos b

        y = \sin l \cos b

        z = \sin b
    """
    lon = np.asanyarray(lon)
    lat = np.asanyarray(lat)

    if degrees:
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
    else:
        lon_rad = lon
        lat_rad = lat

    cos_lat = np.cos(lat_rad)

    return (
        np.cos(lon_rad) * cos_lat,
        np.sin(lon_rad) * cos_lat,
        np.sin(lat_rad))

# Alias for lonlat_to_vector
radec_to_vector = lonlat_to_vector


def vector_to_lonlat(x, y, z, degrees=True):
    r"""
    Converts a vector to longitude and latitude.

    Parameters
    ----------
    x, y, z : scalars or 1-D arrays
        The input vectors

    degrees : bool, optional
        If `True` (default) the result is returned in decimal degrees,
        otherwise radians.

    Returns
    -------
    lon, lat : tuple of scalars or arrays of the same length

    Notes
    -----
    Where longitude is *l* and latitude is *b*:

    .. math::
        l = \arctan2(y, x)

        b = \arctan2(z, \sqrt{x^2 + y^2})
    """
    x = np.asanyarray(x, dtype=np.float64)
    y = np.asanyarray(y, dtype=np.float64)
    z = np.asanyarray(z, dtype=np.float64)

    lon = np.arctan2(y, x)
    lon = np.remainder(lon, 2.0 * np.pi)

    lat = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    result = (lon, lat)

    if degrees:
        return np.rad2deg(result[0]), np.rad2deg(result[1])
    else:
        return result

# Alias for vector_to_lonlat
vector_to_radec = vector_to_lonlat


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

    output = xyz / two_d(l)

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

        X = \\sqrt{\\frac{2}{1-z}}x

    .. math::

        Y = \\sqrt{\\frac{2}{1-z}}y
    """
    scale = np.sqrt(2.0 / (1.0 - points[..., 2]))
    return two_d(scale) * points[:, 0:2]
