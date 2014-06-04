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
The `sphere.vector` module contains the basic operations for handling
vectors and converting them to and from other representations.
"""

from __future__ import unicode_literals

# THIRD-PARTY
import numpy as np


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
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    z = np.asanyarray(z)

    result = (
        np.arctan2(y, x),
        np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))

    if degrees:
        return np.rad2deg(result[0]), np.rad2deg(result[1])
    else:
        return result


def normalize_vector(x, y, z, inplace=False):
    r"""
    Normalizes a vector so it falls on the unit sphere.

    *x*, *y*, *z* may be scalars or 1-D arrays

    Parameters
    ----------
    x, y, z : scalars or 1-D arrays of the same length
        The input vectors

    inplace : bool, optional
        When `True`, the original arrays will be normalized in place,
        otherwise a normalized copy is returned.

    Returns
    -------
    X, Y, Z : scalars of 1-D arrays of the same length
        The normalized output vectors
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    z = np.asanyarray(z)

    l = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if inplace:
        x /= l
        y /= l
        z /= l
        return (x, y, z)
    else:
        return (x / l, y / l, z / l)


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


def equal_area_proj(x, y, z):
    """
    Transform the coordinates to a 2-dimensional plane using the
    Lambert azimuthal equal-area projection.

    Parameters
    ----------
    x, y, z : scalars or 1-D arrays
        The input vectors

    Returns
    -------
    X, Y : tuple of scalars or arrays of the same length

    Notes
    -----

    .. math::

        X = \sqrt{\frac{2}{1-z}}x

    .. math::

        Y = \sqrt{\frac{2}{1-z}}y
    """
    scale = np.sqrt(2.0 / (1.0 - z))
    X = scale * x
    Y = scale * y
    return X, Y
