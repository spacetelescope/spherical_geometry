from __future__ import absolute_import

import os
import random

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_less

from .. import graph
from .. import great_circle_arc
from .. import polygon
from .. import vector

from .test_util import *
from .test_shared import resolve_imagename

graph.DEBUG = True


def test_normalize_vector():
    x, y, z = np.ogrid[-100:100:11,-100:100:11,-100:100:11]
    xn, yn, zn = vector.normalize_vector(x.flatten(), y.flatten(), z.flatten())
    l = np.sqrt(xn ** 2 + yn ** 2 + zn ** 2)
    assert_almost_equal(l, 1.0)


def test_radec_to_vector():
    npx, npy, npz = vector.radec_to_vector(np.arange(-360, 360, 1), 90)
    assert_almost_equal(npx, 0.0)
    assert_almost_equal(npy, 0.0)
    assert_almost_equal(npz, 1.0)

    spx, spy, spz = vector.radec_to_vector(np.arange(-360, 360, 1), -90)
    assert_almost_equal(spx, 0.0)
    assert_almost_equal(spy, 0.0)
    assert_almost_equal(spz, -1.0)

    eqx, eqy, eqz = vector.radec_to_vector(np.arange(-360, 360, 1), 0)
    assert_almost_equal(eqz, 0.0)


def test_vector_to_radec():
    ra, dec = vector.vector_to_radec(0, 0, 1)
    assert_almost_equal(dec, 90)

    ra, dec = vector.vector_to_radec(0, 0, -1)
    assert_almost_equal(dec, -90)

    ra, dec = vector.vector_to_radec(1, 1, 0)
    assert_almost_equal(ra, 45.0)
    assert_almost_equal(dec, 0.0)


def test_intersects_poly_simple():
    ra1 = [-10, 10, 10, -10, -10]
    dec1 = [30, 30, 0, 0, 30]
    poly1 = polygon.SphericalPolygon.from_radec(ra1, dec1)

    ra2 = [-5, 15, 15, -5, -5]
    dec2 = [20, 20, -10, -10, 20]
    poly2 = polygon.SphericalPolygon.from_radec(ra2, dec2)

    assert poly1.intersects_poly(poly2)

    # Make sure it isn't order-dependent
    ra1 = ra1[::-1]
    dec1 = dec1[::-1]
    poly1 = polygon.SphericalPolygon.from_radec(ra1, dec1)

    ra2 = ra2[::-1]
    dec2 = dec2[::-1]
    poly2 = polygon.SphericalPolygon.from_radec(ra2, dec2)

    assert poly1.intersects_poly(poly2)


def test_intersects_poly_fully_contained():
    ra1 = [-10, 10, 10, -10, -10]
    dec1 = [30, 30, 0, 0, 30]
    poly1 = polygon.SphericalPolygon.from_radec(ra1, dec1)

    ra2 = [-5, 5, 5, -5, -5]
    dec2 = [20, 20, 10, 10, 20]
    poly2 = polygon.SphericalPolygon.from_radec(ra2, dec2)

    assert poly1.intersects_poly(poly2)

    # Make sure it isn't order-dependent
    ra1 = ra1[::-1]
    dec1 = dec1[::-1]
    poly1 = polygon.SphericalPolygon.from_radec(ra1, dec1)

    ra2 = ra2[::-1]
    dec2 = dec2[::-1]
    poly2 = polygon.SphericalPolygon.from_radec(ra2, dec2)

    assert poly1.intersects_poly(poly2)


def test_hard_intersects_poly():
    ra1 = [-10, 10, 10, -10, -10]
    dec1 = [30, 30, 0, 0, 30]
    poly1 = polygon.SphericalPolygon.from_radec(ra1, dec1)

    ra2 = [-20, 20, 20, -20, -20]
    dec2 = [20, 20, 10, 10, 20]
    poly2 = polygon.SphericalPolygon.from_radec(ra2, dec2)

    assert poly1.intersects_poly(poly2)

    # Make sure it isn't order-dependent
    ra1 = ra1[::-1]
    dec1 = dec1[::-1]
    poly1 = polygon.SphericalPolygon.from_radec(ra1, dec1)

    ra2 = ra2[::-1]
    dec2 = dec2[::-1]
    poly2 = polygon.SphericalPolygon.from_radec(ra2, dec2)

    assert poly1.intersects_poly(poly2)


def test_not_intersects_poly():
    ra1 = [-10, 10, 10, -10, -10]
    dec1 = [30, 30, 5, 5, 30]
    poly1 = polygon.SphericalPolygon.from_radec(ra1, dec1)

    ra2 = [-20, 20, 20, -20, -20]
    dec2 = [-20, -20, -10, -10, -20]
    poly2 = polygon.SphericalPolygon.from_radec(ra2, dec2)

    assert not poly1.intersects_poly(poly2)

    # Make sure it isn't order-dependent
    ra1 = ra1[::-1]
    dec1 = dec1[::-1]
    poly1 = polygon.SphericalPolygon.from_radec(ra1, dec1)

    ra2 = ra2[::-1]
    dec2 = dec2[::-1]
    poly2 = polygon.SphericalPolygon.from_radec(ra2, dec2)

    assert not poly1.intersects_poly(poly2)


def test_point_in_poly():
    point = np.asarray([-0.27475449, 0.47588873, -0.83548781])
    points = np.asarray([[ 0.04821217, -0.29877206, 0.95310589],
                         [ 0.04451801, -0.47274119, 0.88007608],
                         [-0.14916503, -0.46369786, 0.87334649],
                         [-0.16101648, -0.29210164, 0.94273555],
                         [ 0.04821217, -0.29877206, 0.95310589]])
    inside = np.asarray([-0.03416009, -0.36858623, 0.9289657])
    poly = polygon.SphericalPolygon(points, inside)
    assert not poly.contains_point(point)


def test_point_in_poly_lots():
    import pyfits
    fits = pyfits.open(resolve_imagename(ROOT_DIR, '1904-66_TAN.fits'))
    header = fits[0].header

    poly1 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[0, 87])
    poly2 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[20, 89])
    poly3 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[180, 89])

    points = get_point_set()
    count = 0
    for point in points:
        if poly1.contains_point(point) or poly2.contains_point(point) or \
               poly3.contains_point(point):
            count += 1

    assert count == 5
    assert poly1.intersects_poly(poly2)
    assert not poly1.intersects_poly(poly3)
    assert not poly2.intersects_poly(poly3)


def test_great_circle_arc_intersection():
    A = [-10, -10]
    B = [10, 10]

    C = [-25, 10]
    D = [15, -10]

    E = [-20, 40]
    F = [20, 40]

    correct = [0.99912414, -0.02936109, -0.02981403]

    A = vector.radec_to_vector(*A)
    B = vector.radec_to_vector(*B)
    C = vector.radec_to_vector(*C)
    D = vector.radec_to_vector(*D)
    E = vector.radec_to_vector(*E)
    F = vector.radec_to_vector(*F)

    assert great_circle_arc.intersects(A, B, C, D)
    r = great_circle_arc.intersection(A, B, C, D)
    assert r.shape == (3,)
    assert_almost_equal(r, correct)

    assert np.all(great_circle_arc.intersects([A], [B], [C], [D]))
    r = great_circle_arc.intersection([A], [B], [C], [D])
    assert r.shape == (1, 3)
    assert_almost_equal(r, [correct])

    assert np.all(great_circle_arc.intersects([A], [B], C, D))
    r = great_circle_arc.intersection([A], [B], C, D)
    assert r.shape == (1, 3)
    assert_almost_equal(r, [correct])

    assert not np.all(great_circle_arc.intersects([A, E], [B, F], [C], [D]))
    r = great_circle_arc.intersection([A, E], [B, F], [C], [D])
    assert r.shape == (2, 3)
    assert_almost_equal(r[0], correct)
    assert np.all(np.isnan(r[1]))

    # Test parallel arcs
    r = great_circle_arc.intersection(A, B, A, B)
    assert np.all(np.isnan(r))


def test_great_circle_arc_length():
    A = [90, 0]
    B = [-90, 0]
    A = vector.radec_to_vector(*A)
    B = vector.radec_to_vector(*B)
    assert great_circle_arc.length(A, B) == 180.0

    A = [135, 0]
    B = [-90, 0]
    A = vector.radec_to_vector(*A)
    B = vector.radec_to_vector(*B)
    assert great_circle_arc.length(A, B) == 135.0

    A = [0, 0]
    B = [0, 90]
    A = vector.radec_to_vector(*A)
    B = vector.radec_to_vector(*B)
    assert great_circle_arc.length(A, B) == 90.0


def test_great_circle_arc_angle():
    A = [1, 0, 0]
    B = [0, 1, 0]
    C = [0, 0, 1]
    assert great_circle_arc.angle(A, B, C) == 90.0

    # TODO: More angle tests


def test_cone():
    random.seed(0)
    for i in range(50):
        ra = random.randrange(-180, 180)
        dec = random.randrange(20, 90)
        cone = polygon.SphericalPolygon.from_cone(ra, dec, 8, steps=64)


def test_area():
    triangles = [
        ([[90, 0], [0, -45], [0, 45], [90, 0]], np.pi * 0.5),
        ([[90, 0], [0, -22.5], [0, 22.5], [90, 0]], np.pi * 0.25),
        ([[90, 0], [0, -11.25], [0, 11.25], [90, 0]], np.pi * 0.125)
        ]

    for tri, area in triangles:
        tri = np.array(tri)
        x, y, z = vector.radec_to_vector(tri[:,1], tri[:,0])
        points = np.dstack((x, y, z))[0]
        poly = polygon.SphericalPolygon(points)
        calc_area = poly.area()
