from __future__ import print_function, absolute_import

# STDLIB
import codecs
import functools
import itertools
import math
import os
import random
import sys

# THIRD-PARTY
import numpy as np
from numpy.testing import assert_array_almost_equal

# LOCAL
from .. import polygon
from .test_shared import resolve_imagename

GRAPH_MODE = False
ROOT_DIR = os.path.join(os.path.dirname(__file__), 'data')


class union_test:
    def __init__(self, lon_0, lat_0, proj='ortho'):
        self._lon_0 = lon_0
        self._lat_0 = lat_0
        self._proj = proj

    def __call__(self, func):
        @functools.wraps(func)
        def run(*args, **kwargs):
            if GRAPH_MODE:
                from mpl_toolkits.basemap import Basemap
                from matplotlib import pyplot as plt

            polys = func(*args, **kwargs)

            unions = []
            num_permutations = math.factorial(len(polys))
            step_size = int(max(float(num_permutations) / 20.0, 1.0))
            if GRAPH_MODE:
                print("%d permutations" % num_permutations)

            areas = np.array([x.area() for x in polys])

            for i, permutation in enumerate(
                    itertools.islice(
                    itertools.permutations(polys),
                    None, None, step_size)):
                filename = '%s_union_%04d.svg' % (
                    func.__name__, i)
                print(filename)

                union = polygon.SphericalPolygon.multi_union(
                    permutation)
                unions.append(union)
                union_area = union.area()

                if GRAPH_MODE:
                    fig = plt.figure()
                    m = Basemap(projection=self._proj,
                                lon_0=self._lon_0,
                                lat_0=self._lat_0)
                    m.drawmapboundary(fill_color='white')
                    m.drawparallels(np.arange(-90., 90., 20.))
                    m.drawmeridians(np.arange(0., 420., 20.))

                    union.draw(m, color='red', linewidth=3)
                    for poly in permutation:
                        poly.draw(m, color='blue', alpha=0.5)
                    plt.savefig(filename)
                    fig.clear()

                assert np.all(union_area * 1.1 >= areas)

            lengths = np.array([
                np.sum(len(x._points) for x in y.iter_polygons_flat())
                for y in unions])
            assert np.all(lengths == [lengths[0]])
            areas = np.array([x.area() for x in unions])
            assert_array_almost_equal(areas, areas[0], 1)

        return run


@union_test(0, 90)
def test1():
    from astropy.io import fits
    fits = fits.open(resolve_imagename(ROOT_DIR, '1904-77_TAN.fits'))
    header = fits[0].header

    poly1 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[0, 87])
    poly2 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[20, 89])
    poly3 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[175, 89])
    poly4 = polygon.SphericalPolygon.from_cone(
        90, 70, 10, steps=8)

    return [poly1, poly2, poly3, poly4]


@union_test(0, 90)
def test2():
    poly1 = polygon.SphericalPolygon.from_cone(0, 60, 7, steps=8)
    poly2 = polygon.SphericalPolygon.from_cone(0, 72, 7, steps=8)
    poly3 = polygon.SphericalPolygon.from_cone(20, 60, 7, steps=8)
    poly4 = polygon.SphericalPolygon.from_cone(20, 72, 7, steps=8)
    poly5 = polygon.SphericalPolygon.from_cone(35, 55, 7, steps=8)
    poly6 = polygon.SphericalPolygon.from_cone(60, 60, 3, steps=8)
    return [poly1, poly2, poly3, poly4, poly5, poly6]


def test5():
    from astropy.io import fits
    from astropy import wcs as pywcs

    A = fits.open(os.path.join(ROOT_DIR, '2chipA.fits.gz'))

    wcs = pywcs.WCS(A[1].header, fobj=A)
    chipA1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(A[4].header, fobj=A)
    chipA2 = polygon.SphericalPolygon.from_wcs(wcs)

    null_union = chipA1.union(chipA2)


def test6():
    from astropy.io import fits
    from astropy import wcs as pywcs

    A = fits.open(os.path.join(ROOT_DIR, '2chipC.fits.gz'))

    wcs = pywcs.WCS(A[1].header, fobj=A)
    chipA1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(A[4].header, fobj=A)
    chipA2 = polygon.SphericalPolygon.from_wcs(wcs)

    null_union = chipA1.union(chipA2)


@union_test(0, 90)
def test7():
    from astropy.io import fits
    from astropy import wcs as pywcs

    A = fits.open(os.path.join(ROOT_DIR, '2chipA.fits.gz'))

    wcs = pywcs.WCS(A[1].header, fobj=A)
    chipA1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(A[4].header, fobj=A)
    chipA2 = polygon.SphericalPolygon.from_wcs(wcs)

    B = fits.open(os.path.join(ROOT_DIR, '2chipB.fits.gz'))

    wcs = pywcs.WCS(B[1].header, fobj=B)
    chipB1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(B[4].header, fobj=B)
    chipB2 = polygon.SphericalPolygon.from_wcs(wcs)

    return [chipA1, chipA2, chipB1, chipB2]


@union_test(0, 90)
def test8():
    from astropy.io import fits

    fits = fits.open(resolve_imagename(ROOT_DIR, '1904-66_TAN.fits'))
    header = fits[0].header

    poly1 = polygon.SphericalPolygon.from_wcs(
        header, 1)
    poly2 = polygon.SphericalPolygon.from_wcs(
        header, 1)

    return [poly1, poly2]


def test_union_empty():
    p = polygon.SphericalPolygon.from_cone(
        random.randrange(-180, 180),
        random.randrange(20, 90),
        random.randrange(5, 16),
        steps=16)

    p2 = p.union(polygon.SphericalPolygon([]))

    assert len(p2.polygons) == 1
    assert_array_almost_equal(p2.polygons[0].points, p.polygons[0].points)


def test_difficult_unions():
    # Tests a number of intersections of real data that have been
    # problematic in previous revisions of spherical_geometry

    fname = resolve_imagename(ROOT_DIR, "difficult_intersections.txt")
    with open(fname, 'rb') as fd:
        lines = fd.readlines()

    def to_array(line):
        x = np.frombuffer(codecs.decode(line.strip(), 'hex_codec'), dtype='<f8')
        return x.reshape((len(x) // 3, 3))

    polys = []
    for i in range(0, len(lines), 2):
        points, inside, = [
            to_array(line) for line in lines[i:i+2]]
        poly = polygon.SphericalPolygon(points, inside)
        polys.append(poly)

    polygon.SphericalPolygon.multi_union(polys[:len(polys)//2])


def test_inside_point():
    p = np.array(
        [[ 0.9990579 , -0.02407018,  0.03610999],
         [ 0.9990579 ,  0.02407018,  0.03610999],
         [ 0.9990579 ,  0.02407018, -0.03610999],
         [ 0.9990579 , -0.02407018, -0.03610999],
         [ 0.9990579 , -0.02407018,  0.03610999]])
    c = np.array([  1.00000000e+00,   0.00000000e+00,   6.12323400e-17])

    p1 = np.array(
        [[ 0.98452721,  0.16685201,  0.05354046],
         [ 0.9753416 ,  0.2141079 ,  0.05354046],
         [ 0.97657885,  0.2143484 , -0.01866853],
         [ 0.98576447,  0.1670925 , -0.01866853],
         [ 0.98452721,  0.16685201,  0.05354046]])
    c1 = np.array([ 0.98147768,  0.19077993,  0.01745241])

    p2 = np.array(
        [[ 0.99099541,  0.13258803,  0.01866853],
         [ 0.9834646 ,  0.18013571,  0.01866853],
         [ 0.9822197 ,  0.17993854, -0.05354046],
         [ 0.98975051,  0.13239086, -0.05354046],
         [ 0.99099541,  0.13258803,  0.01866853]])
    c2 = np.array([ 0.98753791,  0.15641064, -0.01745241])

    p3 = np.array(
        [[ 0.99501898,  0.09792202,  0.01866853],
         [ 0.98915214,  0.14570356,  0.01866853],
         [ 0.98790112,  0.14554995, -0.05354046],
         [ 0.99376796,  0.09776842, -0.05354046],
         [ 0.99501898,  0.09792202,  0.01866853]])
    c3 = np.array([ 0.99239498,  0.12185078, -0.01745241])

    p4 = np.array(
        [[ 0.98728923,  0.14964423,  0.05354046],
         [ 0.97892974,  0.19705323,  0.05354046],
         [ 0.98017101,  0.1972721 , -0.01866853],
         [ 0.98853049,  0.14986309, -0.01866853],
         [ 0.98728923,  0.14964423,  0.05354046]])
    c4 = np.array([ 0.98465776,  0.17362173,  0.01745241])

    testFoV = polygon.SphericalPolygon(p, inside=c)
    poly1 = polygon.SphericalPolygon(p1, inside=c1)
    poly2 = polygon.SphericalPolygon(p2, inside=c2)
    poly3 = polygon.SphericalPolygon(p3, inside=c3)
    poly4 = polygon.SphericalPolygon(p4, inside=c4)

    polys = [poly1, poly2, poly3, poly4]

    unionpoly = poly1.multi_union(polys)
    insides = list(unionpoly.inside)
    assert len(insides) == 1
    assert insides[0].shape == (3,)

    unionpoly2 = poly3.union(poly4)
    assert not testFoV.intersects_poly(unionpoly2)

    unionpoly3 = poly1.union(poly2)
    assert not testFoV.intersects_poly(unionpoly3)


if __name__ == '__main__':
    if '--profile' not in sys.argv:
        GRAPH_MODE = True
        from mpl_toolkits.basemap import Basemap
        from matplotlib import pyplot as plt

    functions = [(k, v) for k, v in globals().items() if k.startswith('test')]
    functions.sort()
    for k, v in functions:
        v()
