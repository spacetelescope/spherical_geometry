from __future__ import print_function, absolute_import

# STDLIB
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

            areas = [x.area() for x in polys]

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
                print(union._points)
                print(permutation[0]._points)

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

                print(union_area, areas)
                assert np.all(union_area * 1.1 >= areas)

            lengths = np.array([len(x._points) for x in unions])
            assert np.all(lengths == [lengths[0]])
            areas = np.array([x.area() for x in unions])
            assert_array_almost_equal(areas, areas[0], 1)

        return run


@union_test(0, 90)
def test1():
    import pyfits
    fits = pyfits.open(resolve_imagename(ROOT_DIR, '1904-66_TAN.fits'))
    header = fits[0].header

    poly1 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[0, 87])
    poly2 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[20, 89])
    poly3 = polygon.SphericalPolygon.from_wcs(
        header, 1, crval=[175, 89])
    poly4 = polygon.SphericalPolygon.from_cone(
        90, 70, 10, steps=50)

    return [poly1, poly2, poly3, poly4]


@union_test(0, 90)
def test2():
    poly1 = polygon.SphericalPolygon.from_cone(0, 60, 7, steps=16)
    poly2 = polygon.SphericalPolygon.from_cone(0, 72, 7, steps=16)
    poly3 = polygon.SphericalPolygon.from_cone(20, 60, 7, steps=16)
    poly4 = polygon.SphericalPolygon.from_cone(20, 72, 7, steps=16)
    poly5 = polygon.SphericalPolygon.from_cone(35, 55, 7, steps=16)
    poly6 = polygon.SphericalPolygon.from_cone(60, 60, 3, steps=16)
    return [poly1, poly2, poly3, poly4, poly5, poly6]


@union_test(0, 90)
def test3():
    random.seed(0)
    polys = []
    for i in range(10):
        polys.append(polygon.SphericalPolygon.from_cone(
            random.randrange(-180, 180),
            random.randrange(20, 90),
            random.randrange(5, 16),
            steps=16))
    return polys


@union_test(0, 15)
def test4():
    random.seed(64)
    polys = []
    for i in range(10):
        polys.append(polygon.SphericalPolygon.from_cone(
            random.randrange(-30, 30),
            random.randrange(-15, 60),
            random.randrange(5, 16),
            steps=16))
    return polys


def test5():
    import pyfits
    import pywcs

    A = pyfits.open(os.path.join(ROOT_DIR, '2chipA.fits.gz'))

    wcs = pywcs.WCS(A[1].header, fobj=A)
    chipA1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(A[4].header, fobj=A)
    chipA2 = polygon.SphericalPolygon.from_wcs(wcs)

    null_union = chipA1.union(chipA2)


def test6():
    import pyfits
    import pywcs

    A = pyfits.open(os.path.join(ROOT_DIR, '2chipC.fits.gz'))

    wcs = pywcs.WCS(A[1].header, fobj=A)
    chipA1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(A[4].header, fobj=A)
    chipA2 = polygon.SphericalPolygon.from_wcs(wcs)

    null_union = chipA1.union(chipA2)


@union_test(0, 90)
def test7():
    import pyfits
    import pywcs

    A = pyfits.open(os.path.join(ROOT_DIR, '2chipA.fits.gz'))

    wcs = pywcs.WCS(A[1].header, fobj=A)
    chipA1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(A[4].header, fobj=A)
    chipA2 = polygon.SphericalPolygon.from_wcs(wcs)

    B = pyfits.open(os.path.join(ROOT_DIR, '2chipB.fits.gz'))

    wcs = pywcs.WCS(B[1].header, fobj=B)
    chipB1 = polygon.SphericalPolygon.from_wcs(wcs)
    wcs = pywcs.WCS(B[4].header, fobj=B)
    chipB2 = polygon.SphericalPolygon.from_wcs(wcs)

    return [chipA1, chipA2, chipB1, chipB2]


@union_test(0, 90)
def test8():
    import pyfits
    fits = pyfits.open(resolve_imagename(ROOT_DIR, '1904-66_TAN.fits'))
    header = fits[0].header

    poly1 = polygon.SphericalPolygon.from_wcs(
        header, 1)
    poly2 = polygon.SphericalPolygon.from_wcs(
        header, 1)

    return [poly1, poly2]


if __name__ == '__main__':
    if '--profile' not in sys.argv:
        GRAPH_MODE = True
        from mpl_toolkits.basemap import Basemap
        from matplotlib import pyplot as plt

    functions = [(k, v) for k, v in globals().items() if k.startswith('test')]
    functions.sort()
    for k, v in functions:
        v()
